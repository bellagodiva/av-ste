import logging
import os
import sys
import json
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from fairseq import checkpoint_utils, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, List, Optional, Tuple, Union

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIMI_SR     = 24_000
MIMI_STRIDE = 1920


class MimiLogitExtractor:
    """Extracts cosine-sim logits [T, 2048] from a noisy wav path, matching
    the format produced by extract_noisy_mimi_logits.py."""

    def __init__(self, device):
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.eval()
        self.mimi    = mimi
        self.device  = device
        cb = mimi.quantizer.semantic_quantizer.vq.layers[0]._codebook.embedding.to(device)
        self.codebook = F.normalize(cb, dim=-1)  # [2048, D]

    @torch.no_grad()
    def extract(self, wav_path: str) -> torch.Tensor:
        """Returns [T_frames, 2048] float16 logits at 12.5 Hz."""
        wav, sr = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != MIMI_SR:
            wav = torchaudio.functional.resample(wav, sr, MIMI_SR)
        T   = wav.size(-1)
        pad = (MIMI_STRIDE - T % MIMI_STRIDE) % MIMI_STRIDE
        if pad > 0:
            wav = F.pad(wav, (0, pad))
        wav_in   = wav.unsqueeze(0).to(self.device)          # [1, 1, T]
        enc      = self.mimi.encoder(wav_in)                 # [1, 512, T_f]
        enc_proj = self.mimi.quantizer.semantic_quantizer.input_proj(enc)
        enc_proj = enc_proj.squeeze(0).T                     # [T_f, 256]
        enc_norm = F.normalize(enc_proj, dim=-1)
        logits   = enc_norm @ self.codebook.T                # [T_f, 2048]
        return logits.half()

config_path = Path(__file__).resolve().parent / "conf"


@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={"help": "noise wav file"})
    noise_prob: float = field(default=0, metadata={"help": "noise probability"})
    noise_snr: float = field(default=0, metadata={"help": "noise SNR in audio"})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={"help": "which modality to use"})
    data: Optional[str] = field(default=None, metadata={"help": "path to test data directory"})
    label_dir: Optional[str] = field(default=None, metadata={"help": "path to test label directory"})
    save_audio: bool = field(default=False, metadata={"help": "decode predicted Mimi tokens back to WAV and save under results_path/audio/"})


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(default=False)


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for inference!"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout)


def _to_python_utt_id(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x


def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("mimi.infer")
    if output_file is not sys.stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    use_cuda = torch.cuda.is_available()

    for model in models:
        model.eval()
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()

    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)

    logger.info(cfg)

    if cfg.common.seed is not None:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav

    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    for model in models:
        model.prepare_for_inference_(cfg)

    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    gen_timer = StopwatchMeter()
    wps_meter = TimeMeter()

    results = []
    total_sequences = 0
    total_frames = 0

    model = models[0]

    # On-the-fly noisy logit extraction when NOISY_LOGITS_ROOT is not set
    noisy_logits_root = os.environ.get("NOISY_LOGITS_ROOT", None)
    mimi_extractor = None
    if noisy_logits_root is None:
        logger.info("NOISY_LOGITS_ROOT not set — extracting noisy logits on-the-fly from audio")
        extractor_device = torch.device("cuda" if use_cuda else "cpu")
        mimi_extractor = MimiLogitExtractor(extractor_device)
    else:
        logger.info(f"Loading noisy logits from NOISY_LOGITS_ROOT={noisy_logits_root}")

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        # Inject noisy logits on-the-fly if not already loaded by the dataset
        if mimi_extractor is not None and sample["net_input"].get("noisy_logits") is None:
            audio_paths = sample.get("audio_path", None)
            if audio_paths is not None:
                logits_list = []
                for ap in audio_paths:
                    logits_list.append(mimi_extractor.extract(ap))  # [T, 2048]
                # pad to same length and stack [B, T_max, 2048]
                max_t = max(l.size(0) for l in logits_list)
                V = logits_list[0].size(1)
                noisy_logits = torch.zeros(len(logits_list), max_t, V,
                                           dtype=logits_list[0].dtype)
                for i, l in enumerate(logits_list):
                    noisy_logits[i, :l.size(0)] = l
                if use_cuda:
                    noisy_logits = noisy_logits.cuda()
                # upsample 12.5 Hz → 25 Hz to match AVHuBERT frame rate
                noisy_logits = noisy_logits.repeat_interleave(2, dim=1)
                sample["net_input"]["noisy_logits"] = noisy_logits

        with torch.no_grad():
            gen_timer.start()
            net_output = model(**sample["net_input"])
            logits = net_output["encoder_out"]   # T x B x V
            pred = logits.argmax(dim=-1)         # T x B
            gen_timer.stop(pred.numel())

        if pred.dim() != 2:
            raise RuntimeError(f"Expected argmax output shape [T, B], got {tuple(pred.shape)}")

        pred = pred.transpose(0, 1).contiguous()   # B x T
        pad = net_output.get("encoder_padding_mask", None)

        bsz = pred.size(0)

        for i in range(bsz):
            if "utt_id" in sample:
                utt_id = _to_python_utt_id(sample["utt_id"][i])
            elif "id" in sample:
                utt_id = _to_python_utt_id(sample["id"][i])
            else:
                utt_id = total_sequences

            pred_i = pred[i]

            if pad is not None:
                valid_mask = ~pad[i]
                pred_tokens = pred_i[valid_mask].detach().cpu().tolist()
            else:
                pred_tokens = pred_i.detach().cpu().tolist()

            results.append({
                "utt_id": utt_id,
                "pred_tokens": pred_tokens,
                "pred_len": len(pred_tokens),
            })

            total_sequences += 1
            total_frames += len(pred_tokens)

        wps_meter.update(sum(len(r["pred_tokens"]) for r in results[-bsz:]))
        progress.log({"frames/s": round(wps_meter.avg)})

    logger.info(
        "Processed {:,} utterances ({:,} predicted frames) in {:.1f}s ({:.2f} utt/s, {:.2f} frames/s)".format(
            total_sequences,
            total_frames,
            gen_timer.sum,
            total_sequences / gen_timer.sum if gen_timer.sum > 0 else 0.0,
            total_frames / gen_timer.sum if gen_timer.sum > 0 else 0.0,
        )
    )

    out_json = os.path.join(cfg.common_eval.results_path, "predictions.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "num_utterances": total_sequences,
        "num_predicted_frames": total_frames,
    }

    out_summary = os.path.join(cfg.common_eval.results_path, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved predictions to: {out_json}")
    logger.info(f"Saved summary to: {out_summary}")

    if cfg.override.save_audio:
        logger.info("Decoding predicted tokens to audio...")
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi_dec    = loaders.get_mimi(mimi_weight, device="cuda" if use_cuda else "cpu")
        mimi_dec.eval()

        audio_dir = os.path.join(cfg.common_eval.results_path, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        for r in results:
            utt_id     = r["utt_id"]
            pred_tokens = r["pred_tokens"]
            if not pred_tokens:
                continue

            tokens = torch.tensor(pred_tokens, dtype=torch.long,
                                  device="cuda" if use_cuda else "cpu")
            tokens = tokens.unsqueeze(0).unsqueeze(0)   # [1, 1, T]

            with torch.no_grad():
                audio_out = mimi_dec.decode(tokens)      # [1, 1, T_audio]

            safe_id  = str(utt_id).replace("/", "_").replace("\\", "_")
            wav_path = os.path.join(audio_dir, f"{safe_id}.wav")
            torchaudio.save(wav_path, audio_out.squeeze(0).cpu(), sample_rate=MIMI_SR)

        logger.info(f"Saved {len(results)} audio files → {audio_dir}")


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))


def cli_main() -> None:
    try:
        from hydra._internal.utils import get_args
        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()


if __name__ == "__main__":
    cli_main()