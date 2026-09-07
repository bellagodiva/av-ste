import logging
import os
import sys
import json
from argparse import Namespace

import numpy as np
import torch

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

config_path = Path(__file__).resolve().parent / "conf"


@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={"help": "noise wav file"})
    noise_prob: float = field(default=0, metadata={"help": "noise probability"})
    noise_snr: float = field(default=0, metadata={"help": "noise SNR in audio"})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={"help": "which modality to use"})
    data: Optional[str] = field(default=None, metadata={"help": "path to test data directory"})
    label_dir: Optional[str] = field(default=None, metadata={"help": "path to test label directory"})


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
    logger = logging.getLogger("mimi_rvq.infer")
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

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        with torch.no_grad():
            gen_timer.start()
            net_output = model(**sample["net_input"])
            # net_output is a dict; encoder_out shape: [T, B, num_rvq, V] (tbc=True)
            logits = net_output["encoder_out"].float()   # [T, B, num_rvq, V]
            pred = logits.argmax(dim=-1)                 # [T, B, num_rvq]
            gen_timer.stop(pred[..., 0].numel())

        if pred.dim() != 3:
            raise RuntimeError(f"Expected argmax output shape [T, B, num_rvq], got {tuple(pred.shape)}")

        pred = pred.permute(1, 0, 2).contiguous()   # [B, T, num_rvq]
        pad = net_output.get("encoder_padding_mask", None)

        num_rvq = pred.size(2)
        bsz = pred.size(0)

        for i in range(bsz):
            if "utt_id" in sample:
                utt_id = _to_python_utt_id(sample["utt_id"][i])
            elif "id" in sample:
                utt_id = _to_python_utt_id(sample["id"][i])
            else:
                utt_id = total_sequences

            pred_i = pred[i]   # [T, num_rvq]

            if pad is not None:
                valid_mask = ~pad[i]
                pred_tokens = pred_i[valid_mask].detach().cpu().tolist()   # list of [num_rvq]
            else:
                pred_tokens = pred_i.detach().cpu().tolist()

            # transpose to per-codebook lists: [[cb0_t0, cb0_t1, ...], [cb1_t0, ...], ...]
            per_codebook = [
                [pred_tokens[t][k] for t in range(len(pred_tokens))]
                for k in range(num_rvq)
            ]

            results.append({
                "utt_id": utt_id,
                "pred_tokens": per_codebook,   # list of num_rvq token-id lists
                "pred_len": len(pred_tokens),
                "num_rvq": num_rvq,
            })

            total_sequences += 1
            total_frames += len(pred_tokens)

        wps_meter.update(sum(r["pred_len"] for r in results[-bsz:]))
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
        "num_rvq": results[0]["num_rvq"] if results else None,
    }

    out_summary = os.path.join(cfg.common_eval.results_path, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved predictions to: {out_json}")
    logger.info(f"Saved summary to: {out_summary}")


@hydra.main(config_path=config_path, config_name="infer_mimi")
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
        cfg_name = get_args().config_name or "infer_mimi"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer_mimi"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()


if __name__ == "__main__":
    cli_main()
