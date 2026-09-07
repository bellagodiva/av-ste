import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import metrics, utils

# lazy import — only needed when use_av_sync=True
_AVSyncPredictor = None
def _get_sync_cls():
    global _AVSyncPredictor
    if _AVSyncPredictor is None:
        from ..models.av_sync_predictor import AVSyncPredictor
        _AVSyncPredictor = AVSyncPredictor
    return _AVSyncPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MimiFrameCriterionConfig(FairseqDataclass):
    """
    Auxiliary loss flags:

        criterion:
          _name: mimi_mix_loss

          # CE only (baseline)
          use_ctc: false
          use_regression: false

          # CE + CTC
          use_ctc: true
          lambda_ctc: 0.3

          # CE + Regression with LEARNED token embeddings (old, circular)
          use_regression: true
          use_mimi_preq: false
          lambda_reg: 0.3

          # CE + Regression with REAL MiMi pre-quant features (recommended)
          use_regression: true
          use_mimi_preq: true       # ← key flag
          lambda_reg: 0.3
          mimi_preq_dim: 512        # MiMi encoder output dim before RVQ

          # CE + CTC + Regression (all three)
          use_ctc: true
          lambda_ctc: 0.3
          use_regression: true
          use_mimi_preq: true
          lambda_reg: 0.3

          # Architecture
          d_model: 1024
          mimi_vocab: 2048
          reg_proj_dim: 256
          mimi_preq_dim: 512

    When use_mimi_preq=true the criterion loads MiMi from HuggingFace hub at
    init time and runs its encoder (frozen, no grad) on clean audio during the
    forward pass.  The sample dict must contain clean audio under the key
    specified by clean_audio_key (default "clean_audio") at the top level of
    the sample, i.e. sample["clean_audio"] : Tensor [B, T_wav] at 24 kHz.
    """

    # ── CTC ───────────────────────────────────────────────────────────────
    use_ctc: bool = field(default=False)
    lambda_ctc: float = field(default=0.3)

    # ── Regression ────────────────────────────────────────────────────────
    use_regression: bool = field(default=False)
    lambda_reg: float = field(default=0.3)

    # ── Contrastive (InfoNCE) ─────────────────────────────────────────────
    use_contrastive: bool = field(
        default=False,
        metadata={
            "help": (
                "InfoNCE contrastive loss between AV-HuBERT features and MiMi "
                "pre-quant features from clean audio. Requires use_mimi_preq=True "
                "so that sample['clean_audio'] is in the batch."
            )
        },
    )
    lambda_con: float = field(default=0.1)
    con_temperature: float = field(
        default=0.07,
        metadata={"help": "InfoNCE temperature. Lower = harder negatives."},
    )
    con_proj_dim: int = field(
        default=256,
        metadata={"help": "Projection head output dim for contrastive loss."},
    )

    use_mimi_preq: bool = field(
        default=False,
        metadata={
            "help": (
                "When True: run MiMi encoder (frozen) on clean audio to get "
                "pre-quantization continuous features as the regression target. "
                "This is a FIXED external teacher — no circular co-adaptation. "
                "Requires sample['clean_audio'] in the batch. "
                "When False: use a jointly-trained token embedding (old behaviour)."
            )
        },
    )
    mimi_preq_dim: int = field(
        default=512,
        metadata={"help": "MiMi encoder output dimension before RVQ (pre-quant space)."},
    )
    clean_audio_key: str = field(
        default="clean_audio",
        metadata={
            "help": (
                "Key in the sample dict that holds clean waveforms for MiMi. "
                "Expected shape: [B, T_wav] at 24 kHz (MiMi's native rate). "
                "Add this to your dataset's collater."
            )
        },
    )

    # ── Predictive coding (Mamba model only) ─────────────────────────────
    lambda_pred_coding: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for the predictive coding aux loss returned by the model "
                "as net_output['pred_coding_loss']. 0.0 = disabled. "
                "Recommended: 0.1. Only used when the model produces this key."
            )
        },
    )

    # ── Knowledge Distillation (teacher = lookahead-4 model) ─────────────
    lambda_kd: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for KL-divergence KD loss from a pre-extracted teacher "
                "(lookahead=4) logit distribution. 0.0 = disabled. "
                "Recommended: 0.5. Requires TEACHER_LOGITS_ROOT env var and "
                "sample['teacher_logits'] in the batch."
            )
        },
    )
    kd_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for KD softening. Higher = softer distributions."},
    )

    # ── Speaker matching (target-speaker InfoNCE) ─────────────────────────
    use_speaker_match: bool = field(
        default=False,
        metadata={
            "help": (
                "InfoNCE matching loss between pooled AV-HuBERT speech representation "
                "and the target-speaker ArcFace face embedding. Pulls the predicted "
                "speech embedding toward its paired face embedding while pushing it "
                "away from other speakers in the batch. "
                "Requires ARCFACE_EMBED_ROOT env var and speaker_cond=pretrained_spk "
                "in the model config so that net_input['speaker_embed'] is populated."
            )
        },
    )
    lambda_match: float = field(
        default=0.1,
        metadata={"help": "Weight λ for the speaker matching loss. Eq. (2) in the paper."},
    )
    match_proj_dim: int = field(
        default=256,
        metadata={"help": "Shared projection dimension for the audio and visual branches."},
    )
    match_spk_dim: int = field(
        default=512,
        metadata={"help": "Dimensionality of the raw ArcFace embeddings (default 512)."},
    )
    match_temperature: float = field(
        default=0.07,
        metadata={"help": "InfoNCE temperature for the speaker matching loss."},
    )

    # ── AV Sync predictor loss ────────────────────────────────────────────
    use_av_sync: bool = field(
        default=False,
        metadata={
            "help": (
                "Frame-level AV sync loss using a frozen pre-trained AVSyncPredictor. "
                "The predictor compares the model's 12.5 Hz token distribution (via soft "
                "codebook lookup) against pre-extracted visual-only features, frame by frame. "
                "Requires: (1) AV_SYNC_CKPT env var pointing to the predictor checkpoint, "
                "(2) VISUAL_FEATS_ROOT env var pointing to visual-only .npy feature files. "
                "Pre-train the predictor with: python scripts/train_av_sync.py"
            )
        },
    )
    lambda_sync: float = field(
        default=0.1,
        metadata={"help": "Weight for the AV sync loss term."},
    )
    sync_logit_temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "Softmax temperature applied to logits_12p5 before the soft codebook "
                "lookup that bridges enhancement model output into sync predictor space. "
                "Lower = sharper (more like hard token), higher = smoother."
            )
        },
    )

    # ── Architecture ──────────────────────────────────────────────────────
    d_model: int = field(default=1024)
    mimi_vocab: int = field(default=2048)
    reg_proj_dim: int = field(default=256)


# ─────────────────────────────────────────────────────────────────────────────
# MiMi subprocess worker (needed because moshi requires Python >=3.10)
# ─────────────────────────────────────────────────────────────────────────────

def _write_mimi_worker(path: str):
    """Write the freeze-omni worker script to disk."""
    code = r'''#!/usr/bin/env python3
"""
Persistent MiMi encoder worker.
Reads [B,T] float32 arrays from stdin, writes [B,T_frames,D] float32 to stdout.
Wire format per message: 4-byte little-endian int32 ndim, ndim×4-byte shape, raw float32 data.
"""
import sys, struct, numpy as np

MIMI_STRIDE = 1920  # product of SEANet downsampling strides: 8x6x5x4

def main():
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    import torch

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda" if torch.cuda.is_available() else "cpu")
    mimi.eval()
    device = next(mimi.parameters()).device

    sys.stderr.write("READY\n")
    sys.stderr.flush()

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        header = stdin.read(4)
        if not header:
            break
        ndim  = struct.unpack('<i', header)[0]
        shape = struct.unpack(f'<{ndim}i', stdin.read(4 * ndim))
        data  = stdin.read(4 * int(np.prod(shape)))
        wav_np = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

        with torch.no_grad():
            wav = torch.from_numpy(wav_np).to(device).unsqueeze(1)  # [B, 1, T]

            # Pad T to a multiple of MIMI_STRIDE so SEANet stride assertion passes
            T = wav.shape[-1]
            pad = (MIMI_STRIDE - T % MIMI_STRIDE) % MIMI_STRIDE
            if pad > 0:
                wav = torch.nn.functional.pad(wav, (0, pad))

            preq = mimi.encoder(wav)      # [B, D, T_frames]
            preq = preq.transpose(1, 2)   # [B, T_frames, D]
            preq_np = preq.cpu().float().numpy()

        stdout.write(struct.pack('<i', preq_np.ndim))
        stdout.write(struct.pack(f'<{preq_np.ndim}i', *preq_np.shape))
        stdout.write(preq_np.tobytes())
        stdout.flush()

if __name__ == "__main__":
    main()
'''
    with open(path, "w") as f:
        f.write(code)


class _MimiWorkerHandle:
    """Wraps the freeze-omni subprocess; callable like mimi.encoder."""
    def __init__(self, proc):
        self._proc = proc

    def _read_exactly(self, n: int) -> bytes:
        """Read exactly n bytes from worker stdout, raising clearly if it dies."""
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                # worker died — collect its stderr for diagnosis
                err = self._proc.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"[MimiWorker] process died unexpectedly.\nstderr:\n{err}"
                )
            buf += chunk
        return buf

    def __call__(self, wav_np):
        import struct, numpy as np
        p = self._proc
        arr = wav_np.astype(np.float32)
        p.stdin.write(struct.pack('<i', arr.ndim))
        p.stdin.write(struct.pack(f'<{arr.ndim}i', *arr.shape))
        p.stdin.write(arr.tobytes())
        p.stdin.flush()

        ndim  = struct.unpack('<i', self._read_exactly(4))[0]
        shape = struct.unpack(f'<{ndim}i', self._read_exactly(4 * ndim))
        data  = self._read_exactly(4 * int(np.prod(shape)))
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

    def __del__(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Criterion
# ─────────────────────────────────────────────────────────────────────────────

@register_criterion("mimi_mix_loss", dataclass=MimiFrameCriterionConfig)
class MimiFrameCriterion(FairseqCriterion):
    """
    Frame-wise CE for Mimi semantic token prediction with optional CTC and
    regression auxiliary losses.

    Regression — two modes controlled by use_mimi_preq
    ────────────────────────────────────────────────────
    use_mimi_preq=False (old):
      reg_proj(features) ↔ token_embed(target_id)
      Both sides are learned jointly → circular, no fixed grounding.
      The loss can go down without features learning anything meaningful.

    use_mimi_preq=True (recommended):
      reg_proj(av_features) ↔ mimi_encoder(clean_audio)   [frozen]
      The MiMi encoder runs on clean audio at each training step.
      Its pre-quantization output is the fixed teacher.  AV-HuBERT features
      must move toward this fixed target — no circular co-adaptation.

      Why this works: MiMi's RVQ-1 pre-quant space was distilled from HuBERT
      during MiMi's own training.  AV-HuBERT features share the same HuBERT
      ancestry, so the spaces are already similar and regression is tractable.
    """

    def __init__(self, cfg: MimiFrameCriterionConfig, task):
        super().__init__(task)
        self.padding_idx      = -100
        self.use_ctc          = cfg.use_ctc
        self.lambda_ctc       = cfg.lambda_ctc
        self.use_regression   = cfg.use_regression
        self.lambda_reg       = cfg.lambda_reg
        self.use_mimi_preq    = cfg.use_mimi_preq
        self.clean_audio_key  = cfg.clean_audio_key
        self.use_contrastive      = cfg.use_contrastive
        self.lambda_con           = cfg.lambda_con
        self.con_temperature      = cfg.con_temperature
        self.lambda_pred_coding   = cfg.lambda_pred_coding
        self.lambda_kd            = cfg.lambda_kd
        self.kd_temperature       = cfg.kd_temperature

        # ── CTC head ──────────────────────────────────────────────────────
        if cfg.use_ctc:
            self.ctc_proj      = nn.Linear(cfg.d_model, cfg.mimi_vocab + 1)
            self.ctc_blank_idx = cfg.mimi_vocab
            nn.init.xavier_uniform_(self.ctc_proj.weight)
            nn.init.zeros_(self.ctc_proj.bias)

        # ── Regression heads ──────────────────────────────────────────────
        if cfg.use_regression:
            if cfg.use_mimi_preq:
                # project AV-HuBERT features to MiMi pre-quant dimension
                # target dimension is mimi_preq_dim (fixed external teacher)
                self.reg_proj = nn.Linear(cfg.d_model, cfg.reg_proj_dim)
                # adapter to align MiMi pre-quant dim to reg_proj_dim
                # (identity if they happen to match)
                self.mimi_adapter = nn.Linear(cfg.mimi_preq_dim, cfg.reg_proj_dim,
                                              bias=False)
                nn.init.xavier_uniform_(self.reg_proj.weight)
                nn.init.zeros_(self.reg_proj.bias)
                nn.init.eye_(self.mimi_adapter.weight if
                              cfg.mimi_preq_dim == cfg.reg_proj_dim
                              else self.mimi_adapter.weight)

                # load MiMi encoder from HF hub — frozen, teacher only
                self.mimi_encoder = self._load_mimi_encoder()

            else:  # noqa: E127
                # old behaviour: jointly-trained embedding as proxy target
                self.reg_proj    = nn.Linear(cfg.d_model, cfg.reg_proj_dim)
                self.token_embed = nn.Embedding(cfg.mimi_vocab, cfg.reg_proj_dim)
                nn.init.xavier_uniform_(self.reg_proj.weight)
                nn.init.zeros_(self.reg_proj.bias)
                nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        # ── AV Sync predictor (frozen) ────────────────────────────────────
        self.use_av_sync           = cfg.use_av_sync
        self.lambda_sync           = cfg.lambda_sync
        self.sync_logit_temperature = cfg.sync_logit_temperature
        self.sync_predictor        = None
        if cfg.use_av_sync:
            ckpt_path = os.environ.get("AV_SYNC_CKPT", "")
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    "use_av_sync=True requires AV_SYNC_CKPT env var pointing to "
                    "a pre-trained AVSyncPredictor checkpoint. "
                    "Train one with: python scripts/train_av_sync.py"
                )
            SyncCls = _get_sync_cls()
            self.sync_predictor = SyncCls.from_checkpoint(ckpt_path)
            self.sync_predictor.freeze()   # all params frozen, eval mode

        # ── Speaker matching projection heads ─────────────────────────────
        self.use_speaker_match  = cfg.use_speaker_match
        self.lambda_match       = cfg.lambda_match
        self.match_temperature  = cfg.match_temperature
        if cfg.use_speaker_match:
            # audio branch: pools AV-HuBERT hidden states → shared space
            self.match_proj_audio = nn.Sequential(
                nn.Linear(cfg.d_model,      cfg.match_proj_dim),
                nn.ReLU(),
                nn.Linear(cfg.match_proj_dim, cfg.match_proj_dim),
            )
            # visual branch: raw ArcFace embedding → shared space
            self.match_proj_visual = nn.Sequential(
                nn.Linear(cfg.match_spk_dim,  cfg.match_proj_dim),
                nn.ReLU(),
                nn.Linear(cfg.match_proj_dim, cfg.match_proj_dim),
            )
            for proj in (self.match_proj_audio, self.match_proj_visual):
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

        # ── Contrastive projection heads ──────────────────────────────────
        if cfg.use_contrastive:
            if not cfg.use_mimi_preq:
                raise ValueError(
                    "use_contrastive=True requires use_mimi_preq=True "
                    "(need frozen MiMi encoder to produce the positive target)."
                )
            # two separate MLPs: one per modality, shared projection dimension
            self.con_proj_av   = nn.Sequential(
                nn.Linear(cfg.d_model,    cfg.con_proj_dim),
                nn.ReLU(),
                nn.Linear(cfg.con_proj_dim, cfg.con_proj_dim),
            )
            self.con_proj_mimi = nn.Sequential(
                nn.Linear(cfg.mimi_preq_dim, cfg.con_proj_dim),
                nn.ReLU(),
                nn.Linear(cfg.con_proj_dim,  cfg.con_proj_dim),
            )
            # MiMi encoder shared with regression (load only once)
            if not cfg.use_regression:
                self.mimi_encoder = self._load_mimi_encoder()
            for proj in (self.con_proj_av, self.con_proj_mimi):
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────
    # MiMi encoder loader
    # ─────────────────────────────────────────────────────────────────────

    def _load_mimi_encoder(self):
        """
        Spawn a persistent freeze-omni (Python 3.10) worker that holds MiMi.
        moshi requires Python >=3.10 and cannot be imported in this env (3.8).
        Communication: send [B,T] float32 numpy arrays via stdin, receive
        [B,T_frames,D] float32 arrays via stdout.  Each message is:
          4 bytes little-endian int32 ndim, then ndim×4-byte int32 shape, then data.
        Returns a callable _MimiWorker handle (not a nn.Module).
        """
        import subprocess, struct, numpy as np, io

        _FREEZE_OMNI_PYTHON = "/home/bella/miniconda3/envs/freeze-omni/bin/python"
        _WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_mimi_worker.py")

        # Write the worker script if it doesn't exist
        if not os.path.exists(_WORKER_SCRIPT):
            _write_mimi_worker(_WORKER_SCRIPT)

        try:
            proc = subprocess.Popen(
                [_FREEZE_OMNI_PYTHON, _WORKER_SCRIPT],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # handshake: worker writes "READY\n" to stderr once MiMi is loaded
            while True:
                line = proc.stderr.readline().decode().strip()
                if line == "READY":
                    break
                if line:
                    print(f"[MimiWorker] {line}")
            print("[MimiCriterion] MiMi worker process ready.")
            return _MimiWorkerHandle(proc)
        except Exception as e:
            raise RuntimeError(
                f"[MimiCriterion] Failed to start MiMi worker: {e}\n"
                "Set use_mimi_preq=false to fall back to learned-embedding regression."
            )

    @torch.no_grad()
    def _get_mimi_preq_features(self, clean_audio, device):
        """
        Run MiMi's encoder on clean audio via the worker subprocess.

        clean_audio : [B, T_wav]  waveform at 24 kHz
        returns     : [B, T_frames, mimi_preq_dim]  at 12.5 Hz
        """
        import numpy as np
        wav_np = clean_audio.cpu().float().numpy()          # [B, T]
        preq_np = self.mimi_encoder(wav_np)                 # [B, T_frames, D]
        return torch.from_numpy(preq_np).to(device)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers (unchanged)
    # ─────────────────────────────────────────────────────────────────────

    def _pad_targets(self, target_list, device):
        proc = []
        for t in target_list:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.long, device=device)
            else:
                t = t.to(device=device, dtype=torch.long)
            if t.dim() == 2:
                if t.size(0) == 1:
                    t = t.squeeze(0)
                elif t.size(1) == 1:
                    t = t.squeeze(1)
                else:
                    raise ValueError(f"Unexpected target shape: {tuple(t.shape)}")
            elif t.dim() != 1:
                raise ValueError(f"Expected 1-D target per sample, got {tuple(t.shape)}")
            proc.append(t)
        T   = max(x.numel() for x in proc)
        out = torch.full((len(proc), T), self.padding_idx, dtype=torch.long, device=device)
        for i, t in enumerate(proc):
            out[i, : t.numel()] = t
        return out

    @staticmethod
    def _to_btv(logits, batch_size):
        if logits.dim() != 3:
            raise ValueError(f"Expected 3-D logits, got shape {tuple(logits.shape)}")
        if logits.size(1) == batch_size:
            return logits.transpose(0, 1)
        elif logits.size(0) == batch_size:
            return logits
        raise ValueError(
            f"Cannot infer logits layout: {tuple(logits.shape)}, B={batch_size}"
        )

    def _input_lengths_12hz(self, net_output, T_12, device):
        pm_25 = net_output.get("encoder_padding_mask")
        if pm_25 is None:
            B = net_output["encoder_out"].size(1)
            return torch.full((B,), T_12, dtype=torch.long, device=device)
        pm_12 = pm_25[:, 0::2][:, :T_12]
        return (~pm_12).sum(-1).clamp(min=1, max=T_12).long()

    # ─────────────────────────────────────────────────────────────────────
    # CTC loss (unchanged)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_ctc_loss(self, features_12hz, target_12hz, net_output):
        B, T_12, _ = features_12hz.shape
        ctc_logits = self.ctc_proj(features_12hz)
        log_probs  = F.log_softmax(ctc_logits, dim=-1)
        log_probs  = log_probs.transpose(0, 1).contiguous()
        input_lengths = self._input_lengths_12hz(net_output, T_12,
                                                  features_12hz.device)
        targets_flat   = []
        target_lengths = []
        for b in range(B):
            seq   = target_12hz[b]
            valid = seq[seq != self.padding_idx]
            if valid.numel() == 0:
                targets_flat.append(
                    torch.zeros(1, dtype=torch.long, device=seq.device)
                )
                target_lengths.append(1)
                continue
            keep    = torch.cat([
                torch.tensor([True], device=valid.device),
                valid[1:] != valid[:-1],
            ])
            deduped = valid[keep]
            max_tgt = max(int(input_lengths[b].item()) - 1, 1)
            deduped = deduped[:max_tgt]
            targets_flat.append(deduped)
            target_lengths.append(deduped.numel())
        targets        = torch.cat(targets_flat)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long,
                                      device=features_12hz.device)
        return F.ctc_loss(
            log_probs.float(),
            targets,
            input_lengths,
            target_lengths,
            blank         = self.ctc_blank_idx,
            reduction     = "mean",
            zero_infinity = True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Regression loss  ← KEY CHANGE IS HERE
    # ─────────────────────────────────────────────────────────────────────

    def _compute_regression_loss(self, features_12hz, target_12hz,
                                  clean_audio=None):
        """
        D-axis cosine regression.

        Two modes:

        use_mimi_preq=True  (recommended, fixed teacher)
        ─────────────────────────────────────────────────
          AV-HuBERT features [B, T, D]
              → reg_proj → [B, T, reg_proj_dim]       (trained)
          MiMi.encoder(clean_audio) [B, T, mimi_preq_dim]
              → mimi_adapter → [B, T, reg_proj_dim]   (trained)
          Loss = 1 − cosine_similarity per frame, mean over valid frames.

          The MiMi encoder is FROZEN.  Only reg_proj and mimi_adapter learn.
          This means AV-HuBERT features are pulled toward the actual MiMi
          pre-quantization space — a fixed, meaningful target.

        use_mimi_preq=False  (old, circular)
        ──────────────────────────────────────
          AV-HuBERT features → reg_proj ↔ token_embed(target_id)
          Both sides learn jointly — no fixed grounding.
        """
        # align time axis
        T = min(features_12hz.size(1), target_12hz.size(1))
        features_12hz = features_12hz[:, :T]
        target_12hz   = target_12hz[:, :T]

        valid_mask = target_12hz != self.padding_idx    # [B, T]
        if not valid_mask.any():
            return features_12hz.new_zeros(1).squeeze()

        feat_proj = self.reg_proj(features_12hz)        # [B, T, reg_proj_dim]

        if self.use_mimi_preq:
            # ── fixed teacher: MiMi pre-quant features ────────────────
            if clean_audio is None:
                raise ValueError(
                    "use_mimi_preq=True but clean_audio not found in sample. "
                    f"Add sample['{self.clean_audio_key}'] to your dataset collater."
                )
            # run frozen MiMi encoder on clean audio
            mimi_preq = self._get_mimi_preq_features(
                clean_audio, features_12hz.device
            )                                           # [B, T_mimi, mimi_preq_dim]

            # align frame count (MiMi is 12.5 Hz, same as features_12hz)
            T_reg = min(feat_proj.size(1), mimi_preq.size(1))
            feat_proj   = feat_proj[:, :T_reg]
            mimi_preq   = mimi_preq[:, :T_reg]
            valid_mask  = valid_mask[:, :T_reg]

            # project MiMi pre-quant to shared space
            # mimi_preq comes from subprocess as float32; cast to match model dtype
            mimi_preq = mimi_preq.to(dtype=feat_proj.dtype)
            target_proj = self.mimi_adapter(mimi_preq)  # [B, T, reg_proj_dim]

        else:
            # ── learned proxy: jointly trained token embeddings ───────
            valid_feats  = features_12hz[valid_mask]    # [N, D]
            valid_tokens = target_12hz[valid_mask]      # [N]
            feat_proj    = self.reg_proj(valid_feats)   # [N, reg_proj_dim]
            token_embs   = self.token_embed(valid_tokens)
            f_norm = F.normalize(feat_proj,  dim=-1)
            t_norm = F.normalize(token_embs, dim=-1)
            return 1.0 - (f_norm * t_norm).sum(-1).mean()

        # ── D-axis cosine over all valid frames ───────────────────────
        f_norm = F.normalize(feat_proj,  dim=-1)
        t_norm = F.normalize(target_proj, dim=-1)

        # apply valid mask — only compute over non-padded frames
        cos_sim = (f_norm * t_norm).sum(-1)             # [B, T]
        loss    = 1.0 - cos_sim[valid_mask].mean()
        return loss

    # ─────────────────────────────────────────────────────────────────────
    # Contrastive loss (InfoNCE)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_contrastive_loss(self, features_12hz, target_12hz, clean_audio):
        """
        InfoNCE loss between AV-HuBERT features and MiMi pre-quant features.

        Positives : (AV frame t, Mimi frame t from the same utterance)
        Negatives : all other frames in the batch (cross-sample and cross-time)

        features_12hz : [B, T, D]       — AV-HuBERT encoder output
        target_12hz   : [B, T]          — Mimi token ids (padding = -100)
        clean_audio   : [B, T_wav]      — clean waveform at 24 kHz

        Returns a scalar InfoNCE loss.
        """
        # align time axis
        T = min(features_12hz.size(1), target_12hz.size(1))
        features_12hz = features_12hz[:, :T]
        target_12hz   = target_12hz[:, :T]

        valid_mask = target_12hz != self.padding_idx   # [B, T]
        if not valid_mask.any():
            return features_12hz.new_zeros(1).squeeze()

        # get MiMi pre-quant features (reuse worker already warmed up)
        mimi_preq = self._get_mimi_preq_features(clean_audio, features_12hz.device)
        # [B, T_mimi, mimi_preq_dim]

        # align MiMi frame count
        T_con = min(features_12hz.size(1), mimi_preq.size(1))
        features_12hz = features_12hz[:, :T_con]
        mimi_preq     = mimi_preq[:, :T_con]
        valid_mask    = valid_mask[:, :T_con]

        # flatten to valid frames only: [N, D]
        av_valid   = features_12hz[valid_mask]   # [N, D_av]
        mi_valid   = mimi_preq[valid_mask]       # [N, D_mimi]
        mi_valid   = mi_valid.to(dtype=av_valid.dtype)

        # project both modalities to shared contrastive space
        av_proj = self.con_proj_av(av_valid)     # [N, con_proj_dim]
        mi_proj = self.con_proj_mimi(mi_valid)   # [N, con_proj_dim]

        # L2-normalize
        av_norm = F.normalize(av_proj, dim=-1)   # [N, C]
        mi_norm = F.normalize(mi_proj, dim=-1)   # [N, C]

        # similarity matrix [N, N] — diagonal = positives
        logits = torch.matmul(av_norm, mi_norm.T) / self.con_temperature  # [N, N]
        labels = torch.arange(logits.size(0), device=logits.device)

        # symmetric InfoNCE: av→mimi + mimi→av
        loss = 0.5 * (
            F.cross_entropy(logits,   labels) +
            F.cross_entropy(logits.T, labels)
        )
        return loss

    # ─────────────────────────────────────────────────────────────────────
    # AV Sync loss  (L_sync) — frozen predictor as a critic
    # ─────────────────────────────────────────────────────────────────────

    def _compute_av_sync_loss(
        self,
        logits_12p5: torch.Tensor,
        visual_feats: torch.Tensor,
        padding_mask_25hz,
    ) -> torch.Tensor:
        """
        Frame-level sync loss using the frozen AVSyncPredictor.

        How the bridge works
        ────────────────────
        The sync predictor was pre-trained with clean Mimi token IDs as the speech
        input (discrete → hard embedding lookup).  During enhancement training the
        model produces logits_12p5 [B, T, V] — a soft probability distribution over
        the same vocabulary.  We bridge these two spaces by:

            soft_emb = softmax(logits_12p5 / T) @ token_embed.weight   [B, T, C]

        This uses the predictor's own embedding table, so the distribution is the
        same as during pre-training when tokens are certain (peaked logits → hard
        lookup ≈ soft lookup).

        The frozen predictor then computes frame-level cosine similarity between
        the soft speech embedding and the visual-only features.  The enhancement
        model is rewarded for producing logit distributions that are in sync with
        the visible speaker's lip movements.

        logits_12p5    : [B, T, V]   model's 12.5 Hz token logits
        visual_feats   : [B, T, D]   pre-extracted visual-only features (VISUAL_FEATS_ROOT)
        padding_mask_25hz: [B, T_25] or None

        Returns scalar sync loss in [0, 2].  Typical value at convergence: ~0.1–0.3.
        """
        B, T, _ = logits_12p5.shape

        # derive 12.5 Hz padding mask
        if padding_mask_25hz is not None:
            pm_12 = padding_mask_25hz[:, 0::2][:, :T]   # [B, T]
        else:
            pm_12 = None

        # ── soft embedding bridge ─────────────────────────────────────────
        # softmax(logits / T) @ embed.weight  →  [B, T, sync_proj_dim]
        # The token_embed table is shared — no new parameters needed.
        soft_emb = self.sync_predictor.speech_enc.embed_soft(
            logits_12p5, temperature=self.sync_logit_temperature
        )   # [B, T, C]  gradient flows through logits_12p5

        # align visual feats to token frame count
        T_vis = visual_feats.size(1)
        if T_vis > T:
            visual_feats = visual_feats[:, :T, :]
        elif T_vis < T:
            pad = visual_feats[:, -1:].expand(B, T - T_vis, -1)
            visual_feats = torch.cat([visual_feats, pad], dim=1)

        # ── frozen predictor forward ──────────────────────────────────────
        # is_tokens=False: soft_emb is already in embedding space, skip embed()
        s_ctx, v_ctx = self.sync_predictor.encode(
            soft_emb, visual_feats.to(soft_emb.dtype),
            is_tokens=False, padding_mask=pm_12,
        )
        frame_sync = self.sync_predictor.frame_sync_scores(s_ctx, v_ctx)  # [B, T]

        # ── loss: maximize sync → minimize (1 - sync) ────────────────────
        if pm_12 is not None:
            valid = ~pm_12
            if not valid.any():
                return logits_12p5.new_zeros(1).squeeze()
            loss = 1.0 - frame_sync[valid].mean()
        else:
            loss = 1.0 - frame_sync.mean()

        return loss

    # ─────────────────────────────────────────────────────────────────────
    # Speaker matching loss  (L_match)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_speaker_match_loss(
        self,
        features_12hz: torch.Tensor,
        padding_mask_25hz,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE loss between the pooled multimodal speech representation **a**
        and the target-face visual embedding **v_face**.

        Positives  : (a_i, v_face_i) — same utterance / speaker
        Negatives  : all other speakers in the batch (cross-sample)

        features_12hz  : [B, T, D]   — AV-HuBERT hidden states at 12.5 Hz
        padding_mask_25hz: [B, T_25] or None — True = padded (invalid)
        speaker_embed  : [B, match_spk_dim]  — raw ArcFace face embeddings

        Returns a scalar loss.
        """
        B, T, _ = features_12hz.shape

        # derive 12.5 Hz padding mask from the 25 Hz one (take every 2nd frame)
        if padding_mask_25hz is not None:
            pm_12 = padding_mask_25hz[:, 0::2][:, :T]   # [B, T]
            valid  = ~pm_12                               # [B, T]
            denom  = valid.float().sum(1, keepdim=True).clamp(min=1)  # [B, 1]
            audio_pool = (features_12hz * valid.unsqueeze(-1).float()).sum(1) / denom
        else:
            audio_pool = features_12hz.mean(1)            # [B, D]

        # project to shared contrastive space
        a = self.match_proj_audio(audio_pool)                        # [B, match_proj_dim]
        v = self.match_proj_visual(speaker_embed.to(a.dtype))        # [B, match_proj_dim]

        # L2 normalize
        a = F.normalize(a, dim=-1)   # [B, C]
        v = F.normalize(v, dim=-1)   # [B, C]

        # symmetric InfoNCE over the batch
        # sim[i, j] = similarity between speech-i and face-j
        sim = torch.matmul(a, v.T) / self.match_temperature          # [B, B]
        labels = torch.arange(B, device=sim.device)
        loss = 0.5 * (
            F.cross_entropy(sim,   labels) +   # audio → visual
            F.cross_entropy(sim.T, labels)     # visual → audio
        )
        return loss

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        raw_logits = net_output["encoder_out"]

        # ── parse targets ─────────────────────────────────────────────────
        raw = sample["target_list"]
        if not isinstance(raw, list):
            raise TypeError(f"target_list must be a list, got {type(raw)}")

        if len(raw) == 1 and isinstance(raw[0], list):
            target = self._pad_targets(raw[0], raw_logits.device)
        elif len(raw) == 1 and torch.is_tensor(raw[0]):
            target = raw[0].to(device=raw_logits.device, dtype=torch.long)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            elif target.dim() != 2:
                raise ValueError(f"Expected [B,T] target, got {tuple(target.shape)}")
        else:
            assert all(
                (torch.is_tensor(r) and r.dim() <= 2)
                or isinstance(r, (list, torch.Tensor))
                for r in raw
            )
            target = self._pad_targets(raw, raw_logits.device)

        B = target.size(0)

        logits = self._to_btv(raw_logits, B)
        _, T, V = logits.shape
        Tt = target.size(1)
        if Tt > T:
            target = target[:, :T]
        elif Tt < T:
            pad    = torch.full((B, T - Tt), self.padding_idx,
                                dtype=target.dtype, device=target.device)
            target = torch.cat([target, pad], dim=1)

        # ── CE loss (always active) ────────────────────────────────────────
        logits_flat = logits.reshape(B * T, V)
        target_flat = target.reshape(B * T)
        loss_ce = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index = self.padding_idx,
            reduction    = "sum" if reduce else "none",
        )
        non_pad_mask = target_flat != self.padding_idx
        sample_size  = non_pad_mask.sum().item()

        total_loss     = loss_ce
        loss_ctc_val   = 0.0
        loss_reg_val   = 0.0
        loss_con_val   = 0.0
        loss_pred_val  = 0.0
        loss_kd_val    = 0.0
        loss_match_val = 0.0
        loss_sync_val  = 0.0

        # ── auxiliary losses (training only) ──────────────────────────────
        if self.training and (self.use_ctc or self.use_regression or self.use_contrastive):
            features_12hz = net_output.get("features")   # [B, T/2, D]

            if features_12hz is not None:
                target_12hz  = target[:, 0::2]            # [B, T/2]
                n_valid_12hz = (target_12hz != self.padding_idx).sum().item()

                # fetch clean audio once (shared by regression + contrastive)
                clean_audio = None
                if self.use_mimi_preq and (self.use_regression or self.use_contrastive):
                    clean_audio = sample.get(self.clean_audio_key)
                    if clean_audio is None:
                        clean_audio = sample.get("net_input", {}).get(
                            self.clean_audio_key
                        )

                if n_valid_12hz > 0:
                    if self.use_ctc:
                        loss_ctc   = self._compute_ctc_loss(
                            features_12hz, target_12hz, net_output
                        )
                        total_loss   = total_loss + self.lambda_ctc * loss_ctc * n_valid_12hz
                        loss_ctc_val = loss_ctc.item()

                    if self.use_regression:
                        loss_reg   = self._compute_regression_loss(
                            features_12hz, target_12hz,
                            clean_audio=clean_audio,
                        )
                        total_loss   = total_loss + self.lambda_reg * loss_reg * n_valid_12hz
                        loss_reg_val = loss_reg.item()

                    if self.use_contrastive:
                        loss_con   = self._compute_contrastive_loss(
                            features_12hz, target_12hz,
                            clean_audio=clean_audio,
                        )
                        total_loss   = total_loss + self.lambda_con * loss_con * n_valid_12hz
                        loss_con_val = loss_con.item()

        # ── AV sync loss: L_sync ─────────────────────────────────────────
        # Frozen sync predictor scores how well the model's token distribution
        # aligns frame-by-frame with the target speaker's lip movements.
        # Gradient flows through logits_12p5 → soft embed → cross-attention score.
        if self.training and self.use_av_sync and self.sync_predictor is not None:
            logits_12p5  = net_output.get("logits_12p5")          # [B, T/2, V]
            visual_feats = sample.get("net_input", {}).get("visual_feats")  # [B, T/2, D]
            if logits_12p5 is not None and visual_feats is not None:
                pm_25 = net_output.get("encoder_padding_mask")
                loss_sync = self._compute_av_sync_loss(
                    logits_12p5, visual_feats, pm_25
                )
                total_loss    = total_loss + self.lambda_sync * loss_sync * sample_size
                loss_sync_val = loss_sync.item()

        # ── speaker matching loss: L_match ───────────────────────────────
        # InfoNCE between pooled speech representation and target-face embedding.
        # Scaled by sample_size so its gradient magnitude stays proportional to
        # the CE loss (both are per-token quantities after normalisation).
        if self.training and self.use_speaker_match:
            features_12hz_sm = net_output.get("features")        # [B, T/2, D]
            speaker_embed_sm = sample.get("net_input", {}).get("speaker_embed")
            if features_12hz_sm is not None and speaker_embed_sm is not None and B > 1:
                pm_25 = net_output.get("encoder_padding_mask")   # [B, T_25] or None
                loss_match = self._compute_speaker_match_loss(
                    features_12hz_sm, pm_25, speaker_embed_sm
                )
                total_loss     = total_loss + self.lambda_match * loss_match * sample_size
                loss_match_val = loss_match.item()

        # ── predictive coding aux loss (model-side) ──────────────────────
        # pred_coding_loss is already a per-step mean (scalar), NOT a sum,
        # so we must NOT multiply by sample_size — that would inflate the loss
        # by ~1000x and overwhelm the CE gradient.
        if self.training and self.lambda_pred_coding > 0.0:
            pred_coding_loss = net_output.get("pred_coding_loss")
            if pred_coding_loss is not None:
                total_loss    = total_loss + self.lambda_pred_coding * pred_coding_loss
                loss_pred_val = pred_coding_loss.item()

        # ── KD loss: KL(student || teacher) ──────────────────────────────
        if self.training and self.lambda_kd > 0.0:
            teacher_logits = sample.get("teacher_logits")  # [B, T_25hz, V]
            if teacher_logits is not None:
                T_kd = min(logits.size(1), teacher_logits.size(1))
                stu = logits[:, :T_kd, :]           # [B, T_kd, V]
                tch = teacher_logits[:, :T_kd, :].to(device=stu.device, dtype=stu.dtype)
                tgt_kd = target[:, :T_kd]           # [B, T_kd]
                valid_kd = tgt_kd != self.padding_idx  # [B, T_kd]
                if valid_kd.any():
                    T_kd_val = self.kd_temperature
                    log_p = F.log_softmax(stu[valid_kd] / T_kd_val, dim=-1)
                    q     = F.softmax(tch[valid_kd]     / T_kd_val, dim=-1)
                    # KL(student || teacher) = sum q * (log q - log p)
                    loss_kd = F.kl_div(log_p, q, reduction="batchmean") * (T_kd_val ** 2)
                    total_loss  = total_loss + self.lambda_kd * loss_kd * sample_size
                    loss_kd_val = loss_kd.item()

        # ── logging ───────────────────────────────────────────────────────
        logging_output = {
            "loss":        total_loss.detach().item() if torch.is_tensor(total_loss) else float(total_loss),
            "loss_ce":     loss_ce.detach().item()    if torch.is_tensor(loss_ce)    else float(loss_ce),
            "loss_ctc":    loss_ctc_val,
            "loss_reg":    loss_reg_val,
            "loss_con":    loss_con_val,
            "loss_pred":   loss_pred_val,
            "loss_kd":     loss_kd_val,
            "loss_sync":   loss_sync_val,
            "loss_match":  loss_match_val,
            "ntokens":     sample_size,
            "nsentences":  B,
            "sample_size": sample_size,
        }

        with torch.no_grad():
            pred_flat  = logits_flat.argmax(dim=-1)
            correct_25 = ((pred_flat == target_flat) & non_pad_mask).sum().item()
            total_25   = non_pad_mask.sum().item()

            pred_2d    = pred_flat.reshape(B, T)
            target_2d  = target.reshape(B, T)
            pred_12    = pred_2d[:, 0::2].reshape(-1)
            tgt_12     = target_2d[:, 0::2].reshape(-1)
            non_pad_12 = tgt_12 != self.padding_idx
            correct_12 = ((pred_12 == tgt_12) & non_pad_12).sum().item()
            total_12   = non_pad_12.sum().item()

            logging_output["correct"]    = correct_12
            logging_output["total"]      = total_12
            logging_output["correct_25"] = correct_25
            logging_output["total_25"]   = total_25

        return total_loss, sample_size, logging_output

    # ─────────────────────────────────────────────────────────────────────
    # Metric aggregation
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum     = sum(log.get("loss",     0) for log in logging_outputs)
        loss_ce_sum  = sum(log.get("loss_ce",  0) for log in logging_outputs)
        loss_ctc_sum = sum(log.get("loss_ctc", 0) for log in logging_outputs)
        loss_reg_sum  = sum(log.get("loss_reg",  0) for log in logging_outputs)
        loss_con_sum  = sum(log.get("loss_con",  0) for log in logging_outputs)
        loss_pred_sum = sum(log.get("loss_pred", 0) for log in logging_outputs)
        loss_kd_sum    = sum(log.get("loss_kd",    0) for log in logging_outputs)
        loss_sync_sum  = sum(log.get("loss_sync",  0) for log in logging_outputs)
        loss_match_sum = sum(log.get("loss_match", 0) for log in logging_outputs)
        sample_size  = sum(log.get("sample_size", 0) for log in logging_outputs)
        correct_12   = sum(log.get("correct",    0) for log in logging_outputs)
        total_12     = sum(log.get("total",      0) for log in logging_outputs)
        correct_25   = sum(log.get("correct_25", 0) for log in logging_outputs)
        total_25     = sum(log.get("total_25",   0) for log in logging_outputs)
        n            = len(logging_outputs)

        if torch.is_tensor(loss_sum):
            loss_sum = utils.item(loss_sum)

        if sample_size > 0:
            metrics.log_scalar("loss",    loss_sum    / sample_size, sample_size, round=6)
            metrics.log_scalar("loss_ce", loss_ce_sum / sample_size, sample_size, round=6)
        if n > 0 and loss_ctc_sum > 0:
            metrics.log_scalar("loss_ctc", loss_ctc_sum / n, n, round=6)
        if n > 0 and loss_reg_sum > 0:
            metrics.log_scalar("loss_reg", loss_reg_sum / n, n, round=6)
        if n > 0 and loss_con_sum > 0:
            metrics.log_scalar("loss_con",  loss_con_sum  / n, n, round=6)
        if n > 0 and loss_pred_sum > 0:
            metrics.log_scalar("loss_pred", loss_pred_sum / n, n, round=6)
        if n > 0 and loss_kd_sum > 0:
            metrics.log_scalar("loss_kd",    loss_kd_sum    / n, n, round=6)
        if n > 0 and loss_sync_sum > 0:
            metrics.log_scalar("loss_sync",  loss_sync_sum  / n, n, round=6)
        if n > 0 and loss_match_sum > 0:
            metrics.log_scalar("loss_match", loss_match_sum / n, n, round=6)
        if total_12 > 0:
            metrics.log_scalar("accuracy",      correct_12 / total_12, total_12, round=6)
        if total_25 > 0:
            metrics.log_scalar("accuracy_25hz", correct_25 / total_25, total_25, round=6)