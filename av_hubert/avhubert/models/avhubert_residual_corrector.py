# Causal AV-HuBERT Residual Corrector
#
# Architecture:
#   noisy audio + clean video
#     → CausalAVHubertModel (causal backbone)
#     → causal temporal downsample (25 Hz → 12.5 Hz)
#     → AVHubertResidualCorrector:
#         - takes AV-HuBERT features AND noisy Mimi logits as input
#         - predicts a logit correction (residual)
#         - final logits = noisy_logits + alpha * correction
#     → upsampled to 25 Hz for label alignment
#
# The noisy Mimi logits must be provided externally in the sample dict
# under key "noisy_logits" [B, T_25hz, V] at 25 Hz (already upsampled).

import sys
import logging
import contextlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING, OmegaConf

DBG = True if len(sys.argv) == 1 else False
if DBG:
    from avhubert_causal import CausalAVHubertModel, AVHubertConfig as CausalAVHubertConfig
else:
    from ..avhubert_causal import CausalAVHubertModel, AVHubertConfig as CausalAVHubertConfig

from .avhubert_mimi import MimiHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertResidualCorrectorConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING,
        metadata={"help": "path to pretrained AV-HuBERT checkpoint"},
    )
    no_pretrained_weights: bool = field(default=False)

    # dropouts
    dropout_input: float = field(default=0.0)
    final_dropout: float = field(default=0.0)
    dropout: float = field(default=0.0)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)

    # masking
    apply_mask: bool = field(default=False)
    mask_length: int = field(default=10)
    mask_prob: float = field(default=0.5)
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(default="static")
    mask_other: float = field(default=0.0)
    no_mask_overlap: bool = field(default=False)
    mask_channel_length: int = field(default=10)
    mask_channel_prob: float = field(default=0.0)
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(default="static")
    mask_channel_other: float = field(default=0.0)
    no_mask_channel_overlap: bool = field(default=False)

    freeze_finetune_updates: int = field(default=0)
    feature_grad_mult: float = field(default=1.0)
    layerdrop: float = field(default=0.0)
    lookahead_frames: int = field(
        default=4,
        metadata={"help": "future frames the backbone can attend to. 0=fully causal."},
    )

    # Mimi vocab
    mimi_vocab_size: int = field(default=2048)

    # Residual corrector dims
    proj_dim: int = field(
        default=256,
        metadata={"help": "intermediate projection dim inside the corrector"},
    )

    # key in sample dict holding noisy Mimi logits
    noisy_logits_key: str = field(
        default="noisy_logits",
        metadata={
            "help": (
                "Key in sample['net_input'] holding noisy Mimi cb0 logits. "
                "Shape: [B, T_25hz, mimi_vocab_size] at 25 Hz (upsampled). "
                "These are produced by running Mimi on noisy audio before training."
            )
        },
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    label_dir: str = II("task.label_dir")
    w2v_args: Any = None


# ---------------------------------------------------------------------------
# Residual Corrector module
# ---------------------------------------------------------------------------

class AVHubertResidualCorrector(nn.Module):
    """
    AV-HuBERT predicts a CORRECTION to noisy Mimi logits.

    Final prediction = noisy_logits + alpha * correction

    Analogous to residual learning in image denoising:
    instead of predicting the clean token from scratch,
    predict the adjustment to the noisy token distribution.

    Initialization near identity: correction_head weights/bias = 0,
    alpha = 0.5 → model starts as a small perturbation of noisy logits.
    """

    def __init__(self, d_model: int, mimi_vocab: int, proj_dim: int):
        super().__init__()
        self.noisy_dist_proj = nn.Linear(mimi_vocab, proj_dim)
        self.fusion_proj     = nn.Linear(d_model + proj_dim, d_model)

        # correction head — init near zero so training starts from noisy logits
        self.correction_head = nn.Linear(d_model, mimi_vocab)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

        # learned scalar controlling correction strength
        self.alpha = nn.Parameter(torch.tensor(0.5))

        nn.init.xavier_uniform_(self.noisy_dist_proj.weight)
        nn.init.zeros_(self.noisy_dist_proj.bias)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)

    def forward(self, av_features: torch.Tensor, noisy_logits: torch.Tensor) -> torch.Tensor:
        """
        av_features  : [B, T, d_model] — AV-HuBERT encoder output at 12.5 Hz
        noisy_logits : [B, T, V]       — Mimi(noisy_audio) cb0 logits at 12.5 Hz
        returns      : [B, T, V]       — corrected logits
        """
        # soft noisy distribution as context signal
        noisy_dist = F.softmax(noisy_logits, dim=-1)           # [B, T, V]
        dist_ctx   = self.noisy_dist_proj(noisy_dist)          # [B, T, proj_dim]

        # fuse AV features with noisy context
        fused      = torch.cat([av_features, dist_ctx], dim=-1) # [B, T, d_model+proj_dim]
        fused      = F.gelu(self.fusion_proj(fused))            # [B, T, d_model]

        # predict logit correction
        correction = self.correction_head(fused)               # [B, T, V]

        # residual: clean ≈ noisy + learned correction
        corrected_logits = noisy_logits + self.alpha * correction
        return corrected_logits

    @property
    def correction_weight(self) -> float:
        """Effective correction scale (useful for logging)."""
        return torch.sigmoid(self.alpha).item()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ResidualCorrectorEncoder(FairseqEncoder):
    """
    Wraps CausalAVHubertModel + AVHubertResidualCorrector.

    Expects sample['net_input'][noisy_logits_key] = [B, T_25hz, V]
    (noisy Mimi cb0 logits at 25 Hz, already upsampled by repeat_interleave).
    """

    def __init__(self, cfg: AVHubertResidualCorrectorConfig):
        self.apply_mask = cfg.apply_mask

        # load pretrained backbone
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        elif isinstance(w2v_args, Namespace):
            w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize

        w2v_args.task.data = cfg.data
        task_pretrain = tasks.setup_task(w2v_args.task)
        if "task_state" in state and state["task_state"] is not None:
            task_pretrain.load_state_dict(state["task_state"])

        default_model_cfg = OmegaConf.structured(CausalAVHubertConfig)
        OmegaConf.set_struct(default_model_cfg, False)
        merged_model_cfg = OmegaConf.merge(default_model_cfg, w2v_args.model)
        OmegaConf.set_struct(merged_model_cfg, False)
        merged_model_cfg.lookahead_frames = cfg.lookahead_frames

        causal_model = CausalAVHubertModel(
            merged_model_cfg, task_pretrain.cfg, task_pretrain.dictionaries
        )

        if not cfg.no_pretrained_weights:
            sd = state["model"].copy()
            sd.pop("mask_emb", None)
            sd.pop("label_embs_concat", None)
            sd.pop("final_proj.weight", None)
            sd.pop("final_proj.bias", None)
            remapped = {}
            for k, v in sd.items():
                new_k = k.replace(
                    "feature_extractor_video.resnet.frontend3D.0.weight",
                    "feature_extractor_video.resnet.frontend3D.0.conv.weight",
                )
                remapped[new_k] = v
            missing, unexpected = causal_model.load_state_dict(remapped, strict=False)
            logger.info(f"[ResidualCorrectorEncoder] missing={len(missing)}, unexpected={len(unexpected)}")

        causal_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model = causal_model
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
        self.noisy_logits_key = cfg.noisy_logits_key

        # causal temporal downsample: 25 Hz → 12.5 Hz
        self.temporal_downsample = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=2, stride=2, padding=0,
        )

        # residual corrector (replaces plain mimi_head)
        self.corrector = AVHubertResidualCorrector(
            d_model=d,
            mimi_vocab=cfg.mimi_vocab_size,
            proj_dim=cfg.proj_dim,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def _downsample_padding_mask(self, padding_mask, T_out):
        if padding_mask is None:
            return None
        B = padding_mask.size(0)
        left = padding_mask.new_zeros(B, 1)
        pm = torch.cat([left, padding_mask], dim=1)
        return pm[:, 1::2][:, :T_out]

    def _upsample_by_repeat(self, x, target_len):
        """[B, T, C] → [B, 2T, C], trimmed/padded to target_len."""
        x = x.repeat_interleave(2, dim=1)
        cur = x.size(1)
        if cur > target_len:
            x = x[:, :target_len]
        elif cur < target_len:
            last = x[:, -1:].expand(x.size(0), target_len - cur, x.size(2))
            x = torch.cat([x, last], dim=1)
        return x

    def _upsample_padding_mask(self, pm, target_len):
        if pm is None:
            return None
        pm = pm.repeat_interleave(2, dim=1)
        cur = pm.size(1)
        if cur > target_len:
            pm = pm[:, :target_len]
        elif cur < target_len:
            last = pm[:, -1:].expand(pm.size(0), target_len - cur)
            pm = torch.cat([pm, last], dim=1)
        return pm

    def forward(self, source, padding_mask, noisy_logits=None, tbc=True, **kwargs):
        """
        source        : dict with 'audio' and 'video' keys
        padding_mask  : [B, T]
        noisy_logits  : [B, T_25hz, V] — Mimi(noisy_audio) logits at 25 Hz
                        If None, falls back to zero correction (passthrough of noisy_logits).
        """
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(
                source=source,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training,
            )
        # x: [B, T, D] at 25 Hz

        orig_len = x.size(1)

        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]

        # causal temporal downsample → 12.5 Hz
        x = x.transpose(1, 2)          # [B, D, T]
        x = F.pad(x, (1, 0))           # [B, D, T+1]
        x = self.temporal_downsample(x) # [B, D, T/2]
        x = x.transpose(1, 2)          # [B, T/2, D]

        T_down = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)
        x = self.final_dropout(x)

        # downsample noisy_logits from 25 Hz → 12.5 Hz (take even frames)
        if noisy_logits is not None:
            noisy_logits_12 = noisy_logits[:, 0::2, :]              # [B, T/2, V]
            noisy_logits_12 = noisy_logits_12[:, :T_down, :]
        else:
            logger.warning("[ResidualCorrectorEncoder] noisy_logits not found in batch — "
                           "using zero logits as fallback (correction only).")
            noisy_logits_12 = x.new_zeros(x.size(0), T_down, self.corrector.correction_head.out_features)

        # apply residual corrector
        logits_12 = self.corrector(x, noisy_logits_12)              # [B, T/2, V]

        # upsample back to 25 Hz for label alignment
        logits = self._upsample_by_repeat(logits_12, target_len=orig_len)  # [B, T, V]
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)   # [T, B, V]

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             x,         # [B, T/2, D] at 12.5 Hz
            "correction_weight":    self.corrector.correction_weight,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        for key in ("encoder_out",):
            if encoder_out[key] is not None:
                encoder_out[key] = encoder_out[key].index_select(1, new_order)
        for key in ("encoder_padding_mask", "padding_mask"):
            if encoder_out.get(key) is not None:
                encoder_out[key] = encoder_out[key].index_select(0, new_order)
        if encoder_out.get("features") is not None:
            encoder_out["features"] = encoder_out["features"].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("av_hubert_residual_corrector", dataclass=AVHubertResidualCorrectorConfig)
class AVHubertResidualCorrectorModel(BaseFairseqModel):

    def __init__(self, cfg: AVHubertResidualCorrectorConfig, encoder: ResidualCorrectorEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

    @classmethod
    def build_model(cls, cfg: AVHubertResidualCorrectorConfig, task: FairseqTask):
        encoder = ResidualCorrectorEncoder(cfg)
        return cls(cfg, encoder)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.encoder.set_num_updates(num_updates)

    def get_normalized_probs(self, net_output, log_probs):
        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            logits[padding.T] = float("-inf")
        return logits

    def forward(self, source, padding_mask, noisy_logits=None, tbc=True, **kwargs):
        return self.encoder(source=source, padding_mask=padding_mask,
                            noisy_logits=noisy_logits, tbc=tbc, **kwargs)
