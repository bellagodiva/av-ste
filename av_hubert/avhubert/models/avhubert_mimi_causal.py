# Causal AV-HuBERT for noisy audio + clean video → clean Mimi semantic token prediction.
#
# Architecture:
#   noisy audio + clean video
#     → CausalAVHubertModel (causal backbone, init from pretrained bidirectional weights)
#     → causal temporal downsample Conv1d (25 Hz → 12.5 Hz)
#     → Mimi head (linear/MLP)
#     → clean Mimi codebook-0 token logits at 25 Hz (upsampled for label alignment)
#
# Weight init:
#   Pretrained bidirectional AV-HuBERT weights load into CausalAVHubertModel with
#   strict=False. All weight shapes are identical — causality is enforced purely by
#   attention masking (CausalTransformerEncoder) and left-padding (CausalConv3d),
#   not by changing parameter shapes.

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

from .avhubert_mimi import MimiHead, Linear

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertMimiCausalConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING,
        metadata={"help": "path to pretrained AV-HuBERT checkpoint (bidirectional or causal)"},
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, do not load pretrained weights (train from scratch)"},
    )

    # dropouts forwarded to the backbone
    dropout_input: float = field(default=0.0)
    final_dropout: float = field(default=0.0)
    dropout: float = field(default=0.0)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)

    # masking (kept for API compat, apply_mask=False at fine-tune time)
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

    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "freeze backbone for this many updates"},
    )
    feature_grad_mult: float = field(default=1.0)
    layerdrop: float = field(default=0.0)
    lookahead_frames: int = field(
        default=0,
        metadata={"help": "future frames the backbone transformer can attend to. "
                          "0=fully causal (80ms latency). Each +1 adds 40ms. "
                          "Recommended: 0 (streaming), 2 (160ms), 4 (240ms)."},
    )

    # Mimi head
    mimi_vocab_size: int = field(
        default=2048,
        metadata={"help": "Mimi codebook size"},
    )
    head_hidden_dim: int = field(
        default=0,
        metadata={"help": "0 = linear head, >0 = MLP with this hidden dim"},
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    label_dir: str = II("task.label_dir")
    w2v_args: Any = None


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CausalNoisyEncoder(FairseqEncoder):
    """
    Wraps CausalAVHubertModel to accept noisy audio + clean video and
    predict clean Mimi semantic tokens.
    """

    def __init__(self, cfg: AVHubertMimiCausalConfig):
        self.apply_mask = cfg.apply_mask

        # ----- load pretrained checkpoint -----
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

        assert cfg.normalize == w2v_args.task.normalize, (
            "normalize mismatch between pretrain and finetune configs"
        )

        w2v_args.task.data = cfg.data
        task_pretrain = tasks.setup_task(w2v_args.task)
        if "task_state" in state and state["task_state"] is not None:
            task_pretrain.load_state_dict(state["task_state"])

        # Merge checkpoint model config with the full CausalAVHubertConfig defaults.
        # This fills in any fields that are missing from the checkpoint (added after
        # it was saved) with their correct default values instead of None.
        default_model_cfg = OmegaConf.structured(CausalAVHubertConfig)
        OmegaConf.set_struct(default_model_cfg, False)
        merged_model_cfg = OmegaConf.merge(default_model_cfg, w2v_args.model)
        OmegaConf.set_struct(merged_model_cfg, False)
        merged_model_cfg.lookahead_frames = cfg.lookahead_frames

        # Build the causal backbone with fully-defaulted config
        causal_model = CausalAVHubertModel(
            merged_model_cfg, task_pretrain.cfg, task_pretrain.dictionaries
        )

        if not cfg.no_pretrained_weights:
            sd = state["model"].copy()
            # Remove pretraining-only heads — shapes differ or are unused
            sd.pop("mask_emb", None)
            sd.pop("label_embs_concat", None)
            sd.pop("final_proj.weight", None)
            sd.pop("final_proj.bias", None)

            # Key remapping: pretrained bidirectional model uses Conv3d in frontend3D;
            # CausalResEncoder expects CausalConv3d (same weight, different padding).
            remapped = {}
            for k, v in sd.items():
                new_k = k.replace(
                    "feature_extractor_video.resnet.frontend3D.0.weight",
                    "feature_extractor_video.resnet.frontend3D.0.conv.weight",
                )
                remapped[new_k] = v

            missing, unexpected = causal_model.load_state_dict(remapped, strict=False)
            logger.info(f"[CausalNoisyEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")
            if missing:
                logger.info(f"  missing keys (first 10): {missing[:10]}")

        causal_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model = causal_model
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        # Causal temporal downsample: left-pad by 1 so output[k] = conv(x[2k-1], x[2k])
        self.temporal_downsample = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.mimi_head = MimiHead(
            in_dim=d,
            out_dim=cfg.mimi_vocab_size,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.final_dropout,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def _downsample_padding_mask(self, padding_mask, T_out):
        if padding_mask is None:
            return None
        B = padding_mask.size(0)
        left = padding_mask.new_zeros(B, 1)
        pm = torch.cat([left, padding_mask], dim=1)   # left-pad mirrors conv left-pad
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

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        """
        source: dict with keys:
            'audio': noisy filterbank features [B, T*stack, audio_feat_dim]
            'video': clean lip frames [B, 1, T, H, W]
        padding_mask: [B, T]
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

        # ensure even length for stride-2 conv
        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]

        # causal temporal downsample: left-pad by 1 in time dim
        x = x.transpose(1, 2)          # [B, D, T]
        x = F.pad(x, (1, 0))           # [B, D, T+1]
        x = self.temporal_downsample(x) # [B, D, T/2]
        x = x.transpose(1, 2)          # [B, T/2, D]

        T_down = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        x = self.final_dropout(x)
        logits_12p5 = self.mimi_head(x)   # [B, T/2, V]

        # upsample to 25 Hz to align with duplicated labels
        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)   # [T, B, V]

        return {
            "encoder_out": logits,
            "encoder_padding_mask": out_pm,
            "padding_mask": out_pm,
            "features": x,                    # [B, T/2, D] — for routing agreement score
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        for key in ("encoder_out",):
            if encoder_out[key] is not None:
                encoder_out[key] = encoder_out[key].index_select(1, new_order)
        for key in ("encoder_padding_mask", "padding_mask"):
            if encoder_out[key] is not None:
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

@register_model("av_hubert_mimi_causal", dataclass=AVHubertMimiCausalConfig)
class AVHubertMimiCausal(BaseFairseqModel):
    def __init__(self, cfg: AVHubertMimiCausalConfig, encoder: CausalNoisyEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder

    @classmethod
    def build_model(cls, cfg: AVHubertMimiCausalConfig, task: FairseqTask):
        encoder = CausalNoisyEncoder(cfg)
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
            padding = padding.T
            logits[padding] = float("-inf")
        return logits

    def forward(self, **kwargs):
        return self.encoder(**kwargs)
