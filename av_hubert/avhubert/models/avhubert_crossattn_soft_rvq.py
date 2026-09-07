# Cross-attention AV-HuBERT predicting all 8 Mimi RVQ codebooks.
#
# Identical backbone + soft cross-attention fuser as avhubert_crossattn_soft.py,
# but the prediction head is replaced with num_rvq independent MimiHead modules
# (one per codebook), each producing logits [B, T, V].
#
# Output:
#   encoder_out  [T, B, num_rvq, V]   (4-D tensor; tbc=True)
#
# Dataset requires:
#   net_input["noisy_logits"]  [B, T_25hz, V]  float  (pre-extracted .npy, codebook-0 logits)
#   labels: ["mimi0", "mimi1", ..., "mimi7"]   (8 separate label files, one per RVQ layer)
#
# Set NOISY_LOGITS_ROOT env var to the directory containing {utt_id}.npy files.

import sys
import logging
import contextlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.tasks import FairseqTask
from omegaconf import OmegaConf, MISSING

DBG = True if len(sys.argv) == 1 else False
if DBG:
    from avhubert_causal import CausalAVHubertModel, AVHubertConfig as CausalAVHubertConfig
else:
    from ..avhubert_causal import CausalAVHubertModel, AVHubertConfig as CausalAVHubertConfig

from .avhubert_mimi import MimiHead
from .avhubert_crossattn_soft import (
    PretrainedSpeakerEncoder,
    SoftCrossAttentionFuser,
)

logger = logging.getLogger(__name__)

try:
    from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
except ImportError:
    MASKING_DISTRIBUTION_CHOICES = Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertCrossAttnSoftRVQConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING,
        metadata={"help": "path to pretrained AV-HuBERT checkpoint"},
    )
    no_pretrained_weights: bool = field(default=False)

    dropout_input: float = field(default=0.0)
    final_dropout: float = field(default=0.0)
    dropout: float = field(default=0.0)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)

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
        default=0,
        metadata={"help": "future frames the backbone can attend to. 0=fully causal."},
    )

    mimi_vocab_size: int = field(default=2048)
    num_rvq: int = field(
        default=8,
        metadata={"help": "number of Mimi RVQ codebooks to predict"},
    )

    head_hidden_dim: int = field(
        default=0,
        metadata={"help": "0 = linear head per codebook, >0 = MLP hidden dim"},
    )

    token_embed_dim: int = field(
        default=256,
        metadata={"help": "embedding dim for noisy Mimi codebook lookup table"},
    )
    cross_attn_heads: int = field(default=4)
    cross_attn_dropout: float = field(default=0.1)
    cross_attn_lookahead: int = field(default=0)

    logit_temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for softmax over noisy logits"},
    )

    speaker_cond: str = field(
        default="none",
        metadata={"help": "Speaker conditioning mode: none | pretrained_spk"},
    )
    speaker_embed_dim: int = field(default=256)
    pretrained_spk_dim: int = field(default=512)

    prefix_length: int = field(default=0)

    normalize: bool = field(default=False)


# ---------------------------------------------------------------------------
# Multi-RVQ head
# ---------------------------------------------------------------------------

class MultiRVQHead(nn.Module):
    """
    num_rvq independent MimiHead modules.

    Input  : x  [B, T, D]
    Output : logits  [B, T, num_rvq, V]
    """

    def __init__(self, in_dim: int, vocab_size: int, num_rvq: int = 8,
                 hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([
            MimiHead(in_dim, vocab_size, hidden_dim, dropout)
            for _ in range(num_rvq)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stack along new dim-2: [B, T, num_rvq, V]
        return torch.stack([h(x) for h in self.heads], dim=2)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CrossAttnSoftRVQEncoder(FairseqEncoder):
    """
    Causal AV-HuBERT backbone + soft cross-attention fuser + multi-RVQ head.
    """

    def __init__(self, cfg: AVHubertCrossAttnSoftRVQConfig):
        self.apply_mask = cfg.apply_mask

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
            for key in ["mask_emb", "label_embs_concat",
                        "final_proj.weight", "final_proj.bias"]:
                sd.pop(key, None)
            remapped = {}
            for k, v in sd.items():
                new_k = k.replace(
                    "feature_extractor_video.resnet.frontend3D.0.weight",
                    "feature_extractor_video.resnet.frontend3D.0.conv.weight",
                )
                remapped[new_k] = v
            missing, unexpected = causal_model.load_state_dict(remapped, strict=False)
            logger.info(f"[CrossAttnSoftRVQEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")

        causal_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model             = causal_model
        self.final_dropout         = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates           = 0

        self.temporal_downsample = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=2, stride=2, padding=0,
        )

        self.speaker_cond = cfg.speaker_cond
        spk_dim = cfg.speaker_embed_dim if cfg.speaker_cond != "none" else 0
        if cfg.speaker_cond == "pretrained_spk":
            self.speaker_encoder = PretrainedSpeakerEncoder(
                pretrained_spk_dim=cfg.pretrained_spk_dim,
                speaker_embed_dim=cfg.speaker_embed_dim)
        else:
            self.speaker_encoder = None

        self.cross_attn_fuser = SoftCrossAttentionFuser(
            d_model=d,
            mimi_vocab=cfg.mimi_vocab_size,
            token_embed_dim=cfg.token_embed_dim,
            n_heads=cfg.cross_attn_heads,
            dropout=cfg.cross_attn_dropout,
            lookahead=cfg.cross_attn_lookahead,
            temperature=cfg.logit_temperature,
            speaker_embed_dim=spk_dim,
            prefix_length=cfg.prefix_length,
        )

        # Multi-RVQ prediction head
        self.rvq_head = MultiRVQHead(
            in_dim=d,
            vocab_size=cfg.mimi_vocab_size,
            num_rvq=cfg.num_rvq,
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
        pm = torch.cat([left, padding_mask], dim=1)
        return pm[:, 1::2][:, :T_out]

    def _upsample_by_repeat(self, x, target_len):
        """x: [B, T, ...] → [B, 2T, ...] by repeating each frame."""
        x = x.repeat_interleave(2, dim=1)
        cur = x.size(1)
        if cur > target_len:
            x = x[:, :target_len]
        elif cur < target_len:
            last = x[:, -1:].expand(x.size(0), target_len - cur, *x.shape[2:])
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
        noisy_logits = kwargs.get("noisy_logits", None)

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(
                source=source,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training,
            )

        orig_len = x.size(1)

        spk_emb = None
        if self.speaker_cond == "pretrained_spk":
            speaker_embed = kwargs.get("speaker_embed", None)
            if speaker_embed is not None:
                spk_emb = self.speaker_encoder(speaker_embed.to(x.dtype))

        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]
            if noisy_logits is not None:
                noisy_logits = noisy_logits[:, :-1, :]

        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)          # [B, T/2, D]

        T_down = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]
            x = self.cross_attn_fuser(x, noisy_logits_ds, spk_emb=spk_emb)

        x = self.final_dropout(x)
        logits_12p5 = self.rvq_head(x)   # [B, T/2, num_rvq, V]

        # upsample back to 25 Hz: [B, T, num_rvq, V]
        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)   # [T, B, num_rvq, V]

        return {
            "encoder_out": logits,
            "encoder_padding_mask": out_pm,
            "padding_mask": out_pm,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_logits = encoder_out["encoder_out"].index_select(1, new_order)
        new_pm = None
        if encoder_out["encoder_padding_mask"] is not None:
            new_pm = encoder_out["encoder_padding_mask"].index_select(0, new_order)
        return {
            "encoder_out": new_logits,
            "encoder_padding_mask": new_pm,
            "padding_mask": new_pm,
        }


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@register_model("av_hubert_crossattn_soft_rvq", dataclass=AVHubertCrossAttnSoftRVQConfig)
class AVHubertCrossAttnSoftRVQModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertCrossAttnSoftRVQConfig, task: FairseqTask):
        encoder = CrossAttnSoftRVQEncoder(cfg)
        return cls(encoder)

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        return self.encoder(source, padding_mask, tbc=tbc, **kwargs)

    def get_logits(self, net_output):
        # returns [T, B, num_rvq, V] or [B, T, num_rvq, V] — criterion handles both
        return net_output["encoder_out"].float()

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = self.get_logits(net_output)
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        return F.softmax(logits, dim=-1)

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
