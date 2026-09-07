# Cross-attention AV-HuBERT: AV features attend to noisy Mimi token embeddings.
#
# Architecture:
#   noisy audio + clean video
#     → CausalAVHubertModel (causal backbone, lookahead_frames configurable)
#     → causal temporal downsample Conv1d (25 Hz → 12.5 Hz)
#     → CrossAttentionFuser:
#         query  = AV features  [B, T, d_model]
#         key/value = noisy_proj(noisy_embed(noisy_tokens))  [B, T, d_model]
#         output = norm(AV + cross_attn_out)  [B, T, d_model]
#     → Mimi head (linear/MLP) → clean Mimi token logits at 12.5 Hz
#     → upsample to 25 Hz for label alignment
#
# The cross-attention learns which noisy token frames to trust (consistent with
# visual evidence) vs. ignore (corrupted by noise).

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
class AVHubertCrossAttnNoisyConfig(FairseqDataclass):
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

    # masking (API compat, off at fine-tune time)
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
        metadata={
            "help": "future frames the backbone transformer can attend to. "
                    "0=fully causal. Each +1 adds 40ms latency. "
                    "Recommended: 0 (streaming), 4 (240ms, best quality)."
        },
    )

    # Mimi vocabulary
    mimi_vocab_size: int = field(default=2048)

    # Mimi head
    head_hidden_dim: int = field(
        default=0,
        metadata={"help": "0 = linear head, >0 = MLP hidden dim"},
    )

    # Cross-attention fuser
    token_embed_dim: int = field(
        default=256,
        metadata={"help": "embedding dim for noisy Mimi token lookup table"},
    )
    cross_attn_heads: int = field(
        default=4,
        metadata={"help": "number of heads in cross-attention fuser"},
    )
    cross_attn_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout inside cross-attention fuser"},
    )

    # Causal cross-attention: optionally allow k future frames in the
    # noisy token sequence (same semantics as lookahead_frames in backbone).
    # 0 = purely causal (key/value can only come from t' <= t).
    # Setting this equal to lookahead_frames is the natural choice.
    cross_attn_lookahead: int = field(
        default=0,
        metadata={
            "help": "lookahead for cross-attention key/value masking. "
                    "0 = causal (only past noisy tokens). "
                    "Set to lookahead_frames to match backbone."
        },
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    label_dir: str = II("task.label_dir")
    w2v_args: Any = None


# ---------------------------------------------------------------------------
# Cross-attention fuser
# ---------------------------------------------------------------------------

class CausalCrossAttentionFuser(nn.Module):
    """
    AV-HuBERT features (query) attend to noisy Mimi token embeddings (key/value).

    Supports causal and lookahead-k masking so that at inference time frame t
    can only see noisy token frames in [t - inf, t + cross_attn_lookahead].

    av_features  : [B, T, d_model]
    noisy_tokens : [B, T] LongTensor  (noisy Mimi argmax from MiMi(noisy_audio))
    returns      : [B, T, d_model]
    """

    def __init__(self, d_model: int, mimi_vocab: int,
                 token_embed_dim: int, n_heads: int,
                 dropout: float = 0.1, lookahead: int = 0):
        super().__init__()
        self.lookahead = lookahead
        self.noisy_embed = nn.Embedding(mimi_vocab + 1, token_embed_dim,
                                        padding_idx=mimi_vocab)
        self.noisy_proj  = nn.Linear(token_embed_dim, d_model)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.noisy_proj.weight)
        nn.init.zeros_(self.noisy_proj.bias)

    def _causal_mask(self, T_q: int, T_kv: int, device: torch.device) -> torch.Tensor:
        """
        Returns additive attention mask [T_q, T_kv] where future positions beyond
        `lookahead` are -inf.  mask[i, j] = 0 if j <= i + lookahead else -inf.
        """
        q_idx  = torch.arange(T_q,  device=device)
        kv_idx = torch.arange(T_kv, device=device)
        # allowed: kv_j <= q_i + lookahead
        allowed = (kv_idx.unsqueeze(0) - q_idx.unsqueeze(1)) <= self.lookahead  # [T_q, T_kv]
        mask = torch.zeros(T_q, T_kv, device=device)
        mask[~allowed] = float("-inf")
        return mask

    def forward(self, av_features: torch.Tensor,
                noisy_tokens: torch.Tensor) -> torch.Tensor:
        B, T_q, _ = av_features.shape

        # clamp OOV tokens (e.g. padding -100) to pad idx
        noisy_tokens = noisy_tokens.clamp(min=0, max=self.noisy_embed.num_embeddings - 1)

        # project noisy tokens to d_model key/value space
        noisy_ctx = self.noisy_proj(self.noisy_embed(noisy_tokens))  # [B, T_kv, d_model]
        T_kv = noisy_ctx.size(1)

        # causal (+ optional lookahead) mask [T_q, T_kv], cast to match fp16
        attn_mask = self._causal_mask(T_q, T_kv, av_features.device).to(dtype=av_features.dtype)

        attended, _ = self.cross_attn(
            query=av_features,
            key=noisy_ctx,
            value=noisy_ctx,
            attn_mask=attn_mask,
        )
        return self.norm(av_features + attended)  # [B, T, d_model]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CrossAttnNoisyEncoder(FairseqEncoder):
    """
    Wraps CausalAVHubertModel; fuses AV features with noisy Mimi tokens via
    cross-attention before the Mimi prediction head.
    """

    def __init__(self, cfg: AVHubertCrossAttnNoisyConfig):
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
            logger.info(f"[CrossAttnNoisyEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")
            if missing:
                logger.info(f"  missing keys (first 10): {missing[:10]}")

        causal_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model             = causal_model
        self.final_dropout         = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates           = 0

        # Causal temporal downsample 25 Hz → 12.5 Hz
        self.temporal_downsample = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=2, stride=2, padding=0,
        )

        # Cross-attention fuser: AV queries noisy token context
        self.cross_attn_fuser = CausalCrossAttentionFuser(
            d_model=d,
            mimi_vocab=cfg.mimi_vocab_size,
            token_embed_dim=cfg.token_embed_dim,
            n_heads=cfg.cross_attn_heads,
            dropout=cfg.cross_attn_dropout,
            lookahead=cfg.cross_attn_lookahead,
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
        pm = torch.cat([left, padding_mask], dim=1)
        return pm[:, 1::2][:, :T_out]

    def _upsample_by_repeat(self, x, target_len):
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
        source: dict
            'audio'       : noisy filterbank features [B, T*stack, audio_feat_dim]
            'video'       : clean lip frames [B, 1, T, H, W]
            'noisy_tokens': noisy Mimi argmax [B, T] LongTensor  ← NEW
        padding_mask: [B, T]
        """
        noisy_tokens = source.get("noisy_tokens", None)

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
            if noisy_tokens is not None:
                noisy_tokens = noisy_tokens[:, :-1]

        # causal temporal downsample → 12.5 Hz
        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)                   # [B, T/2, D]

        T_down = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        # Downsample noisy tokens to match 12.5 Hz by taking every 2nd token
        if noisy_tokens is not None:
            noisy_tokens_ds = noisy_tokens[:, 1::2][:, :T_down]  # [B, T/2]
            # cross-attend: AV queries noisy token context
            x = self.cross_attn_fuser(x, noisy_tokens_ds)        # [B, T/2, D]

        x = self.final_dropout(x)
        logits_12p5 = self.mimi_head(x)          # [B, T/2, V]

        # upsample back to 25 Hz for label alignment
        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)      # [T, B, V]

        return {
            "encoder_out": logits,
            "encoder_padding_mask": out_pm,
            "padding_mask": out_pm,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_logits = encoder_out["encoder_out"].index_select(1, new_order)
        new_pm     = None
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

@register_model("av_hubert_crossattn_noisy", dataclass=AVHubertCrossAttnNoisyConfig)
class AVHubertCrossAttnNoisyModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertCrossAttnNoisyConfig, task: FairseqTask):
        encoder = CrossAttnNoisyEncoder(cfg)
        return cls(encoder)

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        return self.encoder(source, padding_mask, tbc=tbc, **kwargs)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        if logits.dim() == 3:         # [T, B, V] or [B, T, V]
            logits = logits.float()
        return logits

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = self.get_logits(net_output)
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        return F.softmax(logits, dim=-1)

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
