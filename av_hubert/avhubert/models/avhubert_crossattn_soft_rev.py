# Cross-attention AV-HuBERT — REVERSED direction: Q=noisy logits, K/V=AV features.
#
# Contrast with avhubert_crossattn_soft.py (Q=AV, K/V=noisy logits):
#
#   forward  (crossattn_soft):      AV features query acoustic context
#   reversed (crossattn_soft_rev):  noisy logits query visual context
#
# Denoising framing:
#   The corrupted acoustic estimate drives the query; the cleaner AV backbone
#   features (which include lip-reading) are the key/value "oracle". The model
#   learns "given this ambiguous token, what does the visual stream say?"
#
# Attention-entropy behaviour:
#   clean audio  → low-entropy noisy_logits → sharp soft-embedding query
#                → focused attention over a few AV feature positions
#   noisy audio  → high-entropy noisy_logits → diffuse query
#                → broad attention integrating visual context
#
# Output space: cross-attn output is in AV feature space [B, T, D] (because
# V=AV features determines output dim), same as the forward direction.
# Residual is added to AV features → same MimiHead as forward model.
#
# Dataset requires:
#   net_input["noisy_logits"]  [B, T_25hz, V]  float  (pre-extracted .npy)
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
from .avhubert_mamba_soft import PredictiveCodingHead

logger = logging.getLogger(__name__)

try:
    from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
except ImportError:
    MASKING_DISTRIBUTION_CHOICES = Any


# ---------------------------------------------------------------------------
# Config  (identical to AVHubertCrossAttnSoftConfig)
# ---------------------------------------------------------------------------

@dataclass
class AVHubertCrossAttnSoftRevConfig(FairseqDataclass):
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

    freeze_finetune_updates: int = field(default=0)
    feature_grad_mult: float = field(default=1.0)
    layerdrop: float = field(default=0.0)
    lookahead_frames: int = field(
        default=0,
        metadata={"help": "future frames the backbone can attend to. 0=fully causal."},
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
        metadata={"help": "embedding dim for noisy Mimi codebook lookup table"},
    )
    cross_attn_heads: int = field(default=4)
    cross_attn_dropout: float = field(default=0.1)
    cross_attn_lookahead: int = field(
        default=0,
        metadata={"help": "lookahead frames for cross-attention mask"},
    )

    # Soft logit temperature
    logit_temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "Temperature for softmax over noisy logits. "
                "<1.0 sharpens (more like hard), >1.0 flattens (more uncertainty). "
                "Default 1.0 = standard softmax."
            )
        },
    )

    # Speaker conditioning
    speaker_cond: str = field(
        default="none",
        metadata={"help": "Speaker conditioning mode: none | pretrained_spk"},
    )
    speaker_embed_dim: int = field(
        default=256,
        metadata={"help": "Internal speaker embedding dimension after projection."},
    )
    pretrained_spk_dim: int = field(
        default=512,
        metadata={"help": "Dimensionality of pre-extracted speaker embeddings (ArcFace=512)."},
    )

    # Prefix tuning (prepended to K/V = AV features in this reversed model)
    prefix_length: int = field(
        default=0,
        metadata={
            "help": (
                "Number of prefix vectors prepended to the AV feature key/value sequence. "
                "0 = disabled. If speaker_cond != none, prefix is speaker-conditioned."
            )
        },
    )

    normalize: bool = field(default=False)

    # Predictive coding auxiliary loss
    pred_coding_k: int = field(
        default=0,
        metadata={"help": "predict next k AV-HuBERT frames. 0 = disabled."},
    )


# ---------------------------------------------------------------------------
# Speaker encoder  (unchanged from forward model)
# ---------------------------------------------------------------------------

class PretrainedSpeakerEncoder(nn.Module):
    """
    Input  : [B, pretrained_spk_dim]  (ArcFace = 512)
    Returns: [B, speaker_embed_dim]   (= 256)
    """

    def __init__(self, pretrained_spk_dim: int, speaker_embed_dim: int):
        super().__init__()
        self.h, self.w = 16, 32
        assert pretrained_spk_dim == self.h * self.w
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, speaker_embed_dim),
            nn.ReLU(),
            nn.Linear(speaker_embed_dim, speaker_embed_dim),
        )

    def forward(self, speaker_embed: torch.Tensor) -> torch.Tensor:
        B = speaker_embed.size(0)
        x = speaker_embed.view(B, 1, self.h, self.w)
        x = self.cnn(x)
        x = x.mean(dim=(-2, -1))
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Reversed soft cross-attention fuser
# ---------------------------------------------------------------------------

class ReversedSoftCrossAttentionFuser(nn.Module):
    """
    Cross-attention with REVERSED query/key-value roles vs. SoftCrossAttentionFuser.

    Forward model:   Q = AV features,        K/V = soft(noisy_logits) @ embed
    This model:      Q = soft(noisy_logits) @ embed,   K/V = AV features

    Intuition: the noisy acoustic estimate asks "what does the visual stream say
    about this ambiguous token?" rather than AV features asking "what does the
    noisy decoder say?".

    noisy_logits : [B, T, V]   float  (pre-extracted Mimi cosine-sim logits)
    av_features  : [B, T, D]
    returns      : [B, T, D]   (AV features + acoustically-selected visual context)
    """

    def __init__(self, d_model: int, mimi_vocab: int,
                 token_embed_dim: int, n_heads: int,
                 dropout: float = 0.1, lookahead: int = 0,
                 temperature: float = 1.0, speaker_embed_dim: int = 0,
                 prefix_length: int = 0):
        super().__init__()
        self.lookahead     = lookahead
        self.temperature   = temperature
        self.mimi_vocab    = mimi_vocab
        self.prefix_length = prefix_length

        # Codebook embedding table + projection: noisy_logits → Q space
        self.noisy_embed = nn.Embedding(mimi_vocab, token_embed_dim)
        self.noisy_proj  = nn.Linear(token_embed_dim, d_model)

        # cross_attn embed_dim = d_model (AV backbone dim)
        # Q [B, T, d_model] = noisy soft embedding projected to d_model
        # K/V [B, T, d_model] = AV features (already d_model)
        # output [B, T, d_model]  ← same as forward model
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.noisy_proj.weight)
        nn.init.zeros_(self.noisy_proj.bias)

        # Speaker-conditioned bias on noisy logits before softmax
        # (shapes the query distribution)
        if speaker_embed_dim > 0:
            self.speaker_logit_bias = nn.Linear(speaker_embed_dim, mimi_vocab, bias=False)
        else:
            self.speaker_logit_bias = None

        # Prefix tuning: prepended to K/V (AV features) in this reversed model
        if prefix_length > 0:
            if speaker_embed_dim > 0:
                self.prefix_k_proj = nn.Linear(speaker_embed_dim, prefix_length * d_model)
                self.prefix_v_proj = nn.Linear(speaker_embed_dim, prefix_length * d_model)
                nn.init.normal_(self.prefix_k_proj.weight, std=0.02)
                nn.init.zeros_(self.prefix_k_proj.bias)
                nn.init.normal_(self.prefix_v_proj.weight, std=0.02)
                nn.init.zeros_(self.prefix_v_proj.bias)
                self.prefix_k_shared = None
                self.prefix_v_shared = None
            else:
                self.prefix_k_shared = nn.Parameter(torch.randn(1, prefix_length, d_model) * 0.02)
                self.prefix_v_shared = nn.Parameter(torch.randn(1, prefix_length, d_model) * 0.02)
                self.prefix_k_proj = None
                self.prefix_v_proj = None
        else:
            self.prefix_k_proj   = None
            self.prefix_v_proj   = None
            self.prefix_k_shared = None
            self.prefix_v_shared = None

    def _causal_mask(self, T_q: int, T_kv: int, K_prefix: int,
                     device, dtype) -> torch.Tensor:
        """[T_q, K_prefix + T_kv]: prefix always attended, KV positions causal+lookahead."""
        T_total = K_prefix + T_kv
        mask = torch.zeros(T_q, T_total, device=device, dtype=dtype)
        if T_kv > 0:
            q_idx  = torch.arange(T_q,  device=device)
            kv_idx = torch.arange(T_kv, device=device)
            allowed = (kv_idx.unsqueeze(0) - q_idx.unsqueeze(1)) <= self.lookahead
            mask[:, K_prefix:][~allowed] = float("-inf")
        return mask

    def forward(self, av_features: torch.Tensor,
                noisy_logits: torch.Tensor,
                spk_emb: torch.Tensor = None) -> torch.Tensor:
        """
        av_features  : [B, T, D]   — AV backbone output (key/value)
        noisy_logits : [B, T, V]   — pre-extracted Mimi logits (drives query)
        spk_emb      : [B, speaker_embed_dim]  (optional)
        """
        B = av_features.size(0)
        T = av_features.size(1)
        K = self.prefix_length

        # Optional speaker bias on the query logit distribution
        if self.speaker_logit_bias is not None and spk_emb is not None:
            bias = self.speaker_logit_bias(spk_emb.to(av_features.dtype))  # [B, V]
            noisy_logits = noisy_logits + bias.unsqueeze(1)

        # Build query: soft codebook lookup → project to d_model
        weights   = F.softmax(noisy_logits.float() / self.temperature, dim=-1).to(av_features.dtype)
        soft_emb  = weights @ self.noisy_embed.weight   # [B, T, token_embed_dim]
        query     = self.noisy_proj(soft_emb)            # [B, T, d_model]  ← Q

        # Build key/value from AV features (+ optional prefix)
        if K > 0:
            if self.prefix_k_proj is not None and spk_emb is not None:
                s = spk_emb.to(av_features.dtype)
                prefix_k = self.prefix_k_proj(s).view(B, K, -1)
                prefix_v = self.prefix_v_proj(s).view(B, K, -1)
            else:
                prefix_k = self.prefix_k_shared.expand(B, -1, -1)
                prefix_v = self.prefix_v_shared.expand(B, -1, -1)
            key   = torch.cat([prefix_k, av_features], dim=1)   # [B, K+T, D]
            value = torch.cat([prefix_v, av_features], dim=1)
        else:
            key   = av_features   # [B, T, D]
            value = av_features

        # Causal mask [T_q, K + T_kv]
        attn_mask = self._causal_mask(T, T, K, av_features.device, av_features.dtype)

        # Q = noisy soft embedding, K/V = AV features
        attended, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )

        # Residual onto AV features (output is in AV feature space via V)
        return self.norm(av_features + attended)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CrossAttnSoftRevEncoder(FairseqEncoder):
    """
    Causal AV-HuBERT backbone + reversed soft cross-attention fuser.
    Noisy Mimi logits query the AV feature space; output is AV-based.
    """

    def __init__(self, cfg: AVHubertCrossAttnSoftRevConfig):
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
            logger.info(f"[CrossAttnSoftRevEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")

        causal_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model              = causal_model
        self.final_dropout          = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates            = 0

        # Causal temporal downsample 25 Hz → 12.5 Hz
        self.temporal_downsample = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=2, stride=2, padding=0,
        )

        # Speaker encoder
        self.speaker_cond = cfg.speaker_cond
        spk_dim = cfg.speaker_embed_dim if cfg.speaker_cond != "none" else 0
        if cfg.speaker_cond == "pretrained_spk":
            self.speaker_encoder = PretrainedSpeakerEncoder(
                pretrained_spk_dim=cfg.pretrained_spk_dim,
                speaker_embed_dim=cfg.speaker_embed_dim)
        else:
            self.speaker_encoder = None

        # Reversed fuser: Q=noisy logits, K/V=AV features
        self.cross_attn_fuser = ReversedSoftCrossAttentionFuser(
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

        self.mimi_head = MimiHead(
            in_dim=d,
            out_dim=cfg.mimi_vocab_size,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.final_dropout,
        )

        self.pred_coding_k = cfg.pred_coding_k
        if cfg.pred_coding_k > 0:
            self.pred_coding_head = PredictiveCodingHead(d, k=cfg.pred_coding_k)
        else:
            self.pred_coding_head = None

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
        noisy_logits = kwargs.get("noisy_logits", None)

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(
                source=source,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training,
            )

        # x: [B, T, D] at 25 Hz
        orig_len = x.size(1)

        spk_emb = None
        if self.speaker_cond == "pretrained_spk":
            speaker_embed = kwargs.get("speaker_embed", None)
            if speaker_embed is not None:
                spk_emb = self.speaker_encoder(speaker_embed.to(x.dtype))

        # ensure even length for stride-2 conv
        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]
            if noisy_logits is not None:
                noisy_logits = noisy_logits[:, :-1, :]

        # causal temporal downsample → 12.5 Hz
        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)          # [B, T/2, D]

        T_down = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        av_features_ds = x.detach() if self.pred_coding_head is not None else None

        # Downsample noisy logits 25 Hz → 12.5 Hz (take every 2nd frame)
        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]  # [B, T/2, V]
            # Reversed cross-attention: Q=noisy logits, K/V=AV features
            x = self.cross_attn_fuser(
                x, noisy_logits_ds, spk_emb=spk_emb,
            )  # [B, T/2, D]

        pred_coding_loss = None
        if self.training and self.pred_coding_head is not None and av_features_ds is not None:
            pred_coding_loss = self.pred_coding_head(x, av_features_ds)

        x = self.final_dropout(x)
        logits_12p5 = self.mimi_head(x)   # [B, T/2, V]

        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)   # [T, B, V]

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             x,
            "pred_coding_loss":     pred_coding_loss,
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

@register_model("av_hubert_crossattn_soft_rev", dataclass=AVHubertCrossAttnSoftRevConfig)
class AVHubertCrossAttnSoftRevModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertCrossAttnSoftRevConfig, task: FairseqTask):
        encoder = CrossAttnSoftRevEncoder(cfg)
        return cls(encoder)

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        return self.encoder(source, padding_mask, tbc=tbc, **kwargs)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        if logits.dim() == 3:
            logits = logits.float()
        return logits

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = self.get_logits(net_output)
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        return F.softmax(logits, dim=-1)

    def upgrade_state_dict_named(self, state_dict, name):
        own_keys = set(self.state_dict().keys())
        for key in list(state_dict.keys()):
            if key not in own_keys:
                logger.info(f"[upgrade_state_dict] dropping unknown key: {key}")
                state_dict.pop(key)
        return state_dict
