# Cross-attention AV-HuBERT with SOFT noisy Mimi logits.
#
# Same architecture as avhubert_crossattn_noisy.py, but the cross-attention
# fuser uses a soft weighted-sum over the Mimi codebook embeddings instead of
# a hard argmax token lookup:
#
#   hard (crossattn_noisy):  key/value = embed(argmax(noisy_logits))
#   soft (this file):        key/value = softmax(noisy_logits / T) @ embed.weight
#
# This propagates acoustic uncertainty into the visual fusion:
# - When the noisy audio is clear, softmax is peaked → behaves like hard lookup
# - When the noisy audio is ambiguous, softmax is flat → AV features dominate
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
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertCrossAttnSoftConfig(FairseqDataclass):
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
    # Options:
    #   "none"           - no speaker conditioning (baseline)
    #   "pretrained_spk" - logit bias + prefix tuning (K/V side of cross-attn)
    #   "film"           - FiLM on AV features (query side of cross-attn)
    #   "film_prefix"    - FiLM on AV features + prefix tuning (both sides, strongest)
    #   "adaln"          - Adaptive LayerNorm on AV features (normalize-then-modulate)
    #                      Peebles & Xie, "DiT", ICCV 2023.
    #   "concat"         - Concatenate speaker embed to every AV frame, project back to d_model
    #
    # All modes load embeddings from ARCFACE_EMBED_ROOT (set env var).
    # Use ArcFace (512-dim) for face identity or ECAPA-TDNN (192-dim) for voice identity.
    # WARNING: if ARCFACE_EMBED_ROOT is unset or paths don't match, conditioning silently
    # disables itself — watch for the "[speaker_cond=…] speaker_embed is None" warning.
    speaker_cond: str = field(
        default="none",
        metadata={"help": "Speaker conditioning mode: none | pretrained_spk | film | film_prefix | adaln | concat"},
    )
    speaker_embed_dim: int = field(
        default=256,
        metadata={"help": "Internal speaker embedding dimension after projection."},
    )
    # Input embedding dimension: ArcFace=512, ECAPA-TDNN=192, WavLM-TDNN=256
    pretrained_spk_dim: int = field(
        default=512,
        metadata={"help": "Dimension of pre-extracted speaker embeddings (ArcFace=512, ECAPA=192)."},
    )

    # Prefix tuning
    prefix_length: int = field(
        default=0,
        metadata={
            "help": (
                "Number of speaker-conditioned prefix vectors prepended to the "
                "cross-attention key/value sequence. 0 = disabled. "
                "All query positions can attend to prefix (always causal-safe). "
                "If speaker_cond != none, prefix is generated from spk_emb. "
                "Otherwise a shared learned prefix is used."
            )
        },
    )

    normalize: bool = field(default=False)

    # Predictive coding auxiliary loss (training only, zero inference cost)
    pred_coding_k: int = field(
        default=0,
        metadata={"help": "predict next k AV-HuBERT frames. 0 = disabled."},
    )


# ---------------------------------------------------------------------------
# Speaker encoder
# ---------------------------------------------------------------------------

class PretrainedSpeakerEncoder(nn.Module):
    """
    MLP encoder for pre-extracted speaker embeddings (ECAPA-TDNN, ArcFace, x-vector, etc.).
    Works with any input dimension — no spatial structure assumed.

    Input  : [B, pretrained_spk_dim]
    Returns: [B, speaker_embed_dim]
    """

    def __init__(self, pretrained_spk_dim: int, speaker_embed_dim: int):
        super().__init__()
        hidden = max(speaker_embed_dim, pretrained_spk_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(pretrained_spk_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, speaker_embed_dim),
        )

    def forward(self, speaker_embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(speaker_embed)  # [B, speaker_embed_dim]


# ---------------------------------------------------------------------------
# Soft cross-attention fuser
# ---------------------------------------------------------------------------

class SoftCrossAttentionFuser(nn.Module):
    """
    Cross-attention where key/value = softmax(noisy_logits / T) @ embed.weight

    noisy_logits : [B, T_kv, V]  float  (pre-extracted Mimi cosine-sim logits)
    av_features  : [B, T_q, D]
    returns      : [B, T_q, D]
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

        # shared codebook embedding table [V, token_embed_dim]
        self.noisy_embed = nn.Embedding(mimi_vocab, token_embed_dim)
        self.noisy_proj  = nn.Linear(token_embed_dim, d_model)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.noisy_proj.weight)
        nn.init.zeros_(self.noisy_proj.bias)

        # speaker-conditioned bias added to noisy logits before softmax
        if speaker_embed_dim > 0:
            self.speaker_logit_bias = nn.Linear(speaker_embed_dim, mimi_vocab, bias=False)
        else:
            self.speaker_logit_bias = None

        # Prefix tuning: speaker-conditioned key/value prefix vectors
        # prefix_k/v are generated per-utterance from spk_emb via a linear layer
        # If no speaker info: fallback to a shared learned prefix (speaker-agnostic)
        if prefix_length > 0:
            if speaker_embed_dim > 0:
                # speaker-conditioned: Linear(spk_emb) → [B, K*D] → reshape [B, K, D]
                self.prefix_k_proj = nn.Linear(speaker_embed_dim, prefix_length * d_model)
                self.prefix_v_proj = nn.Linear(speaker_embed_dim, prefix_length * d_model)
                nn.init.normal_(self.prefix_k_proj.weight, std=0.02)
                nn.init.zeros_(self.prefix_k_proj.bias)
                nn.init.normal_(self.prefix_v_proj.weight, std=0.02)
                nn.init.zeros_(self.prefix_v_proj.bias)
                self.prefix_k_shared = None
                self.prefix_v_shared = None
            else:
                # speaker-agnostic: shared learned prefix [1, K, D]
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
        """
        [T_q, K_prefix + T_kv] mask.
        All query positions can attend to all prefix positions (always allowed).
        For noisy token positions, apply causal + lookahead constraint.
        """
        T_total = K_prefix + T_kv
        mask = torch.zeros(T_q, T_total, device=device, dtype=dtype)

        if T_kv > 0:
            q_idx  = torch.arange(T_q,  device=device)
            kv_idx = torch.arange(T_kv, device=device)
            allowed = (kv_idx.unsqueeze(0) - q_idx.unsqueeze(1)) <= self.lookahead
            # prefix columns (0..K_prefix-1) are always attended; set noisy cols
            mask[:, K_prefix:][~allowed] = float("-inf")

        return mask

    def forward(self, av_features: torch.Tensor,
                noisy_logits: torch.Tensor,
                spk_emb: torch.Tensor = None) -> torch.Tensor:
        """
        av_features  : [B, T_q, D]
        noisy_logits : [B, T_kv, V]
        spk_emb      : [B, speaker_embed_dim]  (optional)
        """
        B    = av_features.size(0)
        T_q  = av_features.size(1)
        T_kv = noisy_logits.size(1)
        K    = self.prefix_length

        # Option A: speaker-specific bias on noisy logits before softmax
        if self.speaker_logit_bias is not None and spk_emb is not None:
            bias = self.speaker_logit_bias(spk_emb.to(av_features.dtype))  # [B, V]
            noisy_logits = noisy_logits + bias.unsqueeze(1)

        # soft weights over codebook: [B, T_kv, V]
        weights = F.softmax(noisy_logits.float() / self.temperature, dim=-1).to(av_features.dtype)

        # weighted codebook lookup → project to d_model
        soft_embed = weights @ self.noisy_embed.weight  # [B, T_kv, E]
        noisy_ctx  = self.noisy_proj(soft_embed)        # [B, T_kv, D]

        # Prefix tuning: prepend K speaker-conditioned vectors to key/value
        if K > 0:
            if self.prefix_k_proj is not None and spk_emb is not None:
                # speaker-conditioned prefix
                s = spk_emb.to(av_features.dtype)
                prefix_k = self.prefix_k_proj(s).view(B, K, -1)  # [B, K, D]
                prefix_v = self.prefix_v_proj(s).view(B, K, -1)
            elif self.prefix_k_shared is not None:
                # shared learned prefix, broadcast to batch
                prefix_k = self.prefix_k_shared.expand(B, -1, -1)
                prefix_v = self.prefix_v_shared.expand(B, -1, -1)
            else:
                # speaker-conditioned prefix configured but no embedding provided → zero prefix
                D = av_features.size(-1)
                prefix_k = torch.zeros(B, K, D, device=av_features.device, dtype=av_features.dtype)
                prefix_v = torch.zeros_like(prefix_k)

            key   = torch.cat([prefix_k, noisy_ctx], dim=1)  # [B, K+T_kv, D]
            value = torch.cat([prefix_v, noisy_ctx], dim=1)
        else:
            key   = noisy_ctx
            value = noisy_ctx

        # causal mask [T_q, K + T_kv]
        attn_mask = self._causal_mask(T_q, T_kv, K, av_features.device, av_features.dtype)

        attended, _ = self.cross_attn(
            query=av_features,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )
        return self.norm(av_features + attended)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CrossAttnSoftEncoder(FairseqEncoder):
    """
    Causal AV-HuBERT backbone + soft cross-attention fuser over noisy Mimi logits.
    """

    def __init__(self, cfg: AVHubertCrossAttnSoftConfig):
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
            logger.info(f"[CrossAttnSoftEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")

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

        # Soft cross-attention fuser
        # Optional speaker encoder
        #
        # Normalize speaker_cond: YAML "None" (Python None) and "none" both mean disabled.
        # Without this, `None != "none"` evaluates True and accidentally enables the encoder.
        _raw_cond = cfg.speaker_cond
        if _raw_cond is None or (isinstance(_raw_cond, str) and _raw_cond.lower() == "none"):
            self.speaker_cond = "none"
        else:
            self.speaker_cond = str(_raw_cond)

        _use_spk = self.speaker_cond != "none"
        spk_dim  = cfg.speaker_embed_dim if _use_spk else 0

        # Speaker encoder: MLP that projects raw embedding → speaker_embed_dim
        # Shared across all speaker_cond modes
        if _use_spk:
            self.speaker_encoder = PretrainedSpeakerEncoder(
                pretrained_spk_dim=cfg.pretrained_spk_dim,
                speaker_embed_dim=cfg.speaker_embed_dim)
        else:
            self.speaker_encoder = None

        # FiLM layers: modulate AV features (query side of cross-attention)
        # Applied after temporal downsample, before cross-attention fuser.
        # γ(z) ⊙ x + β(z)  — initialized to identity (γ=1, β=0).
        # Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", NeurIPS 2018.
        if self.speaker_cond in ("film", "film_prefix"):
            self.film_gamma = nn.Linear(cfg.speaker_embed_dim, d)
            self.film_beta  = nn.Linear(cfg.speaker_embed_dim, d)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)   # γ(z) starts at 1 → identity scale
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)   # β(z) starts at 0 → no shift
        else:
            self.film_gamma = None
            self.film_beta  = None

        # AdaLN: normalize AV features first, then modulate with speaker identity.
        # Differs from FiLM: operates on unit-variance features → more stable gradients.
        # γ(z) ⊙ LayerNorm(x) + β(z)  — initialized to identity.
        # Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)", ICCV 2023.
        if self.speaker_cond == "adaln":
            self.adaln_norm  = nn.LayerNorm(d)
            self.adaln_gamma = nn.Linear(cfg.speaker_embed_dim, d)
            self.adaln_beta  = nn.Linear(cfg.speaker_embed_dim, d)
            nn.init.zeros_(self.adaln_gamma.weight)
            nn.init.ones_(self.adaln_gamma.bias)   # γ starts at 1 → identity
            nn.init.zeros_(self.adaln_beta.weight)
            nn.init.zeros_(self.adaln_beta.bias)   # β starts at 0 → no shift
        else:
            self.adaln_norm  = None
            self.adaln_gamma = None
            self.adaln_beta  = None

        # Concat: broadcast speaker embed to every frame, concat, project back to d_model.
        # Simplest possible grounding — every position directly sees the speaker identity.
        if self.speaker_cond == "concat":
            self._concat_spk_dim = cfg.speaker_embed_dim
            self.concat_proj = nn.Linear(d + cfg.speaker_embed_dim, d)
            # Identity init: x-portion passes through unchanged, spk-portion starts at zero.
            # Same no-op principle as FiLM (γ=1,β=0) — preserves pretrained features at step 0.
            # weight shape: [d, d+spk_dim]; first d columns = I, last spk_dim columns = 0.
            nn.init.eye_(self.concat_proj.weight[:, :d])
            nn.init.zeros_(self.concat_proj.weight[:, d:])
            nn.init.zeros_(self.concat_proj.bias)
        else:
            self._concat_spk_dim = 0
            self.concat_proj = None

        # warn once (not every step) when conditioning is requested but embed is missing
        self._warned_missing_spk = False

        # Prefix tuning and logit bias are K/V-side conditioning:
        # active for pretrained_spk and film_prefix, not for film-only.
        _kv_spk_dim = spk_dim if cfg.speaker_cond in ("pretrained_spk", "film_prefix") else 0
        self.cross_attn_fuser = SoftCrossAttentionFuser(
            d_model=d,
            mimi_vocab=cfg.mimi_vocab_size,
            token_embed_dim=cfg.token_embed_dim,
            n_heads=cfg.cross_attn_heads,
            dropout=cfg.cross_attn_dropout,
            lookahead=cfg.cross_attn_lookahead,
            temperature=cfg.logit_temperature,
            speaker_embed_dim=_kv_spk_dim,
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
        """
        source: dict with 'audio', 'video'
        net_input must also contain 'noisy_logits': [B, T_25hz, V]
        """
        # noisy_logits passed alongside source in net_input
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

        # Speaker embedding [B, speaker_embed_dim]
        spk_emb = None
        if self.speaker_cond != "none":
            speaker_embed = kwargs.get("speaker_embed", None)
            if speaker_embed is not None and self.speaker_encoder is not None:
                spk_emb = self.speaker_encoder(speaker_embed.to(x.dtype))
            if spk_emb is None and not self._warned_missing_spk:
                logger.warning(
                    "[speaker_cond=%s] speaker_embed is None — conditioning disabled for "
                    "this run. Verify ARCFACE_EMBED_ROOT is set and fid paths match.",
                    self.speaker_cond,
                )
                self._warned_missing_spk = True

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

        # FiLM: modulate AV features with speaker identity (query-side conditioning).
        # γ(spk) ⊙ x + β(spk) — initialized to identity so pre-trained weights are preserved.
        # Perez et al., NeurIPS 2018.
        if spk_emb is not None and self.film_gamma is not None:
            gamma = self.film_gamma(spk_emb).unsqueeze(1)  # [B, 1, D]
            beta  = self.film_beta(spk_emb).unsqueeze(1)   # [B, 1, D]
            x = gamma * x + beta

        # AdaLN: normalize then modulate — more stable than FiLM because γ/β operate on
        # unit-variance features regardless of the temporal-downsample output scale.
        # When speaker embed is missing, still normalizes (keeps fuser input stable).
        # Peebles & Xie, ICCV 2023.
        if self.adaln_gamma is not None:
            x_n = self.adaln_norm(x)
            if spk_emb is not None:
                gamma = self.adaln_gamma(spk_emb).unsqueeze(1)  # [B, 1, D]
                beta  = self.adaln_beta(spk_emb).unsqueeze(1)   # [B, 1, D]
                x = gamma * x_n + beta
            else:
                x = x_n  # identity (γ=1, β=0) — no speaker info

        # Concat: broadcast speaker embed to every frame, project [AV ‖ spk] → d_model.
        # Falls back to zero speaker vector when embed is unavailable (explicit zero signal).
        if self.concat_proj is not None:
            if spk_emb is not None:
                spk_t = spk_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, spk_dim]
            else:
                spk_t = x.new_zeros(x.size(0), x.size(1), self._concat_spk_dim)
            x = self.concat_proj(torch.cat([x, spk_t], dim=-1))

        # save AV features before fuser as predictive coding target
        av_features_ds = x.detach() if self.pred_coding_head is not None else None

        # Downsample noisy logits 25 Hz → 12.5 Hz (take every 2nd frame)
        # Pass spk_emb to fuser only for modes that use K/V-side conditioning.
        _fuser_spk = spk_emb if self.speaker_cond in ("pretrained_spk", "film_prefix") else None
        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]  # [B, T/2, V]
            x = self.cross_attn_fuser(
                x, noisy_logits_ds, spk_emb=_fuser_spk,
            )  # [B, T/2, D]

        # predictive coding aux loss (training only)
        pred_coding_loss = None
        if self.training and self.pred_coding_head is not None and av_features_ds is not None:
            pred_coding_loss = self.pred_coding_head(x, av_features_ds)

        x = self.final_dropout(x)
        logits_12p5 = self.mimi_head(x)   # [B, T/2, V]

        logits  = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm  = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)   # [T, B, V]

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             x,
            "logits_12p5":          logits_12p5,   # [B, T/2, V] — for sync loss soft embed
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

@register_model("av_hubert_crossattn_soft", dataclass=AVHubertCrossAttnSoftConfig)
class AVHubertCrossAttnSoftModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertCrossAttnSoftConfig, task: FairseqTask):
        encoder = CrossAttnSoftEncoder(cfg)
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
        # Drop any checkpoint keys that don't exist in the current model
        # (e.g. speaker_encoder CNN trained with an older config that predates
        # the speaker_cond field — fairseq rebuilds with the default "none",
        # so the encoder is absent and strict loading would fail).
        own_keys = set(self.state_dict().keys())
        for key in list(state_dict.keys()):
            if key not in own_keys:
                logger.info(f"[upgrade_state_dict] dropping unknown key: {key}")
                state_dict.pop(key)
        return state_dict
