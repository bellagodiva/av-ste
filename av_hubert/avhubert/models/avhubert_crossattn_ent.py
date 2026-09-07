# Cross-attention AV-HuBERT with NORMALIZED-ENTROPY-GATED soft Mimi logits.
#
# Same architecture as avhubert_crossattn_soft.py, but the noisy context fed
# into cross-attention is scaled by a per-frame confidence gate derived from
# the normalized entropy of the softmax distribution over Mimi logits:
#
#   p        = softmax(noisy_logits / T)            [B, T_kv, V]
#   H_norm   = -∑ p·log(p) / log(V)                [B, T_kv]  ∈ [0, 1]
#   confidence = 1 − H_norm                          [B, T_kv]
#   soft_embed = p @ embed.weight                    [B, T_kv, E]
#   noisy_ctx  = noisy_proj(soft_embed) * confidence [B, T_kv, D]
#
# Compared to max-probability gating:
#   max_prob = max(p)                               peaky dist → near 1
#   1−H_norm captures the full shape of the distribution, not just the peak.
#   For a two-way tie, max_prob=0.5 but H_norm≈1/V·log(V) is much lower →
#   more nuanced suppression of ambiguous frames.
#
# Behaviour:
#   - Clear audio   → p is peaked → H_norm ≈ 0 → confidence ≈ 1 → full noisy signal
#   - Noisy audio   → p is flat   → H_norm ≈ 1 → confidence ≈ 0 → AV features dominate
#
# Dataset requires:
#   net_input["noisy_logits"]  [B, T_25hz, V]  float  (pre-extracted .npy)
#
# Set NOISY_LOGITS_ROOT env var to the directory containing {utt_id}.npy files.

import math
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
class AVHubertCrossAttnEntConfig(FairseqDataclass):
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

    # Soft logit temperature (used for the softmax that also computes entropy)
    logit_temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "Temperature for softmax over noisy logits. "
                "<1.0 sharpens (lower entropy → higher gate), "
                ">1.0 flattens (higher entropy → lower gate). "
                "Default 1.0 = standard softmax."
            )
        },
    )

    # Speaker conditioning
    speaker_cond: str = field(
        default="none",
        metadata={"help": "Speaker conditioning mode: none | pretrained_spk | film | film_prefix | adaln | concat"},
    )
    speaker_embed_dim: int = field(
        default=256,
        metadata={"help": "Internal speaker embedding dimension after projection."},
    )
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
                "cross-attention key/value sequence. 0 = disabled."
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
# Speaker encoder
# ---------------------------------------------------------------------------

class PretrainedSpeakerEncoder(nn.Module):
    def __init__(self, pretrained_spk_dim: int, speaker_embed_dim: int):
        super().__init__()
        hidden = max(speaker_embed_dim, pretrained_spk_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(pretrained_spk_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, speaker_embed_dim),
        )

    def forward(self, speaker_embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(speaker_embed)


# ---------------------------------------------------------------------------
# Entropy-gated soft cross-attention fuser
# ---------------------------------------------------------------------------

class EntropyCrossAttentionFuser(nn.Module):
    """
    Post-attention entropy blend:

        e_t    = noisy_proj(softmax(logits/T) @ embed.weight)   [B, T, D]
        c_t    = LayerNorm(h_av + CrossAttn(Q=h_av, K=V=e_t))  [B, T, D]
        λ_t    = 1 − H_norm(softmax(logits/T))                  [B, T, 1]
        h̃_t   = λ_t · e_t  +  (1 − λ_t) · c_t

    e_t enters cross-attention unscaled. The entropy gate is applied only to
    the final blend, not to the K/V inputs.

    When audio is clear  (peaked p): λ_t≈1 → h̃_t ≈ e_t  (noisy embed dominates).
    When audio is noisy  (flat p):   λ_t≈0 → h̃_t ≈ c_t  (AV cross-attn dominates).
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
        self._log_V        = math.log(mimi_vocab)  # normalisation constant

        self.noisy_embed = nn.Embedding(mimi_vocab, token_embed_dim)
        self.noisy_proj  = nn.Linear(token_embed_dim, d_model)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.noisy_proj.weight)
        nn.init.zeros_(self.noisy_proj.bias)

        # Entropy-seeded learned gate:
        #   λ_t = σ(gate_w · λ̂_entropy + gate_mlp([e_t ‖ h_av]))
        # gate_w init=1 → full entropy weight at step 0.
        # gate_mlp last layer zeroed → MLP residual ≈ 0 at step 0,
        # so the gate initialises to σ(λ̂_entropy) ≈ entropy prior.
        self.gate_w   = nn.Parameter(torch.ones(1))
        hidden = max(d_model // 4, 64)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        if speaker_embed_dim > 0:
            self.speaker_logit_bias = nn.Linear(speaker_embed_dim, mimi_vocab, bias=False)
        else:
            self.speaker_logit_bias = None

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
        B    = av_features.size(0)
        T_q  = av_features.size(1)
        T_kv = noisy_logits.size(1)
        K    = self.prefix_length

        if self.speaker_logit_bias is not None and spk_emb is not None:
            bias = self.speaker_logit_bias(spk_emb.to(av_features.dtype))
            noisy_logits = noisy_logits + bias.unsqueeze(1)

        # softmax distribution over codebook
        p = F.softmax(noisy_logits.float() / self.temperature, dim=-1)  # [B, T_kv, V]
        p = p.to(av_features.dtype)

        # normalized entropy prior: λ̂_t ∈ [0, 1], 0=uncertain, 1=certain
        H = -(p * p.clamp(min=1e-8).log()).sum(dim=-1)      # [B, T_kv]
        lam_entropy = (1.0 - H / self._log_V).unsqueeze(-1)  # [B, T_kv, 1]

        # soft codebook lookup → project to d_model (unscaled — gate is post-attn)
        e_t = self.noisy_proj(p @ self.noisy_embed.weight)  # [B, T_kv, D]

        # entropy-seeded learned gate: σ(w · λ̂_entropy + MLP([e_t ‖ h_av]))
        # MLP is zero-inited → starts at σ(w · λ̂_entropy) ≈ entropy prior
        gate_in = torch.cat([e_t, av_features], dim=-1)      # [B, T_kv, 2D]
        lam = torch.sigmoid(
            self.gate_w * lam_entropy + self.gate_mlp(gate_in)
        )                                                     # [B, T_kv, 1]

        # Prefix tuning: prepend K vectors to K/V only; gate uses T_kv frames only
        if K > 0:
            if self.prefix_k_proj is not None and spk_emb is not None:
                s = spk_emb.to(av_features.dtype)
                prefix_k = self.prefix_k_proj(s).view(B, K, -1)
                prefix_v = self.prefix_v_proj(s).view(B, K, -1)
            elif self.prefix_k_shared is not None:
                prefix_k = self.prefix_k_shared.expand(B, -1, -1)
                prefix_v = self.prefix_v_shared.expand(B, -1, -1)
            else:
                D = av_features.size(-1)
                prefix_k = torch.zeros(B, K, D, device=av_features.device, dtype=av_features.dtype)
                prefix_v = torch.zeros_like(prefix_k)

            key   = torch.cat([prefix_k, e_t], dim=1)
            value = torch.cat([prefix_v, e_t], dim=1)
        else:
            key   = e_t
            value = e_t

        # c_t = LayerNorm(h_av + CrossAttn(Q=h_av, K=V=e_t))
        attn_mask = self._causal_mask(T_q, T_kv, K, av_features.device, av_features.dtype)
        attended, _ = self.cross_attn(
            query=av_features, key=key, value=value, attn_mask=attn_mask,
        )
        c_t = self.norm(av_features + attended)  # [B, T_q, D]

        # post-attention entropy blend: λ_t · e_t + (1 − λ_t) · c_t
        # T_q == T_kv == T_down after the 25 Hz→12.5 Hz downsampling in the encoder
        self.last_lam = lam.detach().cpu()   # [B, T, 1] — logged for analysis
        return lam * e_t + (1.0 - lam) * c_t


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CrossAttnEntEncoder(FairseqEncoder):
    """
    Causal AV-HuBERT backbone + entropy-gated soft cross-attention fuser.
    """

    def __init__(self, cfg: AVHubertCrossAttnEntConfig):
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
            logger.info(f"[CrossAttnEntEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")

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

        _raw_cond = cfg.speaker_cond
        if _raw_cond is None or (isinstance(_raw_cond, str) and _raw_cond.lower() == "none"):
            self.speaker_cond = "none"
        else:
            self.speaker_cond = str(_raw_cond)

        _use_spk = self.speaker_cond != "none"
        spk_dim  = cfg.speaker_embed_dim if _use_spk else 0

        if _use_spk:
            self.speaker_encoder = PretrainedSpeakerEncoder(
                pretrained_spk_dim=cfg.pretrained_spk_dim,
                speaker_embed_dim=cfg.speaker_embed_dim)
        else:
            self.speaker_encoder = None

        if self.speaker_cond in ("film", "film_prefix"):
            self.film_gamma = nn.Linear(cfg.speaker_embed_dim, d)
            self.film_beta  = nn.Linear(cfg.speaker_embed_dim, d)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        else:
            self.film_gamma = None
            self.film_beta  = None

        if self.speaker_cond == "adaln":
            self.adaln_norm  = nn.LayerNorm(d)
            self.adaln_gamma = nn.Linear(cfg.speaker_embed_dim, d)
            self.adaln_beta  = nn.Linear(cfg.speaker_embed_dim, d)
            nn.init.zeros_(self.adaln_gamma.weight)
            nn.init.ones_(self.adaln_gamma.bias)
            nn.init.zeros_(self.adaln_beta.weight)
            nn.init.zeros_(self.adaln_beta.bias)
        else:
            self.adaln_norm  = None
            self.adaln_gamma = None
            self.adaln_beta  = None

        if self.speaker_cond == "concat":
            self._concat_spk_dim = cfg.speaker_embed_dim
            self.concat_proj = nn.Linear(d + cfg.speaker_embed_dim, d)
            nn.init.eye_(self.concat_proj.weight[:, :d])
            nn.init.zeros_(self.concat_proj.weight[:, d:])
            nn.init.zeros_(self.concat_proj.bias)
        else:
            self._concat_spk_dim = 0
            self.concat_proj = None

        self._warned_missing_spk = False

        _kv_spk_dim = spk_dim if cfg.speaker_cond in ("pretrained_spk", "film_prefix") else 0
        self.cross_attn_fuser = EntropyCrossAttentionFuser(
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
        if self.speaker_cond != "none":
            speaker_embed = kwargs.get("speaker_embed", None)
            if speaker_embed is not None and self.speaker_encoder is not None:
                spk_emb = self.speaker_encoder(speaker_embed.to(x.dtype))
            if spk_emb is None and not self._warned_missing_spk:
                logger.warning(
                    "[speaker_cond=%s] speaker_embed is None — conditioning disabled for "
                    "this run. Verify ARCFACE_EMBED_ROOT is set and uid paths match.",
                    self.speaker_cond,
                )
                self._warned_missing_spk = True

        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]
            if noisy_logits is not None:
                noisy_logits = noisy_logits[:, :-1, :]

        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)

        T_down = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        if spk_emb is not None and self.film_gamma is not None:
            gamma = self.film_gamma(spk_emb).unsqueeze(1)
            beta  = self.film_beta(spk_emb).unsqueeze(1)
            x = gamma * x + beta

        if self.adaln_gamma is not None:
            x_n = self.adaln_norm(x)
            if spk_emb is not None:
                gamma = self.adaln_gamma(spk_emb).unsqueeze(1)
                beta  = self.adaln_beta(spk_emb).unsqueeze(1)
                x = gamma * x_n + beta
            else:
                x = x_n

        if self.concat_proj is not None:
            if spk_emb is not None:
                spk_t = spk_emb.unsqueeze(1).expand(-1, x.size(1), -1)
            else:
                spk_t = x.new_zeros(x.size(0), x.size(1), self._concat_spk_dim)
            x = self.concat_proj(torch.cat([x, spk_t], dim=-1))

        av_features_ds = x.detach() if self.pred_coding_head is not None else None

        _fuser_spk = spk_emb if self.speaker_cond in ("pretrained_spk", "film_prefix") else None
        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]
            x = self.cross_attn_fuser(x, noisy_logits_ds, spk_emb=_fuser_spk)

        pred_coding_loss = None
        if self.training and self.pred_coding_head is not None and av_features_ds is not None:
            pred_coding_loss = self.pred_coding_head(x, av_features_ds)

        x = self.final_dropout(x)
        logits_12p5 = self.mimi_head(x)

        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             x,
            "logits_12p5":          logits_12p5,
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

@register_model("av_hubert_crossattn_ent", dataclass=AVHubertCrossAttnEntConfig)
class AVHubertCrossAttnEntModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertCrossAttnEntConfig, task: FairseqTask):
        encoder = CrossAttnEntEncoder(cfg)
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
