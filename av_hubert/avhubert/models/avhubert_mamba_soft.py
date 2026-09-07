# AV-HuBERT + Mamba cross-fuser for streaming Mimi semantic token prediction.
#
# Replaces the cross-attention fuser in avhubert_crossattn_soft.py with a
# selective state-space model (Mamba-style). Key advantages over cross-attention:
#
#   - True streaming: O(1) state per step, no future context needed
#   - Infinite effective past context compressed into fixed-size hidden state
#   - AV features condition the SSM input gate (no causal mask required)
#
# Architecture:
#   noisy_ctx [B, T, D]  ──► MambaCrossFuser ◄── av_features [B, T, D]
#                                  │
#                             [B, T, D]  ──► MimiHead ──► logits [B, T, V]
#
# The Mamba block is implemented in pure PyTorch (no CUDA extension required).
# Selective parameters B, C, Δ are projected from noisy_ctx; av_features are
# injected as an additive bias on the SSM input before the selective projection.
#
# Drop-in replacement for avhubert_crossattn_soft: same fairseq task/criterion,
# same data pipeline (noisy_logits required in net_input).

import sys
import logging
import contextlib
import math
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

logger = logging.getLogger(__name__)

try:
    from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
except ImportError:
    MASKING_DISTRIBUTION_CHOICES = Any


# ---------------------------------------------------------------------------
# Pure-PyTorch selective SSM (Mamba core)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Selective state-space model core (Mamba-style), pure PyTorch.

    Processes one sequence step at a time or the full sequence in training.
    At inference call step() for O(1) streaming.

    Args:
        d_model  : model dimension D
        d_state  : SSM state dimension N (default 16)
        d_conv   : causal depthwise conv width (default 4)
        expand   : inner expansion factor (default 2)
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.d_inner  = d_model * expand

        # input projection: x → [z, x_inner]
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # causal depthwise conv
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        # selective projections (input-dependent B, C, Δ)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, log_Δ
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)                  # Δ → d_inner
        # output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

        # A: log-initialized diagonal state matrix [d_inner, d_state]
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.dt_proj.bias, math.log(0.001))  # small initial Δ

    # ------------------------------------------------------------------
    # Full-sequence forward (training)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        residual = x
        B, T, D  = x.shape

        xz = self.in_proj(x)                          # [B, T, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)              # each [B, T, d_inner]

        # causal depthwise conv (trim right padding)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)                       # [B, T, d_inner]

        # selective parameters from x
        bcd    = self.x_proj(x_conv)                  # [B, T, 2N+1]
        B_sel  = bcd[:, :, :self.d_state]             # [B, T, N]
        C_sel  = bcd[:, :, self.d_state:2*self.d_state]
        log_dt = bcd[:, :, -1:]                       # [B, T, 1]
        dt     = F.softplus(self.dt_proj(log_dt))     # [B, T, d_inner]

        # run SSM in float32 for numerical stability, cast back after
        orig_dtype = x.dtype
        A   = -torch.exp(self.A_log.float())                    # [d_inner, N]
        dt  = dt.float()
        B_sel = B_sel.float()
        C_sel = C_sel.float()
        x_conv_f = x_conv.float()

        dA = torch.exp(dt.unsqueeze(-1) * A)                    # [B, T, d_inner, N]
        dB = dt.unsqueeze(-1) * B_sel.unsqueeze(2)              # [B, T, d_inner, N]

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_conv_f[:, t].unsqueeze(-1)
            y_t = (h * C_sel[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1).to(orig_dtype)               # [B, T, d_inner]

        y = y * F.silu(z)                             # gating
        y = self.out_proj(y)                          # [B, T, D]
        return self.norm(residual + y)

    # ------------------------------------------------------------------
    # Single-step streaming forward
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device, dtype):
        """Returns initial SSM hidden state for streaming."""
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=torch.float32)
        conv_buf = torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        return {"h": h, "conv_buf": conv_buf}

    def step(self, x_t: torch.Tensor, state: dict) -> tuple:
        """
        x_t   : [B, D]
        state : dict returned by init_state / previous step
        returns (y_t [B, D], new_state)
        """
        h, conv_buf = state["h"], state["conv_buf"]

        xz    = self.in_proj(x_t)                     # [B, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)

        # causal conv: append new input, apply, drop oldest
        x_buf = torch.cat([conv_buf, x_inner.unsqueeze(-1)], dim=-1)  # [B, d_inner, d_conv]
        weight = self.conv1d.weight.squeeze(1)                          # [d_inner, d_conv]
        bias   = self.conv1d.bias
        x_conv = (x_buf * weight.unsqueeze(0)).sum(-1) + bias          # [B, d_inner]
        x_conv = F.silu(x_conv)
        new_conv_buf = x_buf[:, :, 1:]                                  # drop oldest

        bcd    = self.x_proj(x_conv)
        B_sel  = bcd[:, :self.d_state]
        C_sel  = bcd[:, self.d_state:2*self.d_state]
        log_dt = bcd[:, -1:]
        dt     = F.softplus(self.dt_proj(log_dt))      # [B, d_inner]

        orig_dtype = x_t.dtype
        A     = -torch.exp(self.A_log.float())
        dA    = torch.exp(dt.float().unsqueeze(-1) * A)
        dB    = dt.float().unsqueeze(-1) * B_sel.float().unsqueeze(1)

        new_h = dA * h.float() + dB * x_conv.float().unsqueeze(-1)
        y_t   = (new_h * C_sel.float().unsqueeze(1)).sum(-1).to(orig_dtype)

        y_t = y_t * F.silu(z)
        y_t = self.out_proj(y_t)
        y_t = self.norm(x_t + y_t)

        new_state = {"h": new_h, "conv_buf": new_conv_buf}
        return y_t, new_state


# ---------------------------------------------------------------------------
# Predictive coding auxiliary head
# ---------------------------------------------------------------------------

class PredictiveCodingHead(nn.Module):
    """
    Auxiliary head that predicts the next k AV-HuBERT hidden states from the
    current Mamba output. Used only during training — zero inference cost.

    Forces the SSM recurrent state to encode anticipatory structure, compensating
    for the absence of future context in a fully-causal model.

    Input  : x         [B, T, D] — Mamba fuser output at 12.5 Hz
    Target : av_target [B, T, D] — AV-HuBERT features at 12.5 Hz (stop-gradient)
    Loss   : mean cosine-similarity loss over k future steps
    """

    def __init__(self, d_model: int, k: int = 4):
        super().__init__()
        self.k = k
        # one linear head per future step (lightweight)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(k)
        ])
        for h in self.heads:
            nn.init.eye_(h.weight)   # start as identity — predict "no change"

    def forward(self, x: torch.Tensor, av_target: torch.Tensor) -> torch.Tensor:
        """Returns scalar aux loss."""
        T    = x.size(1)
        loss = x.new_zeros(1).squeeze()
        n    = 0
        for i, head in enumerate(self.heads):
            step = i + 1
            if T <= step:
                continue
            pred = head(x[:, :T - step])              # [B, T-step, D]
            tgt  = av_target[:, step:].detach()       # stop-gradient on target
            cos  = F.cosine_similarity(pred, tgt, dim=-1)   # [B, T-step]
            loss = loss + (1.0 - cos).mean()
            n   += 1
        return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Mamba cross-fuser
# ---------------------------------------------------------------------------

class MambaCrossFuser(nn.Module):
    """
    Mamba-based cross-fuser replacing SoftCrossAttentionFuser.

    noisy_logits [B, T, V]  → soft embed → noisy_ctx [B, T, D]
    av_features  [B, T, D]  → additive conditioning on noisy_ctx input
    SSM processes the conditioned noisy_ctx sequence.
    Output [B, T, D] is added back to av_features (residual).
    """

    def __init__(self, d_model: int, mimi_vocab: int,
                 token_embed_dim: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.mimi_vocab  = mimi_vocab

        self.noisy_embed = nn.Embedding(mimi_vocab, token_embed_dim)
        self.noisy_proj  = nn.Linear(token_embed_dim, d_model)

        # AV features condition the noisy ctx via a learned gate
        self.av_gate     = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        self.ssm  = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.noisy_proj.weight)
        nn.init.zeros_(self.noisy_proj.bias)

    def forward(self, av_features: torch.Tensor,
                noisy_logits: torch.Tensor) -> torch.Tensor:
        """
        av_features  : [B, T, D]
        noisy_logits : [B, T, V]
        returns      : [B, T, D]
        """
        # soft weighted codebook lookup
        weights   = F.softmax(noisy_logits.float() / self.temperature, dim=-1).to(av_features.dtype)
        soft_emb  = weights @ self.noisy_embed.weight   # [B, T, token_embed_dim]
        noisy_ctx = self.noisy_proj(soft_emb)            # [B, T, D]

        # AV conditioning: gate the noisy context
        noisy_ctx = noisy_ctx + self.av_gate(av_features)

        # selective SSM scan
        out = self.ssm(noisy_ctx)                        # [B, T, D]
        out = self.drop(out)

        # residual fusion into AV features
        return self.norm(av_features + out)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertMambaSoftConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING)
    no_pretrained_weights: bool = field(default=False)

    dropout_input:       float = field(default=0.0)
    final_dropout:       float = field(default=0.0)
    dropout:             float = field(default=0.0)
    attention_dropout:   float = field(default=0.0)
    activation_dropout:  float = field(default=0.0)

    apply_mask:              bool  = field(default=False)
    mask_length:             int   = field(default=10)
    mask_prob:               float = field(default=0.5)
    mask_selection:          MASKING_DISTRIBUTION_CHOICES = field(default="static")
    mask_other:              float = field(default=0.0)
    no_mask_overlap:         bool  = field(default=False)
    mask_channel_length:     int   = field(default=10)
    mask_channel_prob:       float = field(default=0.0)
    mask_channel_selection:  MASKING_DISTRIBUTION_CHOICES = field(default="static")
    mask_channel_other:      float = field(default=0.0)
    no_mask_channel_overlap: bool  = field(default=False)

    freeze_finetune_updates: int   = field(default=0)
    feature_grad_mult:       float = field(default=1.0)
    layerdrop:               float = field(default=0.0)
    lookahead_frames:        int   = field(
        default=0,
        metadata={"help": "AV-HuBERT backbone lookahead. 0=fully causal."},
    )

    mimi_vocab_size: int = field(default=2048)
    head_hidden_dim: int = field(default=0)

    # Mamba fuser hyperparameters
    token_embed_dim: int = field(
        default=256,
        metadata={"help": "embedding dim for noisy Mimi codebook lookup"},
    )
    mamba_d_state: int = field(
        default=16,
        metadata={"help": "SSM state dimension N"},
    )
    mamba_d_conv: int = field(
        default=4,
        metadata={"help": "causal depthwise conv width"},
    )
    mamba_expand: int = field(
        default=2,
        metadata={"help": "inner expansion factor"},
    )
    mamba_dropout: float = field(default=0.1)
    logit_temperature: float = field(default=1.0)

    # Predictive coding auxiliary loss
    pred_coding_k: int = field(
        default=4,
        metadata={"help": "number of future frames to predict (at 12.5 Hz). 0 = disabled."},
    )

    normalize: bool = field(default=False)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MambaSoftEncoder(FairseqEncoder):

    def __init__(self, cfg: AVHubertMambaSoftConfig):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout":                cfg.dropout,
            "activation_dropout":     cfg.activation_dropout,
            "dropout_input":          cfg.dropout_input,
            "attention_dropout":      cfg.attention_dropout,
            "mask_length":            cfg.mask_length,
            "mask_prob":              cfg.mask_prob,
            "mask_selection":         cfg.mask_selection,
            "mask_other":             cfg.mask_other,
            "no_mask_overlap":        cfg.no_mask_overlap,
            "mask_channel_length":    cfg.mask_channel_length,
            "mask_channel_prob":      cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other":     cfg.mask_channel_other,
            "no_mask_channel_overlap":cfg.no_mask_channel_overlap,
            "encoder_layerdrop":      cfg.layerdrop,
            "feature_grad_mult":      cfg.feature_grad_mult,
        }

        state    = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
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
        merged_model_cfg  = OmegaConf.merge(default_model_cfg, w2v_args.model)
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
            logger.info(f"[MambaSoftEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")

        causal_model.remove_pretraining_modules()

        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model             = causal_model
        self.final_dropout         = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates           = 0

        # causal temporal downsample 25 Hz → 12.5 Hz
        self.temporal_downsample = nn.Conv1d(
            in_channels=d, out_channels=d,
            kernel_size=2, stride=2, padding=0,
        )

        # Mamba cross-fuser (replaces SoftCrossAttentionFuser)
        self.mamba_fuser = MambaCrossFuser(
            d_model        = d,
            mimi_vocab     = cfg.mimi_vocab_size,
            token_embed_dim= cfg.token_embed_dim,
            d_state        = cfg.mamba_d_state,
            d_conv         = cfg.mamba_d_conv,
            expand         = cfg.mamba_expand,
            dropout        = cfg.mamba_dropout,
            temperature    = cfg.logit_temperature,
        )

        self.mimi_head = MimiHead(
            in_dim    = d,
            out_dim   = cfg.mimi_vocab_size,
            hidden_dim= cfg.head_hidden_dim,
            dropout   = cfg.final_dropout,
        )

        # Predictive coding auxiliary head (training only)
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
        B    = padding_mask.size(0)
        left = padding_mask.new_zeros(B, 1)
        pm   = torch.cat([left, padding_mask], dim=1)
        return pm[:, 1::2][:, :T_out]

    def _upsample_by_repeat(self, x, target_len):
        x   = x.repeat_interleave(2, dim=1)
        cur = x.size(1)
        if cur > target_len:
            x = x[:, :target_len]
        elif cur < target_len:
            last = x[:, -1:].expand(x.size(0), target_len - cur, x.size(2))
            x    = torch.cat([x, last], dim=1)
        return x

    def _upsample_padding_mask(self, pm, target_len):
        if pm is None:
            return None
        pm  = pm.repeat_interleave(2, dim=1)
        cur = pm.size(1)
        if cur > target_len:
            pm = pm[:, :target_len]
        elif cur < target_len:
            last = pm[:, -1:].expand(pm.size(0), target_len - cur)
            pm   = torch.cat([pm, last], dim=1)
        return pm

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        noisy_logits = kwargs.get("noisy_logits", None)

        if noisy_logits is None and self.training:
            raise RuntimeError(
                "[MambaSoftEncoder] noisy_logits missing from net_input during training. "
                "Check that NOISY_LOGITS_ROOT is set and the dataset is loading .npy files."
            )

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(
                source=source,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training,
            )

        orig_len = x.size(1)

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

        T_down       = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        # save AV features before fuser as predictive coding target
        av_features_ds = x.detach() if self.pred_coding_head is not None else None

        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]  # [B, T/2, V]
            x = self.mamba_fuser(x, noisy_logits_ds)                   # [B, T/2, D]

        # predictive coding aux loss (training only)
        pred_coding_loss = None
        if self.training and self.pred_coding_head is not None and av_features_ds is not None:
            pred_coding_loss = self.pred_coding_head(x, av_features_ds)

        x      = self.final_dropout(x)
        logits = self.mimi_head(x)                                      # [B, T/2, V]
        logits = self._upsample_by_repeat(logits, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)   # [T, B, V]

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             x,              # [B, T/2, D] for aux losses in criterion
            "pred_coding_loss":     pred_coding_loss,  # scalar or None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_logits = encoder_out["encoder_out"].index_select(1, new_order)
        new_pm = None
        if encoder_out["encoder_padding_mask"] is not None:
            new_pm = encoder_out["encoder_padding_mask"].index_select(0, new_order)
        return {
            "encoder_out":         new_logits,
            "encoder_padding_mask": new_pm,
            "padding_mask":        new_pm,
        }


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@register_model("av_hubert_mamba_soft", dataclass=AVHubertMambaSoftConfig)
class AVHubertMambaSoftModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertMambaSoftConfig, task: FairseqTask):
        encoder = MambaSoftEncoder(cfg)
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
