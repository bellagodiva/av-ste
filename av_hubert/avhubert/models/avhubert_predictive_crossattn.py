# Predictive Cross-Attention AV-HuBERT  —  JEPA-style latent prediction
#
# Architecture (causal=0, fully streaming-safe):
#
#   1. Causal AV-HuBERT backbone → h_t  [B, T, D]  (lookahead=0)
#   2. Online encoder:   z_t  = L2_norm(online_proj(h_t))          [B, T, d_lat]
#      EMA    encoder:   z̄_t  = L2_norm(ema_proj(h_t))  (stop-grad, EMA weights)
#   3. Causal predictor: [z_{<=t}] → ẑ_{t+1..t+L}                [B, T, L, d_lat]
#      Loss:  L_pred = MSE(ẑ_{t+l}, z̄_{t+l})   l=1..L
#   4. Augmented query:  [h_t ; ẑ_{t+1} ; ... ; ẑ_{t+L}] → linear → D
#      (shifted by 1: at step t we use predictions made at t-1, never future h)
#   5. Soft cross-attention fuser (noisy logits as K/V) → clean Mimi logits
#
# At inference: EMA encoder & predictor run on past frames only → no future leakage.
# Compatible with Moshi RingKVCache streaming at 12.5 Hz.
#
# Losses:
#   L_ce   : cross-entropy on clean Mimi token
#   L_pred : JEPA MSE in latent space  (weight: lambda_pred)
#
# Env vars:
#   NOISY_LOGITS_ROOT  — dir of {utt_id}.npy noisy audio logit files

import copy
import sys
import logging
import contextlib
from argparse import Namespace
from dataclasses import dataclass, field

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
from .avhubert_crossattn_soft import SoftCrossAttentionFuser, PretrainedSpeakerEncoder

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# JEPA latent predictor
# ---------------------------------------------------------------------------

class JEPAFuturePredictor(nn.Module):
    """
    JEPA-style future latent predictor.

    Online encoder   : online_proj(h)  → z   [L2-normalised, d_lat]
    EMA encoder      : ema_proj(h)     → z̄   [L2-normalised, stop-grad]
    Causal predictor : z_{<=t} → ẑ_{t+1..t+L}

    Loss: mean MSE(ẑ_{t+l}, z̄_{t+l}.detach())  over l=1..L

    EMA update (call update_ema() after each optimiser step):
        ema_proj ← τ * ema_proj + (1-τ) * online_proj
    """

    def __init__(self, d_model: int, d_lat: int, lookahead: int,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, ema_decay: float = 0.999):
        super().__init__()
        self.lookahead = lookahead
        self.d_lat     = d_lat
        self.ema_decay = ema_decay

        # online projection (trained)
        self.online_proj = nn.Sequential(
            nn.Linear(d_model, d_lat),
            nn.GELU(),
            nn.Linear(d_lat, d_lat),
        )
        # EMA projection (not trained directly — updated via update_ema())
        self.ema_proj = copy.deepcopy(self.online_proj)
        for p in self.ema_proj.parameters():
            p.requires_grad_(False)

        # causal transformer predictor operating in latent space
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_lat, nhead=n_heads,
            dim_feedforward=d_lat * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # per-step head: d_lat → L * d_lat
        self.pred_head = nn.Linear(d_lat, lookahead * d_lat)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

        for m in self.online_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_ema(self):
        """Exponential moving average update of ema_proj weights."""
        τ = self.ema_decay
        for p_online, p_ema in zip(self.online_proj.parameters(),
                                   self.ema_proj.parameters()):
            p_ema.data.mul_(τ).add_(p_online.data, alpha=1 - τ)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    def encode_online(self, h: torch.Tensor) -> torch.Tensor:
        """h [B,T,D] → z [B,T,d_lat], L2-normalised."""
        return F.normalize(self.online_proj(h), dim=-1)

    @torch.no_grad()
    def encode_ema(self, h: torch.Tensor) -> torch.Tensor:
        """h [B,T,D] → z̄ [B,T,d_lat], L2-normalised, stop-grad."""
        return F.normalize(self.ema_proj(h), dim=-1)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : [B, T, d_lat]  online latents
        Returns ẑ : [B, T, L, d_lat]  predicted future latents at each step.
        """
        B, T, _ = z.shape
        mask  = self._causal_mask(T, z.device)
        ctx   = self.predictor(z, mask=mask)          # [B, T, d_lat]
        pred  = self.pred_head(ctx)                   # [B, T, L*d_lat]
        return pred.view(B, T, self.lookahead, self.d_lat)

    def prediction_loss(self, h: torch.Tensor) -> torch.Tensor:
        """
        Full JEPA loss on a sequence h [B, T, D].
        Returns scalar MSE averaged over valid (t, l) pairs.
        """
        B, T, D = h.shape
        L = self.lookahead
        if T <= L:
            return h.new_zeros(1).squeeze()

        z_online = self.encode_online(h)      # [B, T, d_lat]
        z_ema    = self.encode_ema(h)         # [B, T, d_lat]  stop-grad
        z_pred   = self.predict(z_online)     # [B, T, L, d_lat]

        loss  = h.new_zeros(1).squeeze()
        count = 0
        for l in range(1, L + 1):
            T_valid  = T - l
            if T_valid <= 0:
                continue
            pred_l   = z_pred[:, :T_valid, l - 1, :]    # [B, T_valid, d_lat]
            target_l = z_ema[:, l:l + T_valid, :].detach()
            loss  = loss + F.mse_loss(pred_l, target_l)
            count += 1

        return loss / max(count, 1)

    def future_latents_for_query(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted future latents shifted by 1 so that at position t
        we use predictions made at t-1 (strictly causal).

        Returns ẑ_shifted : [B, T, L, d_lat]
        """
        B, T, _ = h.shape
        z      = self.encode_online(h)
        pred   = self.predict(z)   # [B, T, L, d_lat]  — pred[t] = prediction for t+1..t+L

        # shift: at position t use pred[t-1]; pad position 0 with zeros
        shifted = torch.cat([
            torch.zeros(B, 1, self.lookahead, self.d_lat,
                        device=h.device, dtype=h.dtype),
            pred[:, :-1, :, :],
        ], dim=1)  # [B, T, L, d_lat]
        return shifted


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertPredictiveCrossAttnConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING,
                          metadata={"help": "pretrained AV-HuBERT checkpoint"})
    no_pretrained_weights: bool = field(default=False)

    # dropouts
    dropout_input: float        = field(default=0.0)
    final_dropout: float        = field(default=0.0)
    dropout: float              = field(default=0.0)
    attention_dropout: float    = field(default=0.0)
    activation_dropout: float   = field(default=0.0)

    # masking (API compat)
    apply_mask: bool                             = field(default=False)
    mask_length: int                             = field(default=10)
    mask_prob: float                             = field(default=0.5)
    mask_selection: str   = field(default="static")
    mask_other: float                            = field(default=0.0)
    no_mask_overlap: bool                        = field(default=False)
    mask_channel_length: int                     = field(default=10)
    mask_channel_prob: float                     = field(default=0.0)
    mask_channel_selection: str = field(default="static")
    mask_channel_other: float                    = field(default=0.0)
    no_mask_channel_overlap: bool                = field(default=False)

    freeze_finetune_updates: int = field(default=0)
    feature_grad_mult: float     = field(default=1.0)
    layerdrop: float             = field(default=0.0)
    lookahead_frames: int        = field(default=0)   # always 0

    mimi_vocab_size: int = field(default=2048)
    head_hidden_dim: int = field(default=0)

    # Cross-attention fuser
    token_embed_dim: int      = field(default=256)
    cross_attn_heads: int     = field(default=4)
    cross_attn_dropout: float = field(default=0.1)
    logit_temperature: float  = field(default=1.0)

    # JEPA predictor
    ffp_lookahead: int   = field(default=4,
                                 metadata={"help": "future frames to predict (L)"})
    ffp_d_lat: int       = field(default=256,
                                 metadata={"help": "JEPA latent dimension"})
    ffp_n_heads: int     = field(default=4)
    ffp_n_layers: int    = field(default=2)
    ffp_dropout: float   = field(default=0.1)
    ffp_ema_decay: float = field(default=0.999,
                                 metadata={"help": "EMA decay τ for target encoder"})
    lambda_pred: float   = field(default=1.0,
                                 metadata={"help": "weight on JEPA MSE loss"})

    # ablation: set False to train predictor as aux only, no query augmentation
    use_predicted_future: bool = field(default=True)

    # Speaker conditioning
    speaker_cond: str       = field(default="none")
    speaker_embed_dim: int  = field(default=256)
    pretrained_spk_dim: int = field(default=512)
    prefix_length: int      = field(default=0)

    normalize: bool = field(default=False)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PredictiveCrossAttnEncoder(FairseqEncoder):

    def __init__(self, cfg: AVHubertPredictiveCrossAttnConfig):
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
        merged_model_cfg = OmegaConf.merge(default_model_cfg, w2v_args.model)
        OmegaConf.set_struct(merged_model_cfg, False)
        merged_model_cfg.lookahead_frames = 0

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
            logger.info(f"[PredCrossAttn] missing={len(missing)}, unexpected={len(unexpected)}")

        causal_model.remove_pretraining_modules()
        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model             = causal_model
        self.final_dropout         = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates           = 0

        # 25 Hz → 12.5 Hz causal downsample
        self.temporal_downsample = nn.Conv1d(d, d, kernel_size=2, stride=2, padding=0)

        # JEPA future predictor
        self.jepa = JEPAFuturePredictor(
            d_model=d,
            d_lat=cfg.ffp_d_lat,
            lookahead=cfg.ffp_lookahead,
            n_heads=cfg.ffp_n_heads,
            n_layers=cfg.ffp_n_layers,
            dropout=cfg.ffp_dropout,
            ema_decay=cfg.ffp_ema_decay,
        )
        self.ffp_lookahead       = cfg.ffp_lookahead
        self.lambda_pred         = cfg.lambda_pred
        self.use_predicted_future = cfg.use_predicted_future

        # merge h_t + L predicted latents → d_model for cross-attn query
        if cfg.use_predicted_future:
            self.future_merge = nn.Linear(d + cfg.ffp_lookahead * cfg.ffp_d_lat, d)
            nn.init.xavier_uniform_(self.future_merge.weight)
            nn.init.zeros_(self.future_merge.bias)
        else:
            self.future_merge = None

        # speaker conditioning
        self.speaker_cond = cfg.speaker_cond
        spk_dim = cfg.speaker_embed_dim if cfg.speaker_cond != "none" else 0
        if cfg.speaker_cond == "pretrained_spk":
            self.speaker_encoder = PretrainedSpeakerEncoder(
                pretrained_spk_dim=cfg.pretrained_spk_dim,
                speaker_embed_dim=cfg.speaker_embed_dim,
            )
        else:
            self.speaker_encoder = None

        # soft cross-attention fuser
        self.cross_attn_fuser = SoftCrossAttentionFuser(
            d_model=d,
            mimi_vocab=cfg.mimi_vocab_size,
            token_embed_dim=cfg.token_embed_dim,
            n_heads=cfg.cross_attn_heads,
            dropout=cfg.cross_attn_dropout,
            lookahead=0,
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

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        # EMA update after every optimiser step
        if self.training:
            self.jepa.update_ema()

    def _downsample_padding_mask(self, padding_mask, T_out):
        if padding_mask is None:
            return None
        B    = padding_mask.size(0)
        left = padding_mask.new_zeros(B, 1)
        pm   = torch.cat([left, padding_mask], dim=1)
        return pm[:, 1::2][:, :T_out]

    def _upsample_by_repeat(self, x, target_len):
        x = x.repeat_interleave(2, dim=1)
        cur = x.size(1)
        if cur > target_len:
            x = x[:, :target_len]
        elif cur < target_len:
            x = torch.cat([x, x[:, -1:].expand(x.size(0), target_len - cur, x.size(2))], dim=1)
        return x

    def _upsample_padding_mask(self, pm, target_len):
        if pm is None:
            return None
        pm = pm.repeat_interleave(2, dim=1)
        cur = pm.size(1)
        if cur > target_len:
            pm = pm[:, :target_len]
        elif cur < target_len:
            pm = torch.cat([pm, pm[:, -1:].expand(pm.size(0), target_len - cur)], dim=1)
        return pm

    def _build_query(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, D]  causal AV features at 12.5 Hz
        Returns augmented query [B, T, D].
        Uses JEPA predictions made at t-1 for positions t (strictly causal shift).
        """
        B, T, D = x.shape
        shifted = self.jepa.future_latents_for_query(x)  # [B, T, L, d_lat]
        # flatten L latents per step: [B, T, L*d_lat]
        flat = shifted.view(B, T, self.ffp_lookahead * self.jepa.d_lat)
        aug  = torch.cat([x, flat], dim=-1)              # [B, T, D + L*d_lat]
        return self.future_merge(aug)                    # [B, T, D]

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

        # 25 Hz → 12.5 Hz
        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)   # [B, T/2, D]

        T_down       = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        # --- JEPA prediction loss (training only) ---
        pred_loss = None
        if self.training:
            pred_loss = self.jepa.prediction_loss(x) * self.lambda_pred

        # --- Build cross-attention query (augmented with predicted future) ---
        if self.use_predicted_future:
            query = self._build_query(x)
        else:
            query = x

        # --- Soft cross-attention fuser ---
        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]
            fused = self.cross_attn_fuser(query, noisy_logits_ds, spk_emb=spk_emb)
        else:
            fused = query

        fused       = self.final_dropout(fused)
        logits_12p5 = self.mimi_head(fused)

        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             fused,
            "pred_coding_loss":     pred_loss,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_logits = encoder_out["encoder_out"].index_select(1, new_order)
        new_pm = None
        if encoder_out["encoder_padding_mask"] is not None:
            new_pm = encoder_out["encoder_padding_mask"].index_select(0, new_order)
        return {"encoder_out": new_logits,
                "encoder_padding_mask": new_pm,
                "padding_mask": new_pm}


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@register_model("av_hubert_predictive_crossattn", dataclass=AVHubertPredictiveCrossAttnConfig)
class AVHubertPredictiveCrossAttnModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertPredictiveCrossAttnConfig, task: FairseqTask):
        return cls(PredictiveCrossAttnEncoder(cfg))

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
