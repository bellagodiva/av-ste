# Noisy-Logit-Predictive Cross-Attention AV-HuBERT  (causal=0, streaming-safe)
#
# WHY this works differently from the JEPA approach (avhubert_predictive_crossattn):
#
#   JEPA predicted future AV-HuBERT *features* and used them to augment the Query.
#   But the bottleneck in 0-lookahead is the K/V side of the cross-attention:
#   with 4 lookahead frames the fuser can attend to *noisy logits at t+1..t+4*,
#   providing future acoustic evidence that directly disambiguates the token at t.
#   Augmenting Q doesn't help when K/V contains no future information.
#
# THIS MODEL:
#   Predicts future noisy-logit soft embeddings and injects them as extra K/V slots.
#   At streaming time, the predictor runs causally on frames 0..t and outputs
#   predictions for t+1..t+L — no future leakage.
#
# Architecture:
#   1. Causal AV-HuBERT backbone → h_t  [B, T, D]   (lookahead=0)
#   2. noisy_logits → softmax → codebook lookup → soft_emb  [B, T, E]
#   3. NoisyLogitPredictor (causal transformer in E-dim space):
#        soft_emb[0..t] → predicted soft_emb for t+1..t+L  [B, T, L, E]
#      Loss: L_pred = cosine(pred, real_future_soft_emb.stop_grad)
#   4. Augmented cross-attention:
#        Q   = h_t  [B, T, D]
#        K/V = [real_noisy_kv[≤t], predicted_future_kv[t+1..t+L from context≤t]]
#      Mask: lower-triangular for real part, block-diagonal for predicted part.
#   5. Mimi head → clean token logits
#
# K/V bank layout (at query position t):
#   positions 0 .. T-1          : real noisy K/V, causal (attend if kv_pos ≤ t)
#   positions T + l*T + t       : l-th future prediction for t (l=0..L-1)
#   → mask allows query t to attend to block [T+l*T+t] for each l
#
# Env vars:
#   NOISY_LOGITS_ROOT  — dir of {utt_id}.npy noisy audio logit files

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
from .avhubert_crossattn_soft import PretrainedSpeakerEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noisy Logit Predictor
# ---------------------------------------------------------------------------

class NoisyLogitPredictor(nn.Module):
    """
    Predicts future noisy-logit soft embeddings from past context.

    Input  : soft_emb [B, T, E]  — softmax(noisy_logits) @ codebook  (current & past)
    Output : [B, T, L, E]        — predicted soft_emb for t+1..t+L at each position t

    The predictor is a small causal transformer operating entirely in E-dim space.
    At position t it has seen frames 0..t (causal mask), so its predictions for
    t+1..t+L are strictly causal — safe for streaming inference.

    Training loss: cosine-similarity loss vs. real future soft_emb (stop-gradient target).
    """

    def __init__(self, embed_dim: int, lookahead: int,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lookahead = lookahead
        self.embed_dim = embed_dim

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # at each t predicts L future embeddings: E → L*E
        self.pred_head = nn.Linear(embed_dim, lookahead * embed_dim)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    def forward(self, soft_emb: torch.Tensor) -> torch.Tensor:
        """
        soft_emb : [B, T, E]
        Returns  : [B, T, L, E]
                   result[:, t, l, :] = predicted soft_emb for position t+l+1,
                   using causal context 0..t.
        """
        B, T, E = soft_emb.shape
        mask = self._causal_mask(T, soft_emb.device)
        ctx  = self.transformer(soft_emb, mask=mask)   # [B, T, E]
        pred = self.pred_head(ctx)                      # [B, T, L*E]
        return pred.view(B, T, self.lookahead, E)

    def prediction_loss(self, soft_emb: torch.Tensor) -> torch.Tensor:
        """
        Cosine-similarity loss: predicted future soft_emb vs. real (stop-gradient).

        For each lookahead step l=1..L:
          pred[:, :T-l, l-1, :]  vs.  soft_emb[:, l:, :].detach()

        Returns scalar mean loss.
        """
        B, T, E = soft_emb.shape
        L = self.lookahead
        if T <= L:
            return soft_emb.new_zeros(1).squeeze()

        pred  = self.forward(soft_emb)   # [B, T, L, E]
        loss  = soft_emb.new_zeros(1).squeeze()
        count = 0
        for l in range(1, L + 1):
            T_valid  = T - l
            if T_valid <= 0:
                continue
            pred_l   = pred[:, :T_valid, l - 1, :]          # [B, T_valid, E]
            target_l = soft_emb[:, l:l + T_valid, :].detach()  # [B, T_valid, E]
            cos      = F.cosine_similarity(pred_l, target_l, dim=-1)  # [B, T_valid]
            loss     = loss + (1.0 - cos).mean()
            count   += 1

        return loss / max(count, 1)


# ---------------------------------------------------------------------------
# Predictive Noisy Cross-Attention Fuser
# ---------------------------------------------------------------------------

class PredictiveNoisyCrossAttnFuser(nn.Module):
    """
    Cross-attention fuser where K/V is extended with predicted future noisy embeddings.

    K/V bank layout for query at position t  (T_kv == T_q == T_down):
      Positions [0 .. T-1]           real noisy K/V          (attend if pos ≤ t)
      Positions [T + 0*T + t]        l=0 future prediction   (only own slot, diagonal)
      Positions [T + 1*T + t]        l=1 future prediction
      ...
      Positions [T + (L-1)*T + t]    l=L-1 future prediction

    The noisy_embed and noisy_proj are shared for both real and predicted embeddings
    so predicted K/V lives in the same D-dimensional space as real K/V.
    """

    def __init__(self, d_model: int, mimi_vocab: int,
                 token_embed_dim: int, lookahead: int,
                 n_heads: int = 4, dropout: float = 0.1,
                 temperature: float = 1.0,
                 pred_n_heads: int = 4, pred_n_layers: int = 2,
                 speaker_embed_dim: int = 0):
        super().__init__()
        self.lookahead   = lookahead
        self.temperature = temperature
        self.embed_dim   = token_embed_dim

        # codebook + projection shared between real and predicted K/V
        self.noisy_embed = nn.Embedding(mimi_vocab, token_embed_dim)
        self.noisy_proj  = nn.Linear(token_embed_dim, d_model)
        nn.init.xavier_uniform_(self.noisy_proj.weight)
        nn.init.zeros_(self.noisy_proj.bias)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # speaker-specific bias on noisy logits (optional)
        if speaker_embed_dim > 0:
            self.speaker_logit_bias = nn.Linear(speaker_embed_dim, mimi_vocab, bias=False)
        else:
            self.speaker_logit_bias = None

        # noisy logit predictor (operates in token_embed_dim space)
        self.predictor = NoisyLogitPredictor(
            embed_dim=token_embed_dim,
            lookahead=lookahead,
            n_heads=pred_n_heads,
            n_layers=pred_n_layers,
            dropout=dropout,
        )

    def _soft_embed(self, noisy_logits: torch.Tensor,
                    spk_emb: torch.Tensor = None) -> torch.Tensor:
        """noisy_logits [B, T, V]  →  soft_emb [B, T, E]"""
        if self.speaker_logit_bias is not None and spk_emb is not None:
            bias = self.speaker_logit_bias(spk_emb.to(noisy_logits.dtype))
            noisy_logits = noisy_logits + bias.unsqueeze(1)
        weights = F.softmax(noisy_logits.float() / self.temperature, dim=-1)
        return (weights.to(noisy_logits.dtype) @ self.noisy_embed.weight)   # [B, T, E]

    def _build_mask(self, T_q: int, T_kv: int,
                    device, dtype) -> torch.Tensor:
        """
        [T_q, T_kv + L*T_q] attention mask.

        Real K/V positions [0..T_kv-1]: causal — query t attends if pos ≤ t.
        Predicted K/V positions [T_kv + l*T_q + t]: diagonal per l —
          query t attends only to its own prediction slot for each l.
        """
        L = self.lookahead
        T_total = T_kv + L * T_q
        mask = torch.full((T_q, T_total), float("-inf"), device=device, dtype=dtype)

        # real part: lower-triangular
        q_idx  = torch.arange(T_q,  device=device).unsqueeze(1)   # [T_q, 1]
        kv_idx = torch.arange(T_kv, device=device).unsqueeze(0)   # [1,  T_kv]
        mask[:, :T_kv][kv_idx <= q_idx] = 0.0

        # predicted future part: block-diagonal (one slot per query per lookahead step)
        t_idx = torch.arange(T_q, device=device)
        for l in range(L):
            col = T_kv + l * T_q + t_idx   # [T_q]
            mask[t_idx, col] = 0.0

        return mask

    def forward(self, av_features: torch.Tensor,
                noisy_logits: torch.Tensor,
                spk_emb: torch.Tensor = None):
        """
        av_features  : [B, T_q, D]
        noisy_logits : [B, T_kv, V]   (T_kv == T_q after 12.5 Hz downsample)

        Returns
        -------
        fused     : [B, T_q, D]
        pred_loss : scalar (None at eval time)
        """
        B, T_q, D = av_features.shape
        T_kv = noisy_logits.size(1)
        L    = self.lookahead

        # ── soft embedding of real noisy logits ──────────────────────────────
        soft_emb = self._soft_embed(noisy_logits, spk_emb)   # [B, T_kv, E]

        # ── prediction loss (training only) ──────────────────────────────────
        pred_loss = self.predictor.prediction_loss(soft_emb) if self.training else None

        # ── predicted future soft embeddings ─────────────────────────────────
        # pred_future[b, t, l, e] = predicted soft_emb for position t+l+1
        #                           using causal context 0..t  (no future leakage)
        pred_future = self.predictor(soft_emb)   # [B, T_q, L, E]

        # ── project real K/V ─────────────────────────────────────────────────
        real_kv = self.noisy_proj(soft_emb)      # [B, T_kv, D]

        # ── project predicted K/V ────────────────────────────────────────────
        # Reshape to [B, L*T_q, E] with layout [l=0 block | l=1 block | ...]
        # pred_future: [B, T_q, L, E] → permute → [B, L, T_q, E] → [B, L*T_q, E]
        pred_flat = pred_future.permute(0, 2, 1, 3).reshape(B, L * T_q, self.embed_dim)
        pred_kv   = self.noisy_proj(pred_flat)   # [B, L*T_q, D]

        # ── full K/V bank ────────────────────────────────────────────────────
        full_kv = torch.cat([real_kv, pred_kv], dim=1)   # [B, T_kv + L*T_q, D]

        # ── attention mask ────────────────────────────────────────────────────
        attn_mask = self._build_mask(T_q, T_kv, av_features.device, av_features.dtype)

        attended, _ = self.cross_attn(
            query    = av_features,
            key      = full_kv,
            value    = full_kv,
            attn_mask = attn_mask,
        )
        fused = self.norm(av_features + attended)
        return fused, pred_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertNoisyPredCrossAttnConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING,
                          metadata={"help": "pretrained AV-HuBERT checkpoint"})
    no_pretrained_weights: bool = field(default=False)

    # dropouts
    dropout_input: float     = field(default=0.0)
    final_dropout: float     = field(default=0.0)
    dropout: float           = field(default=0.0)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)

    # masking (API compat)
    apply_mask: bool                  = field(default=False)
    mask_length: int                  = field(default=10)
    mask_prob: float                  = field(default=0.5)
    mask_selection: str               = field(default="static")
    mask_other: float                 = field(default=0.0)
    no_mask_overlap: bool             = field(default=False)
    mask_channel_length: int          = field(default=10)
    mask_channel_prob: float          = field(default=0.0)
    mask_channel_selection: str       = field(default="static")
    mask_channel_other: float         = field(default=0.0)
    no_mask_channel_overlap: bool     = field(default=False)

    freeze_finetune_updates: int = field(default=0)
    feature_grad_mult: float     = field(default=1.0)
    layerdrop: float             = field(default=0.0)
    lookahead_frames: int        = field(default=0)

    mimi_vocab_size: int = field(default=2048)
    head_hidden_dim: int = field(default=0)

    # Cross-attention fuser
    token_embed_dim: int      = field(default=256)
    cross_attn_heads: int     = field(default=4)
    cross_attn_dropout: float = field(default=0.1)
    logit_temperature: float  = field(default=1.0)

    # Noisy logit predictor
    pred_lookahead: int   = field(default=4,
                                  metadata={"help": "future frames to predict (L)"})
    pred_n_heads: int     = field(default=4)
    pred_n_layers: int    = field(default=2)
    pred_dropout: float   = field(default=0.1)
    lambda_pred: float    = field(default=1.0,
                                  metadata={"help": "weight on predictor cosine loss"})

    # Speaker conditioning
    speaker_cond: str      = field(default="none")
    speaker_embed_dim: int = field(default=256)
    pretrained_spk_dim: int = field(default=512)

    normalize: bool = field(default=False)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class NoisyPredCrossAttnEncoder(FairseqEncoder):

    def __init__(self, cfg: AVHubertNoisyPredCrossAttnConfig):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout":                  cfg.dropout,
            "activation_dropout":       cfg.activation_dropout,
            "dropout_input":            cfg.dropout_input,
            "attention_dropout":        cfg.attention_dropout,
            "mask_length":              cfg.mask_length,
            "mask_prob":                cfg.mask_prob,
            "mask_selection":           cfg.mask_selection,
            "mask_other":               cfg.mask_other,
            "no_mask_overlap":          cfg.no_mask_overlap,
            "mask_channel_length":      cfg.mask_channel_length,
            "mask_channel_prob":        cfg.mask_channel_prob,
            "mask_channel_selection":   cfg.mask_channel_selection,
            "mask_channel_other":       cfg.mask_channel_other,
            "no_mask_channel_overlap":  cfg.no_mask_channel_overlap,
            "encoder_layerdrop":        cfg.layerdrop,
            "feature_grad_mult":        cfg.feature_grad_mult,
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
            logger.info(f"[NoisyPredCrossAttn] missing={len(missing)}, unexpected={len(unexpected)}")

        causal_model.remove_pretraining_modules()
        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim

        self.w2v_model             = causal_model
        self.final_dropout         = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates           = 0
        self.lambda_pred           = cfg.lambda_pred

        # 25 Hz → 12.5 Hz causal downsample
        self.temporal_downsample = nn.Conv1d(d, d, kernel_size=2, stride=2, padding=0)

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

        # predictive noisy cross-attention fuser
        self.cross_attn_fuser = PredictiveNoisyCrossAttnFuser(
            d_model         = d,
            mimi_vocab      = cfg.mimi_vocab_size,
            token_embed_dim = cfg.token_embed_dim,
            lookahead       = cfg.pred_lookahead,
            n_heads         = cfg.cross_attn_heads,
            dropout         = cfg.cross_attn_dropout,
            temperature     = cfg.logit_temperature,
            pred_n_heads    = cfg.pred_n_heads,
            pred_n_layers   = cfg.pred_n_layers,
            speaker_embed_dim = spk_dim,
        )

        self.mimi_head = MimiHead(
            in_dim     = d,
            out_dim    = cfg.mimi_vocab_size,
            hidden_dim = cfg.head_hidden_dim,
            dropout    = cfg.final_dropout,
        )

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
            x = torch.cat([x, x[:, -1:].expand(x.size(0), target_len - cur, x.size(2))], dim=1)
        return x

    def _upsample_padding_mask(self, pm, target_len):
        if pm is None:
            return None
        pm  = pm.repeat_interleave(2, dim=1)
        cur = pm.size(1)
        if cur > target_len:
            pm = pm[:, :target_len]
        elif cur < target_len:
            pm = torch.cat([pm, pm[:, -1:].expand(pm.size(0), target_len - cur)], dim=1)
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

        # 25 Hz → 12.5 Hz
        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)   # [B, T/2, D]

        T_down       = x.size(1)
        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        pred_coding_loss = None
        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]
            x, pred_loss = self.cross_attn_fuser(x, noisy_logits_ds, spk_emb=spk_emb)
            if pred_loss is not None:
                pred_coding_loss = pred_loss * self.lambda_pred

        x           = self.final_dropout(x)
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
            "pred_coding_loss":     pred_coding_loss,
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

@register_model("av_hubert_noisy_pred_crossattn", dataclass=AVHubertNoisyPredCrossAttnConfig)
class AVHubertNoisyPredCrossAttnModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertNoisyPredCrossAttnConfig, task: FairseqTask):
        return cls(NoisyPredCrossAttnEncoder(cfg))

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
