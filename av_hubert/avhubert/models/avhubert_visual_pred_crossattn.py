# Visual-Predictive Cross-Attention AV-HuBERT  (causal=0, streaming-safe)
#
# WHY this works:
#   4-lookahead frames help because the AV-HuBERT transformer can attend to
#   future CLEAN VIDEO features (lip movements) inside its self-attention.
#   Future visual context disambiguates the current phoneme — e.g. lips closing
#   next frame strongly predicts a bilabial, helping resolve noisy audio.
#
#   The JEPA approach failed because it predicted future AV-HuBERT *output*
#   features (h_t), which are already contaminated by noisy audio — both in
#   their inputs and in what the transformer attended to.
#
#   This model predicts future VISUAL-ONLY features (ResNet output, before any
#   audio fusion or transformer processing).  These are:
#     - completely clean — zero noise contamination
#     - the exact signal 4-lookahead provides extra access to
#     - smooth and physically predictable from past lip dynamics
#
# Architecture:
#   1. CausalAVHuBERT backbone (lookahead=0) → h_t [B,T,D]
#      extract_finetune(..., return_visual=True) also returns
#      visual_feats [B,T,D]  — ResNet output, before fusion/transformer
#   2. VisualPredictor (causal transformer in D-dim visual space):
#        visual_feats[0..t]  →  predicted visual_feats[t+1..t+L]  [B,T,L,D]
#      Loss: cosine_sim_loss(predicted, real_future_visual.stop_grad)
#   3. Augment h_t with predicted future visual context:
#        aug_h = linear([h_t ; mean(predicted_visual[t+1..t+L])])  → [B,T,D]
#   4. Soft cross-attention fuser:
#        Q = aug_h,  K/V = soft noisy logit embeddings  (causal, 0-lookahead)
#   5. Mimi head → clean token logits
#
# At streaming inference: VisualPredictor runs causally on frames 0..t,
# produces predicted future visual context — no leakage.
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
from .avhubert_crossattn_soft import SoftCrossAttentionFuser, PretrainedSpeakerEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visual Feature Predictor
# ---------------------------------------------------------------------------

class VisualPredictor(nn.Module):
    """
    Predicts future clean visual features from past visual context.

    Input  : visual_feats [B, T, D]  — ResNet output (clean, pre-fusion, pre-transformer)
    Output : [B, T, L, D]             — predicted visual_feats for t+1..t+L at each t

    Uses a small causal transformer in D-dim space.  The predictor operates
    entirely on the clean visual stream so the prediction target is noise-free.

    Training loss: cosine-similarity vs. real future visual feats (stop-gradient).
    The model learns lip-movement dynamics: "given how the lips moved up to t,
    predict where they will be at t+1..t+L."

    Streaming: at time t the predictor has seen frames 0..t (causal mask) and
    outputs predictions for t+1..t+L — strictly causal, zero future leakage.
    """

    def __init__(self, d_model: int, lookahead: int,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lookahead = lookahead
        self.d_model   = d_model

        # input projection: D → pred_dim (separate from backbone weights)
        pred_dim = d_model // 2   # keep predictor lightweight
        self.in_proj = nn.Linear(d_model, pred_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=pred_dim, nhead=n_heads,
            dim_feedforward=pred_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # output head: pred_dim → L * D  (one D-dim prediction per future step)
        self.out_head = nn.Linear(pred_dim, lookahead * d_model)
        nn.init.xavier_uniform_(self.in_proj.weight);  nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_head.weight); nn.init.zeros_(self.out_head.bias)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        visual_feats : [B, T, D]
        Returns      : [B, T, L, D]
                       result[:, t, l, :] = predicted visual_feat for t+l+1,
                       using causal context 0..t.
        """
        B, T, D = visual_feats.shape
        mask = self._causal_mask(T, visual_feats.device)
        z    = self.in_proj(visual_feats)              # [B, T, pred_dim]
        ctx  = self.transformer(z, mask=mask)          # [B, T, pred_dim]
        pred = self.out_head(ctx)                      # [B, T, L*D]
        return pred.view(B, T, self.lookahead, D)

    def prediction_loss(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        Cosine-similarity loss between predicted and real future visual feats.
        Targets are stop-gradient so only the predictor (not the backbone) is
        pushed — we don't want to collapse the visual features.

        For l = 1..L:
          pred[:, :T-l, l-1, :]  vs.  visual_feats[:, l:, :].detach()
        """
        B, T, D = visual_feats.shape
        L = self.lookahead
        if T <= L:
            return visual_feats.new_zeros(1).squeeze()

        pred  = self.forward(visual_feats)   # [B, T, L, D]
        loss  = visual_feats.new_zeros(1).squeeze()
        count = 0
        for l in range(1, L + 1):
            T_valid  = T - l
            if T_valid <= 0:
                continue
            pred_l   = pred[:, :T_valid, l - 1, :]               # [B, T_valid, D]
            target_l = visual_feats[:, l:l + T_valid, :].detach()  # [B, T_valid, D]
            cos      = F.cosine_similarity(pred_l, target_l, dim=-1)
            loss     = loss + (1.0 - cos).mean()
            count   += 1

        return loss / max(count, 1)

    def get_future_context(self, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean predicted future visual feature at each position t:
          mean over l=1..L of predicted visual_feat[t+l]  →  [B, T, D]

        This is used to augment h_t: the model at position t gets a summary
        of where the lips are predicted to move next, enabling it to use that
        visual future context even though the backbone is causal.
        """
        pred = self.forward(visual_feats)        # [B, T, L, D]
        return pred.mean(dim=2)                  # [B, T, D]  — mean over lookahead steps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AVHubertVisualPredCrossAttnConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING,
                          metadata={"help": "pretrained AV-HuBERT checkpoint"})
    no_pretrained_weights: bool = field(default=False)

    # dropouts
    dropout_input: float      = field(default=0.0)
    final_dropout: float      = field(default=0.0)
    dropout: float            = field(default=0.0)
    attention_dropout: float  = field(default=0.0)
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

    # Cross-attention fuser (same as crossattn_soft)
    token_embed_dim: int      = field(default=256)
    cross_attn_heads: int     = field(default=4)
    cross_attn_dropout: float = field(default=0.1)
    logit_temperature: float  = field(default=1.0)

    # Visual predictor
    pred_lookahead: int   = field(default=4,
                                  metadata={"help": "future visual frames to predict (L)"})
    pred_n_heads: int     = field(default=4)
    pred_n_layers: int    = field(default=2)
    pred_dropout: float   = field(default=0.1)
    lambda_pred: float    = field(default=1.0,
                                  metadata={"help": "weight on visual predictor cosine loss"})

    # Speaker conditioning
    speaker_cond: str       = field(default="none")
    speaker_embed_dim: int  = field(default=256)
    pretrained_spk_dim: int = field(default=512)
    prefix_length: int      = field(default=0)

    normalize: bool = field(default=False)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VisualPredCrossAttnEncoder(FairseqEncoder):

    def __init__(self, cfg: AVHubertVisualPredCrossAttnConfig):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout":                 cfg.dropout,
            "activation_dropout":      cfg.activation_dropout,
            "dropout_input":           cfg.dropout_input,
            "attention_dropout":       cfg.attention_dropout,
            "mask_length":             cfg.mask_length,
            "mask_prob":               cfg.mask_prob,
            "mask_selection":          cfg.mask_selection,
            "mask_other":              cfg.mask_other,
            "no_mask_overlap":         cfg.no_mask_overlap,
            "mask_channel_length":     cfg.mask_channel_length,
            "mask_channel_prob":       cfg.mask_channel_prob,
            "mask_channel_selection":  cfg.mask_channel_selection,
            "mask_channel_other":      cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop":       cfg.layerdrop,
            "feature_grad_mult":       cfg.feature_grad_mult,
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
            logger.info(f"[VisualPredCrossAttn] missing={len(missing)}, unexpected={len(unexpected)}")

        causal_model.remove_pretraining_modules()
        super().__init__(task_pretrain.source_dictionary)

        d = causal_model.encoder.embedding_dim   # 1024 for large

        self.w2v_model             = causal_model
        self.final_dropout         = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates           = 0
        self.lambda_pred           = cfg.lambda_pred

        # 25 Hz → 12.5 Hz causal downsample
        self.temporal_downsample = nn.Conv1d(d, d, kernel_size=2, stride=2, padding=0)

        # Visual predictor — operates on ResNet features [B, T, D]
        # pred_lookahead=4 matches the 4-lookahead experiment
        self.visual_predictor = VisualPredictor(
            d_model   = d,
            lookahead = cfg.pred_lookahead,
            n_heads   = cfg.pred_n_heads,
            n_layers  = cfg.pred_n_layers,
            dropout   = cfg.pred_dropout,
        )

        # Merge h_t + mean predicted future visual  → D
        # Linear([h_t ; pred_visual]) → D
        self.future_merge = nn.Linear(d * 2, d)
        nn.init.xavier_uniform_(self.future_merge.weight)
        nn.init.zeros_(self.future_merge.bias)

        # Speaker conditioning
        self.speaker_cond = cfg.speaker_cond
        spk_dim = cfg.speaker_embed_dim if cfg.speaker_cond != "none" else 0
        if cfg.speaker_cond == "pretrained_spk":
            self.speaker_encoder = PretrainedSpeakerEncoder(
                pretrained_spk_dim=cfg.pretrained_spk_dim,
                speaker_embed_dim=cfg.speaker_embed_dim,
            )
        else:
            self.speaker_encoder = None

        # Standard soft cross-attention fuser (unchanged from crossattn_soft)
        self.cross_attn_fuser = SoftCrossAttentionFuser(
            d_model         = d,
            mimi_vocab      = cfg.mimi_vocab_size,
            token_embed_dim = cfg.token_embed_dim,
            n_heads         = cfg.cross_attn_heads,
            dropout         = cfg.cross_attn_dropout,
            lookahead       = 0,   # still causal — no future noisy logits
            temperature     = cfg.logit_temperature,
            speaker_embed_dim = spk_dim,
            prefix_length   = cfg.prefix_length,
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
            # return_visual=True gives us the clean ResNet features before
            # any audio fusion or transformer processing — the prediction target
            x, padding_mask, visual_feats = self.w2v_model.extract_finetune(
                source=source,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training,
                return_visual=True,
            )

        # visual_feats: [B, T_25hz, D] — clean, pure visual stream
        orig_len = x.size(1)

        spk_emb = None
        if self.speaker_cond == "pretrained_spk":
            speaker_embed = kwargs.get("speaker_embed", None)
            if speaker_embed is not None:
                spk_emb = self.speaker_encoder(speaker_embed.to(x.dtype))

        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]
            visual_feats = visual_feats[:, :-1, :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-1]
            if noisy_logits is not None:
                noisy_logits = noisy_logits[:, :-1, :]

        # ── downsample 25 Hz → 12.5 Hz ───────────────────────────────────────
        x = x.transpose(1, 2)
        x = F.pad(x, (1, 0))
        x = self.temporal_downsample(x)
        x = x.transpose(1, 2)   # [B, T/2, D]
        T_down = x.size(1)

        # visual_feats: take every other frame (same causal stride-2 logic)
        visual_feats_ds = visual_feats[:, 1::2, :][:, :T_down, :]   # [B, T/2, D]

        padding_mask = self._downsample_padding_mask(padding_mask, T_down)

        # ── visual prediction loss (training) & future context ───────────────
        pred_coding_loss = None
        if self.training:
            vis_loss = self.visual_predictor.prediction_loss(visual_feats_ds)
            pred_coding_loss = vis_loss * self.lambda_pred

        # future_ctx: [B, T/2, D] — mean predicted future visual features at each t
        # These are predicted from causal context (0..t), no leakage.
        future_ctx = self.visual_predictor.get_future_context(visual_feats_ds)

        # ── augment h_t with predicted future visual context ─────────────────
        # Concatenate h_t with predicted future visual features, project back to D.
        # This injects "where will the lips be in the next L frames" into each
        # position before the cross-attention retrieval step.
        aug_x = self.future_merge(
            torch.cat([x, future_ctx.to(x.dtype)], dim=-1)
        )   # [B, T/2, D]

        # ── soft cross-attention fuser (noisy logits as K/V) ─────────────────
        if noisy_logits is not None:
            noisy_logits_ds = noisy_logits[:, 1::2, :][:, :T_down, :]
            aug_x = self.cross_attn_fuser(aug_x, noisy_logits_ds, spk_emb=spk_emb)

        aug_x       = self.final_dropout(aug_x)
        logits_12p5 = self.mimi_head(aug_x)

        logits = self._upsample_by_repeat(logits_12p5, target_len=orig_len)
        out_pm = self._upsample_padding_mask(padding_mask, target_len=orig_len)

        if tbc:
            logits = logits.transpose(0, 1)

        return {
            "encoder_out":          logits,
            "encoder_padding_mask": out_pm,
            "padding_mask":         out_pm,
            "features":             aug_x,
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

@register_model("av_hubert_visual_pred_crossattn", dataclass=AVHubertVisualPredCrossAttnConfig)
class AVHubertVisualPredCrossAttnModel(BaseFairseqModel):

    @classmethod
    def build_model(cls, cfg: AVHubertVisualPredCrossAttnConfig, task: FairseqTask):
        return cls(VisualPredCrossAttnEncoder(cfg))

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
