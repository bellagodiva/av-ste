"""
AV Sync Score Predictor
=======================

Architecture: MulT-style bidirectional cross-attention.

                 ┌─────────────────┐
clean tokens     │  SpeechEncoder  │ → speech features [B, T, C]
[B, T] int  ──▶ │  embed + xfmr   │          │
                 └─────────────────┘          ▼  cross-attention (both ways)
                                       ┌──────────────┐
                 ┌─────────────────┐   │  speech_xattn│ → speech_ctx [B, T, C]
visual feats     │  VisualEncoder  │──▶│  visual_xattn│ → visual_ctx [B, T, C]
[B, T, D]   ──▶ │  proj + xfmr    │   └──────────────┘
                 └─────────────────┘          │
                                              ▼
                              frame_sync = cosine_sim(avgpool_W(proj(s)), avgpool_W(proj(v)))  [B, T]
                              speech_emb = mean_pool(proj(s))             [B, C]
                              visual_emb = mean_pool(proj(v))             [B, C]

Pre-training losses
───────────────────
  L_nce    : symmetric InfoNCE on (speech_emb, visual_emb) across batch
             positives = same utterance, negatives = other speakers in batch

  L_margin : max-margin frame-level loss with temporal-shift negatives
             frame_sync(aligned) - frame_sync(shifted by k) > margin
             teaches the model WHERE sync breaks, not just WHETHER

  L = L_nce + λ_frame · L_margin

Using as frozen loss in enhancement training
────────────────────────────────────────────
  1. Load predictor from checkpoint, call predictor.freeze()
  2. Compute soft speech embeddings from the enhancement model's 12.5 Hz logits:
       soft_emb = softmax(logits_12p5 / T) @ predictor.speech_enc.token_embed.weight
  3. Forward: s_ctx, v_ctx = predictor.encode(soft_emb, visual_feats, is_tokens=False)
  4. frame_sync = predictor.frame_sync_scores(s_ctx, v_ctx)      [B, T]
  5. L_sync = 1.0 - frame_sync[valid_mask].mean()                 scalar

  The token_embed table is shared between pre-training (hard) and fine-tuning (soft),
  so no bridging layer is needed — the embedding space is identical.

Required env vars
─────────────────
  Pre-training:
    VISUAL_FEATS_ROOT   — dir of {utt_id}.npy  [T, D] visual-only features
                          Extract with: process_dataset/extract_visual_features.py
    CLEAN_TOKENS_ROOT   — dir of {utt_id}.npy  [T] int32 clean Mimi token IDs
                          (or pass label_dir from the main task config)

  Enhancement training:
    VISUAL_FEATS_ROOT   — same directory (visual features are reused)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AVSyncConfig:
    mimi_vocab_size: int   = 2048   # Mimi codebook size
    visual_feat_dim: int   = 1024   # AV-HuBERT hidden dim (visual-only run)
    sync_proj_dim:   int   = 256    # internal dim C for both streams
    sync_attn_heads: int   = 4
    sync_enc_layers: int   = 2      # transformer depth per stream
    sync_xattn_layers: int = 1      # cross-attention depth
    sync_dropout:    float = 0.1
    nce_temperature: float = 0.07   # InfoNCE softmax temperature


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    One cross-attention layer: query from stream A, key/value from stream B.
    Includes residual + feedforward (pre-LN style for stability).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                key_padding_mask=None) -> torch.Tensor:
        """
        query     : [B, T_q, C]
        key_value : [B, T_kv, C]
        returns   : [B, T_q, C]
        """
        q  = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        query = query + attended
        query = query + self.ff(self.norm_ff(query))
        return query


class SpeechSyncEncoder(nn.Module):
    """
    Encodes either discrete token IDs (pre-training) or continuous soft
    embeddings (enhancement-training bridge via softmax(logits) @ embed.weight).

    The token_embed table is the shared bridge — hard lookup during pre-training,
    soft weighted average during enhancement training.
    """

    def __init__(self, vocab_size: int, proj_dim: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, proj_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[B, T] int → [B, T, proj_dim]  (hard lookup)."""
        return self.token_embed(token_ids)

    def embed_soft(self, logits: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
        """
        [B, T, vocab_size] logits → [B, T, proj_dim]  (soft codebook lookup).
        Used during enhancement training to bridge continuous model output
        into the same embedding space the sync predictor was trained on.
        """
        weights = F.softmax(logits.float() / temperature, dim=-1).to(
            self.token_embed.weight.dtype
        )
        return weights @ self.token_embed.weight   # [B, T, proj_dim]

    def forward(self, x: torch.Tensor, is_tokens: bool = True,
                src_key_padding_mask=None) -> torch.Tensor:
        """
        x : [B, T] int  (is_tokens=True)
          | [B, T, proj_dim] float  (is_tokens=False, already embedded)
        returns: [B, T, proj_dim]
        """
        if is_tokens:
            x = self.token_embed(x)
        # else: x is already [B, T, proj_dim] — soft embed or external proj
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class VisualSyncEncoder(nn.Module):
    """
    Encodes pre-extracted AV-HuBERT visual-only features.
    These are extracted by running AV-HuBERT with audio zeroed out.
    """

    def __init__(self, visual_dim: int, proj_dim: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(visual_dim, proj_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask=None) -> torch.Tensor:
        """[B, T, visual_dim] → [B, T, proj_dim]"""
        x = self.proj(x)
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class AVSyncPredictor(nn.Module):
    """
    Bidirectional cross-attention AV sync predictor.

    Outputs
    ───────
    frame_sync : [B, T]  cosine similarity per frame in [-1, 1]
                 high  → speech tokens match lip movements at this frame
                 low   → mismatch (interferer content or wrong speaker)

    speech_emb : [B, C]  mean-pooled, L2-normed — for InfoNCE pre-training
    visual_emb : [B, C]  mean-pooled, L2-normed — for InfoNCE pre-training
    """

    def __init__(self, cfg: AVSyncConfig):
        super().__init__()
        C = cfg.sync_proj_dim

        self.speech_enc = SpeechSyncEncoder(
            vocab_size=cfg.mimi_vocab_size,
            proj_dim=C,
            n_heads=cfg.sync_attn_heads,
            n_layers=cfg.sync_enc_layers,
            dropout=cfg.sync_dropout,
        )
        self.visual_enc = VisualSyncEncoder(
            visual_dim=cfg.visual_feat_dim,
            proj_dim=C,
            n_heads=cfg.sync_attn_heads,
            n_layers=cfg.sync_enc_layers,
            dropout=cfg.sync_dropout,
        )

        # Bidirectional cross-attention: each stream queries the other
        self.speech_xattn_layers = nn.ModuleList([
            CrossAttentionBlock(C, cfg.sync_attn_heads, cfg.sync_dropout)
            for _ in range(cfg.sync_xattn_layers)
        ])
        self.visual_xattn_layers = nn.ModuleList([
            CrossAttentionBlock(C, cfg.sync_attn_heads, cfg.sync_dropout)
            for _ in range(cfg.sync_xattn_layers)
        ])

        # Final projection to contrastive / scoring space
        self.speech_proj = nn.Linear(C, C)
        self.visual_proj = nn.Linear(C, C)
        nn.init.xavier_uniform_(self.speech_proj.weight)
        nn.init.zeros_(self.speech_proj.bias)
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.zeros_(self.visual_proj.bias)

    def encode(self, speech_input: torch.Tensor, visual_feats: torch.Tensor,
               is_tokens: bool = True,
               padding_mask=None) -> tuple:
        """
        speech_input  : [B, T] int        (is_tokens=True,  hard lookup)
                      | [B, T, C] float   (is_tokens=False, already embedded)
        visual_feats  : [B, T, D_vis]
        padding_mask  : [B, T] bool        True = padded frame (invalid)

        Returns (speech_ctx, visual_ctx), each [B, T, C].
        """
        s = self.speech_enc(speech_input, is_tokens=is_tokens,
                            src_key_padding_mask=padding_mask)
        v = self.visual_enc(visual_feats,
                            src_key_padding_mask=padding_mask)

        # stacked bidirectional cross-attention
        for s_xattn, v_xattn in zip(self.speech_xattn_layers,
                                     self.visual_xattn_layers):
            s_new = s_xattn(query=s, key_value=v)   # speech queries visual
            v_new = v_xattn(query=v, key_value=s)   # visual queries speech
            s, v = s_new, v_new

        return s, v

    def frame_sync_scores(self, speech_ctx: torch.Tensor,
                          visual_ctx: torch.Tensor,
                          window: int = 4) -> torch.Tensor:
        """
        Chunk-averaged cosine similarity between cross-attended streams.

        Each score covers a local window of `window` frames (default 4 = 320ms
        at 12.5Hz), enough to observe a full consonant-vowel lip transition.
        Frames near the boundary use whatever context is available (same-pad).

        Returns [B, T] in [-1, 1].  Higher = in sync.
        """
        s = self.speech_proj(speech_ctx)   # [B, T, C]
        v = self.visual_proj(visual_ctx)   # [B, T, C]

        # Local average over a window of W frames (same-padding on both sides)
        if window > 1:
            pad = window // 2
            # [B, T, C] → [B, C, T] for F.avg_pool1d
            s = F.avg_pool1d(s.transpose(1, 2), window, stride=1,
                             padding=pad).transpose(1, 2)[:, :speech_ctx.size(1), :]
            v = F.avg_pool1d(v.transpose(1, 2), window, stride=1,
                             padding=pad).transpose(1, 2)[:, :visual_ctx.size(1), :]

        s = F.normalize(s, dim=-1)
        v = F.normalize(v, dim=-1)
        return (s * v).sum(dim=-1)   # [B, T]

    def pool_embeddings(self, speech_ctx: torch.Tensor,
                        visual_ctx: torch.Tensor,
                        padding_mask=None) -> tuple:
        """Mean-pool over valid frames → L2-normed [B, C] embeddings for InfoNCE."""
        sp = self.speech_proj(speech_ctx)
        vp = self.visual_proj(visual_ctx)
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            denom = valid.sum(1).clamp(min=1)
            s_emb = (sp * valid).sum(1) / denom
            v_emb = (vp * valid).sum(1) / denom
        else:
            s_emb = sp.mean(1)
            v_emb = vp.mean(1)
        return F.normalize(s_emb, dim=-1), F.normalize(v_emb, dim=-1)

    def forward(self, speech_input: torch.Tensor, visual_feats: torch.Tensor,
                is_tokens: bool = True, padding_mask=None) -> dict:
        """Full forward pass. Returns frame_sync, speech_emb, visual_emb."""
        s_ctx, v_ctx = self.encode(speech_input, visual_feats,
                                   is_tokens=is_tokens,
                                   padding_mask=padding_mask)
        s_emb, v_emb = self.pool_embeddings(s_ctx, v_ctx, padding_mask)
        return {
            "frame_sync": self.frame_sync_scores(s_ctx, v_ctx),  # [B, T]
            "speech_emb": s_emb,                                  # [B, C]
            "visual_emb": v_emb,                                  # [B, C]
            "speech_ctx": s_ctx,
            "visual_ctx": v_ctx,
        }

    def freeze(self):
        """Freeze all parameters. Call before using as a training loss."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str, cfg: AVSyncConfig = None,
                        map_location="cpu") -> "AVSyncPredictor":
        import sys as _sys
        _this = _sys.modules.get(__name__)
        if _this is not None and "av_sync_predictor" not in _sys.modules:
            _sys.modules["av_sync_predictor"] = _this
        ckpt = torch.load(path, map_location=map_location)
        if cfg is None:
            cfg = ckpt["cfg"]
        model = cls(cfg)
        model.load_state_dict(ckpt["model"])
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training losses
# ─────────────────────────────────────────────────────────────────────────────

def _pool_valid(frame_sync: torch.Tensor, valid_mask=None) -> torch.Tensor:
    """Mean-pool frame_sync over valid frames → [B]."""
    if valid_mask is not None:
        valid_f = valid_mask.float()
        return (frame_sync * valid_f).sum(1) / valid_f.sum(1).clamp(min=1)
    return frame_sync.mean(1)


def compute_bce_loss(outputs_pos: dict, outputs_neg: dict,
                     valid_mask=None) -> torch.Tensor:
    """
    SyncNet-style BCE loss.
    Positive (aligned) → label 1, negative (time-shifted) → label 0.
    Cosine similarity scores are mean-pooled per utterance then used as logits.
    """
    B = outputs_pos["frame_sync"].size(0)
    scores = torch.cat([_pool_valid(outputs_pos["frame_sync"], valid_mask),
                        _pool_valid(outputs_neg["frame_sync"], valid_mask)])
    labels = torch.cat([torch.ones(B,  device=scores.device),
                        torch.zeros(B, device=scores.device)])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_nce_loss(speech_emb: torch.Tensor, visual_emb: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE (utterance-level).
    Positives: diagonal (speech_i, visual_i) — same utterance.
    Negatives: off-diagonal — different utterances / speakers in the batch.
    """
    B = speech_emb.size(0)
    sim = torch.matmul(speech_emb, visual_emb.T) / temperature   # [B, B]
    labels = torch.arange(B, device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def compute_sync_pretraining_loss(outputs_pos: dict, outputs_neg: dict,
                                   cfg: "AVSyncConfig",
                                   valid_mask=None) -> dict:
    """
    Combined pre-training loss: BCE (temporal sync) + InfoNCE (speaker swap).

    L = L_bce + L_nce

    L_bce  teaches: in-sync pairs score higher than time-shifted pairs.
    L_nce  teaches: speech and visual embeddings from the same utterance
                    should be closer than embeddings from different speakers.

    outputs_pos : forward() on aligned (speech, visual) pairs
    outputs_neg : forward() on temporally-shifted speech with same visual
    """
    loss_bce = compute_bce_loss(outputs_pos, outputs_neg, valid_mask)
    loss_nce = compute_nce_loss(
        outputs_pos["speech_emb"], outputs_pos["visual_emb"],
        temperature=cfg.nce_temperature,
    )
    return {
        "loss":     loss_bce + loss_nce,
        "loss_bce": loss_bce.item(),
        "loss_nce": loss_nce.item(),
    }
