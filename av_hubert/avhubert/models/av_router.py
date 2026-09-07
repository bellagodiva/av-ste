# Audio-Visual Agreement Router
#
# At each 12.5 Hz token step, computes the cosine similarity between the
# audio and video SubModel representations (pre-fusion, post-projection).
# Low similarity → audio is corrupted → use AV-HuBERT predicted semantic token.
# High similarity → audio is clean → use Mimi codebook-0 token.
#
# The router is self-supervised: no routing labels are needed.
# It can run in hard mode (discrete binary decision) or soft mode
# (differentiable weighted blend, useful for analysis).
#
# Usage (inference):
#   router = AVAgreementRouter(threshold=0.0, temperature=0.1)
#   semantic_token = router(
#       avhubert_logits,   # [B, T, V] at 12.5 Hz
#       mimi_cb0_token,    # [B, T] int token ids at 12.5 Hz
#       feat_audio,        # [B, D, T] pre-fusion audio features
#       feat_video,        # [B, D, T] pre-fusion video features
#   )

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RouterConfig:
    # Cosine similarity threshold:
    #   agreement > threshold  → trust Mimi (clean audio)
    #   agreement <= threshold → trust AV-HuBERT (corrupted audio)
    # A good starting value is 0.0 (random agreement baseline).
    # Tune using the histogram analysis on your validation set.
    threshold: float = 0.0

    # Temperature for soft routing sigmoid: lower = harder decision
    temperature: float = 0.1

    # If True: hard binary routing (inference mode)
    # If False: soft differentiable blend (analysis / training)
    hard: bool = True


class AVAgreementRouter(nn.Module):
    """
    Self-supervised audio-visual agreement router.

    Inputs pre-fusion per-modality features from CausalAVHubertModel
    (after SubModel projection to encoder_embed_dim, before concat/add fusion).

    These are at 25 Hz (same as backbone output). We downsample to 12.5 Hz
    by averaging pairs to match the token rate before computing agreement.
    """

    def __init__(self, cfg: RouterConfig = RouterConfig()):
        super().__init__()
        self.threshold = cfg.threshold
        self.temperature = cfg.temperature
        self.hard = cfg.hard

    def agreement_score(
        self,
        feat_audio: torch.Tensor,  # [B, D, T] at 25 Hz
        feat_video: torch.Tensor,  # [B, D, T] at 25 Hz
    ) -> torch.Tensor:
        """
        Returns cosine similarity per token at 12.5 Hz: [B, T/2]

        Downsamples 25 Hz → 12.5 Hz by averaging pairs before comparing,
        matching the temporal resolution of Mimi tokens.
        """
        # Ensure even length
        T = feat_audio.size(2)
        if T % 2 == 1:
            feat_audio = feat_audio[:, :, :-1]
            feat_video = feat_video[:, :, :-1]
            T = T - 1

        # Average pairs: [B, D, T] → [B, D, T/2]
        fa = feat_audio.reshape(feat_audio.size(0), feat_audio.size(1), T // 2, 2).mean(-1)
        fv = feat_video.reshape(feat_video.size(0), feat_video.size(1), T // 2, 2).mean(-1)

        # [B, D, T/2] → [B, T/2, D]
        fa = fa.transpose(1, 2)
        fv = fv.transpose(1, 2)

        # Cosine similarity per frame: [B, T/2]
        agreement = F.cosine_similarity(fa, fv, dim=-1)
        return agreement

    def routing_weight(self, agreement: torch.Tensor) -> torch.Tensor:
        """
        agreement: [B, T/2], cosine similarity in [-1, 1]

        Returns alpha: [B, T/2] in [0, 1]
            alpha ~ 1 → use AV-HuBERT (corrupted audio, low agreement)
            alpha ~ 0 → use Mimi (clean audio, high agreement)

        Hard mode: binary threshold
        Soft mode: sigmoid((threshold - agreement) / temperature)
            → 1 when agreement << threshold
            → 0 when agreement >> threshold
        """
        if self.hard:
            return (agreement <= self.threshold).float()
        else:
            return torch.sigmoid((self.threshold - agreement) / self.temperature)

    def forward(
        self,
        avhubert_logits: torch.Tensor,  # [B, T/2, V] at 12.5 Hz
        mimi_cb0_tokens: torch.Tensor,  # [B, T/2] int token ids at 12.5 Hz
        feat_audio: torch.Tensor,       # [B, D, T] at 25 Hz (pre-fusion)
        feat_video: torch.Tensor,       # [B, D, T] at 25 Hz (pre-fusion)
    ) -> torch.Tensor:
        """
        Returns selected semantic token ids: [B, T/2]

        Hard mode:
            alpha=1 frames → argmax(avhubert_logits)
            alpha=0 frames → mimi_cb0_tokens

        Soft mode (analysis only, not for actual token selection):
            Returns soft-blended token probabilities — use for training
            the router's threshold/temperature via cross-entropy.
        """
        agreement = self.agreement_score(feat_audio, feat_video)  # [B, T/2]
        alpha = self.routing_weight(agreement)                     # [B, T/2]

        T_half = avhubert_logits.size(1)
        B = avhubert_logits.size(0)

        # Align lengths (agreement may be shorter by 1 if T was odd)
        T_agree = alpha.size(1)
        if T_agree < T_half:
            avhubert_logits = avhubert_logits[:, :T_agree, :]
            mimi_cb0_tokens = mimi_cb0_tokens[:, :T_agree]
            T_half = T_agree

        if self.hard:
            avhubert_tokens = avhubert_logits.argmax(dim=-1)  # [B, T/2]
            # alpha=1 → avhubert, alpha=0 → mimi
            selected = torch.where(alpha.bool(), avhubert_tokens, mimi_cb0_tokens)
            return selected   # [B, T/2]
        else:
            # Soft blend of log-probs for analysis
            avhubert_probs = F.softmax(avhubert_logits, dim=-1)        # [B, T/2, V]
            V = avhubert_probs.size(-1)

            mimi_onehot = F.one_hot(mimi_cb0_tokens.clamp(min=0), num_classes=V).float()  # [B, T/2, V]

            alpha_3d = alpha.unsqueeze(-1)  # [B, T/2, 1]
            blended = alpha_3d * avhubert_probs + (1 - alpha_3d) * mimi_onehot  # [B, T/2, V]
            return blended   # [B, T/2, V] — soft distribution


# ---------------------------------------------------------------------------
# Convenience: agreement score only (for analysis / histogram plotting)
# ---------------------------------------------------------------------------

def compute_agreement_scores(
    feat_audio: torch.Tensor,  # [B, D, T]
    feat_video: torch.Tensor,  # [B, D, T]
) -> torch.Tensor:
    """Returns [B, T/2] cosine similarity scores. Use for threshold calibration."""
    router = AVAgreementRouter()
    return router.agreement_score(feat_audio, feat_video)


# ---------------------------------------------------------------------------
# Modified extract_finetune that also returns pre-fusion modality features
# Monkey-patch onto CausalAVHubertModel at import time.
# ---------------------------------------------------------------------------

def _extract_finetune_with_modality_features(
    self, source, padding_mask=None, mask=False, output_layer=None
):
    """
    Same as extract_finetune but also returns pre-fusion per-modality features
    for the agreement router.

    Returns:
        x            : [B, T, D]   fused encoder output
        padding_mask : [B, T]
        feat_audio   : [B, D, T]   audio SubModel output at 25 Hz
        feat_video   : [B, D, T]   video SubModel output at 25 Hz
    """
    src_audio = source['audio']
    src_video = source['video']

    features_video = self.forward_features(src_video, modality='video')  # [B, D, T]
    features_audio = self.forward_features(src_audio, modality='audio')  # [B, D, T]

    feat_audio = features_audio.detach()  # detach: router is not trained
    feat_video = features_video.detach()

    if self.modality_fuse == 'concat':
        features = torch.cat([features_audio, features_video], dim=1)
    elif self.modality_fuse == 'add':
        features = features_audio + features_video

    features = features.transpose(1, 2)
    features = self.layer_norm(features)

    if padding_mask is not None:
        padding_mask = self.forward_padding_mask(features, padding_mask)

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    features = self.dropout_input(features)
    x, _ = self.encoder(
        features,
        padding_mask=padding_mask,
        layer=None if output_layer is None else output_layer - 1,
    )
    return x, padding_mask, feat_audio, feat_video


def patch_causal_avhubert(causal_avhubert_cls):
    """
    Call this after importing CausalAVHubertModel to attach
    extract_finetune_with_modality_features as an instance method.

    Usage:
        from avhubert.models.av_router import patch_causal_avhubert
        from avhubert.avhubert_causal import CausalAVHubertModel
        patch_causal_avhubert(CausalAVHubertModel)
    """
    import types
    causal_avhubert_cls.extract_finetune_with_modality_features = (
        _extract_finetune_with_modality_features
    )
