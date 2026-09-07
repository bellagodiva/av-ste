#!/usr/bin/env python
"""
AV-STE inference on a single audio + lip-ROI video clip.

Input
-----
  --audio       16 kHz WAV of the noisy target speaker
  --video       96×96 px grayscale mouth-ROI MP4 at 25 fps
                (use scripts/prepare_lip_roi.py to extract from a face video)
  --checkpoint  avste.pt checkpoint
  --utt_id      clip identifier written into the JSON output (default: audio stem)

Output
------
  --output  path to save the result. Format is chosen by file extension:
            .pt   → LongTensor of shape [T] at 25 Hz, enhanced tokens only
                    (raw, for custom pipelines)
            .json → [{"utt_id": ..., "pred_tokens": [...]}]  (for compare_demo.py)
            .wav  → reconstructed speech: the enhanced semantic tokens
                    combined with the noisy input's own acoustic codebooks,
                    decoded through Mimi. This is a denoised-*sounding*
                    reconstruction of your input, not the clean original --
                    see the paper's WER discussion for why this is a
                    secondary/diagnostic output, not the primary one.

Examples
--------
  # Save as .pt (raw tokens)
  python scripts/infer_avste.py \\
      --audio      examples/sample_noisy.wav \\
      --video      examples/sample_lip.mp4 \\
      --checkpoint checkpoints/avste.pt \\
      --output     outputs/enhanced_tokens.pt

  # Save as .json (ready for compare_demo.py / Moshi streaming inference)
  python scripts/infer_avste.py \\
      --audio      examples/sample_noisy.wav \\
      --video      examples/sample_lip.mp4 \\
      --checkpoint checkpoints/avste.pt \\
      --output     outputs/my_predictions.json \\
      --utt_id     my_clip_001

  # Save as .wav (reconstructed speech, decoded through Mimi)
  python scripts/infer_avste.py \\
      --audio      examples/sample_noisy.wav \\
      --video      examples/sample_lip.mp4 \\
      --checkpoint checkpoints/avste.pt \\
      --output     outputs/enhanced_speech.wav
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Import fairseq from site-packages NOW, before load_model calls import_user_module,
# which prepends av_hubert/ to sys.path and would shadow the installed fairseq
# with the bundled copy at av_hubert/fairseq/.
import fairseq                           # noqa: F401  (side-effect: caches in sys.modules)
import fairseq.checkpoint_utils          # noqa: F401
import fairseq.tasks                     # noqa: F401
import fairseq.utils                     # noqa: F401

# ── repo paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]   # av-ste/
AVHUBERT_DIR = ROOT / "av_hubert" / "avhubert"

# ── audio constants ────────────────────────────────────────────────────────────
AUDIO_SR    = 16_000
STACK_ORDER = 4           # consecutive fbank frames stacked → 26 × 4 = 104 dim

# ── video constants ────────────────────────────────────────────────────────────
IMAGE_MEAN  = 0.421       # AV-HuBERT pixel mean (grayscale, [0,1])
IMAGE_STD   = 0.165       # AV-HuBERT pixel std
IMAGE_CROP  = 88          # centre-crop from 96×96 to 88×88

# ── Mimi tokenizer constants ───────────────────────────────────────────────────
MIMI_SR     = 24_000
MIMI_STRIDE = 1920        # 24000 / 12.5 = 1920 samples per Mimi frame


# ── preprocessing ──────────────────────────────────────────────────────────────

def load_audio_feats(wav_path: str, normalize: bool = True) -> np.ndarray:
    """Load 16 kHz WAV → stacked log-filterbank features [T, 320]."""
    from python_speech_features import logfbank
    from scipy.io import wavfile

    sr, wav = wavfile.read(wav_path)
    if sr != AUDIO_SR:
        raise ValueError(
            f"Expected 16 kHz audio, got {sr} Hz. "
            "Resample with: ffmpeg -i input.wav -ar 16000 output.wav"
        )

    feats = logfbank(wav, samplerate=sr).astype(np.float32)  # [T_audio, 26]

    # stack STACK_ORDER consecutive fbank frames → one video frame of audio context
    feat_dim = feats.shape[1]
    if len(feats) % STACK_ORDER != 0:
        pad = STACK_ORDER - len(feats) % STACK_ORDER
        feats = np.concatenate([feats, np.zeros([pad, feat_dim], dtype=feats.dtype)])
    feats = feats.reshape(-1, STACK_ORDER * feat_dim)         # [T_video, 320]

    if normalize:
        m, s = feats.mean(axis=0), feats.std(axis=0)
        feats = (feats - m) / (s + 1e-8)

    return feats


def load_video_feats(mp4_path: str) -> np.ndarray:
    """Load 96×96 grayscale mouth-ROI MP4 → normalized frames [T, 88, 88, 1]."""
    import cv2

    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # [H, W]
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {mp4_path}")

    frames = np.stack(frames).astype(np.float32)  # [T, 96, 96]
    frames /= 255.0                               # → [0, 1]

    # centre-crop 96 → 88
    h, w = frames.shape[1], frames.shape[2]
    y0 = (h - IMAGE_CROP) // 2
    x0 = (w - IMAGE_CROP) // 2
    frames = frames[:, y0:y0 + IMAGE_CROP, x0:x0 + IMAGE_CROP]   # [T, 88, 88]

    frames = (frames - IMAGE_MEAN) / IMAGE_STD    # standardize
    return frames[:, :, :, np.newaxis]            # [T, 88, 88, 1]


class MimiLogitExtractor:
    """
    Extracts cosine-similarity logits over the Mimi semantic codebook.
    Returns [T, 2048] float16 at 12.5 Hz.
    """

    def __init__(self, device: torch.device):
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(weight, device=device)
        mimi.eval()
        self.mimi   = mimi
        self.device = device
        cb = mimi.quantizer.semantic_quantizer.vq.layers[0]._codebook.embedding.to(device)
        self.codebook = F.normalize(cb, dim=-1)  # [2048, D]

    @torch.no_grad()
    def extract(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != MIMI_SR:
            wav = torchaudio.functional.resample(wav, sr, MIMI_SR)
        T   = wav.size(-1)
        pad = (MIMI_STRIDE - T % MIMI_STRIDE) % MIMI_STRIDE
        if pad:
            wav = F.pad(wav, (0, pad))

        enc      = self.mimi.encoder(wav.unsqueeze(0).to(self.device))    # [1, 512, T_f]
        enc_proj = self.mimi.quantizer.semantic_quantizer.input_proj(enc)
        enc_proj = enc_proj.squeeze(0).T                                   # [T_f, 256]
        enc_norm = F.normalize(enc_proj, dim=-1)
        return (enc_norm @ self.codebook.T).half()                         # [T_f, 2048]

    @torch.no_grad()
    def decode_enhanced(self, wav_path: str, enhanced_cb0_25hz: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a waveform from the enhanced semantic tokens, keeping the
        noisy input's own acoustic codebooks (cb1-7) unchanged -- same
        splice used for the paper's reconstructed-speech WER metric.
        Returns a [1, T] waveform at MIMI_SR (24 kHz).
        """
        wav, sr = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != MIMI_SR:
            wav = torchaudio.functional.resample(wav, sr, MIMI_SR)
        T_raw = wav.size(-1)
        pad = (MIMI_STRIDE - T_raw % MIMI_STRIDE) % MIMI_STRIDE
        if pad:
            wav = F.pad(wav, (0, pad))

        acoustic_codes = self.mimi.encode(wav.unsqueeze(0).to(self.device))  # [1, 8, T_mimi]

        # enhanced_cb0_25hz is at 25 Hz; acoustic codes are at 12.5 Hz
        cb0 = enhanced_cb0_25hz.cpu()[::2]
        T = min(cb0.numel(), acoustic_codes.size(2))
        codes = acoustic_codes[:, :, :T].clone()
        codes[0, 0, :T] = cb0[:T].to(acoustic_codes.device)

        return self.mimi.decode(codes).squeeze(0).cpu()  # [1, T_samples]


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device, fp16: bool,
                w2v_path: str | None = None):
    """
    Load AV-STE from a fairseq checkpoint.
    Requires the av_hubert user_dir to be registered (see AVHUBERT_DIR).

    avste.pt already contains the full fine-tuned state dict (backbone +
    fusion module), but fairseq still needs to build the model skeleton
    first, and that build step reads the AV-HuBERT backbone's own
    architecture config from `w2v_path` (whatever machine/path avste.pt
    was originally trained on). Its *weights* get completely overwritten
    by avste.pt's state dict right after, but its *config* is required to
    construct the right architecture -- so `w2v_path` must point to a real
    checkpoint file (weights aside), not just any string. We override it
    here via fairseq's `arg_overrides` mechanism so avste.pt's baked-in
    training-machine path never needs to resolve.
    """
    from fairseq import checkpoint_utils
    from fairseq import utils as fu

    fu.import_user_module(
        type("_ns", (), {"user_dir": str(AVHUBERT_DIR)})()
    )

    if w2v_path is None:
        w2v_path = str(ROOT / "checkpoints" / "large_vox_iter5.pt")
    if not Path(w2v_path).is_file():
        raise FileNotFoundError(
            f"AV-HuBERT backbone checkpoint not found: {w2v_path}\n"
            "This is a separate file from avste.pt -- see checkpoints/README.md "
            "to download it (only its architecture config is used; its weights "
            "are immediately overwritten by avste.pt)."
        )

    models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint_path)],
        arg_overrides={"w2v_path": w2v_path},
    )
    model = models[0]
    model.eval()
    if fp16:
        model.half()
    model.to(device)
    return model, saved_cfg


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AV-STE: enhance speech tokens from noisy audio + lip video"
    )
    p.add_argument("--audio",      required=True, help="path to noisy 16 kHz WAV")
    p.add_argument("--video",      required=True,
                   help="path to 96×96 grayscale mouth-ROI MP4 at 25 fps")
    p.add_argument("--checkpoint", required=True, help="path to avste.pt checkpoint")
    p.add_argument("--w2v_path",   default=None,
                   help="path to the AV-HuBERT backbone checkpoint (large_vox_iter5.pt) "
                        "used only to build the model architecture -- its weights are "
                        "immediately overwritten by --checkpoint. "
                        "Default: checkpoints/large_vox_iter5.pt")
    p.add_argument("--output",     required=True,
                   help="output path: .pt for raw LongTensor, .json for predictions format")
    p.add_argument("--utt_id",    default=None,
                   help="clip ID written into the JSON output (default: stem of --audio)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true",
                   help="run in fp16 (recommended on GPU)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[AV-STE] device={device}  fp16={args.fp16}")
    print(f"         audio  → {args.audio}")
    print(f"         video  → {args.video}")

    # 1. Preprocess audio
    print("[1/4] Preprocessing audio …")
    audio_np = load_audio_feats(args.audio, normalize=True)   # [T, 320]

    # 2. Preprocess video
    print("[2/4] Preprocessing video …")
    video_np = load_video_feats(args.video)                   # [T, 88, 88, 1]

    # align frame counts (small rounding differences between audio/video)
    T = min(len(audio_np), len(video_np))
    audio_np, video_np = audio_np[:T], video_np[:T]

    dtype = torch.float16 if args.fp16 else torch.float32
    # audio_np is [T, 104]; model expects [B, F, T] = [1, 104, T]
    audio_t = torch.from_numpy(audio_np).to(dtype).T.unsqueeze(0).to(device)
    # video_np is [T, 88, 88, 1]; model expects [B, C, T, H, W] = [1, 1, T, 88, 88]
    video_t = torch.from_numpy(video_np[:, :, :, 0]).to(dtype).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Extract Mimi logits on-the-fly at 12.5 Hz, upsample to 25 Hz
    print("[3/4] Extracting Mimi logits …")
    extractor = MimiLogitExtractor(device)
    logits_12hz = extractor.extract(args.audio)                  # [T_mimi, 2048]
    logits_25hz = logits_12hz.repeat_interleave(2, dim=0)        # → 25 Hz
    noisy_logits = logits_25hz[:T].unsqueeze(0).to(device)       # [1, T, 2048]
    if not args.fp16:
        noisy_logits = noisy_logits.float()

    # 4. Load model and run inference
    print("[4/4] Loading checkpoint and running inference …")
    model, _ = load_model(args.checkpoint, device, args.fp16, w2v_path=args.w2v_path)

    with torch.no_grad():
        net_output = model(
            source={"video": video_t, "audio": audio_t},
            padding_mask=None,
            noisy_logits=noisy_logits,
        )

    # encoder_out is [T, B, V] when tbc=True
    enc_out = net_output["encoder_out"]             # [T, 1, 2048]
    tokens  = enc_out.argmax(dim=-1).squeeze(1).cpu()   # [T]

    # 5. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".json":
        utt_id = args.utt_id or Path(args.audio).stem
        with open(out_path, "w") as f:
            json.dump([{"utt_id": utt_id, "pred_tokens": tokens.tolist()}], f)
        print(f"\nDone. {tokens.shape[0]} enhanced tokens saved → {out_path}  (utt_id={utt_id!r})")
        print(f"First 20 token IDs: {tokens[:20].tolist()}")
    elif out_path.suffix in (".wav", ".flac"):
        print("[5/5] Decoding enhanced tokens + noisy acoustic codes to speech …")
        wav_out = extractor.decode_enhanced(args.audio, tokens)
        torchaudio.save(str(out_path), wav_out, MIMI_SR)
        print(f"\nDone. Reconstructed speech saved → {out_path}")
    else:
        torch.save(tokens, out_path)
        print(f"\nDone. {tokens.shape[0]} enhanced tokens saved → {out_path}")
        print(f"First 20 token IDs: {tokens[:20].tolist()}")


if __name__ == "__main__":
    main()
