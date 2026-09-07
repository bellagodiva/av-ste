"""
Standalone teacher logit extractor — no fairseq DataLoader.

Reads train.tsv directly, loads audio+video+noisy_logits per sample,
runs the frozen teacher model, saves encoder_out as [T, 2048] float16 .npy.

Usage:
    CUDA_VISIBLE_DEVICES=2 \\
    NOISY_LOGITS_ROOT=/mnt/hard1/bella/EMNLP26/dataset/LRS3/noisy_mimi_logits \\
    python extract_teacher_logits.py \\
        --tsv   /mnt/hard1/bella/EMNLP26/dataset/LRS3/433h/train.tsv \\
        --ckpt  /mnt/hard1/bella/EMNLP26/experiment/av_causal_lookahead4_crossattn_soft/checkpoints/checkpoint_best.pt \\
        --user_dir /mnt/hard1/bella/EMNLP26/source/av_hubert/avhubert \\
        --out   /mnt/hard1/bella/EMNLP26/dataset/LRS3/teacher_logits \\
        [--rank 0 --world_size 1]
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from python_speech_features import logfbank


# ─── audio helpers (mirrors hubert_dataset.py) ──────────────────────────────

STACK_ORDER = 4   # stack_order_audio

def load_audio_features(wav_path: str) -> torch.Tensor:
    """Returns log-fbank features [T_audio, 104] normalised, as float32."""
    sample_rate, wav = wavfile.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    feats = logfbank(wav, samplerate=sample_rate).astype(np.float32)  # [T, 26]
    # stack 4 consecutive frames → [T//4, 104]
    T = len(feats)
    T_cut = (T // STACK_ORDER) * STACK_ORDER
    feats = feats[:T_cut].reshape(T_cut // STACK_ORDER, -1)          # [T', 104]
    return torch.from_numpy(feats)   # [T', 104]


# ─── video helpers ──────────────────────────────────────────────────────────

def load_video_frames(mp4_path: str) -> torch.Tensor:
    """Returns [T, H, W] uint8 grayscale tensor."""
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # [H, W]
        frames.append(gray)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {mp4_path}")
    return torch.from_numpy(np.stack(frames))   # [T, H, W]


def preprocess_video(frames: torch.Tensor) -> torch.Tensor:
    """Normalize + centre-crop 88×88, returns [1, T, 88, 88] float32."""
    frames = frames.float()
    frames = frames / 255.0
    frames = (frames - 0.421) / 0.165
    H, W = frames.shape[1], frames.shape[2]
    h0 = (H - 88) // 2
    w0 = (W - 88) // 2
    frames = frames[:, h0:h0+88, w0:w0+88]        # [T, 88, 88]
    return frames.unsqueeze(0)                     # [1, T, 88, 88]


# ─── model loading ──────────────────────────────────────────────────────────

def load_model(ckpt_path: str, user_dir: str, device: torch.device):
    av_hubert_root = os.path.dirname(user_dir)
    fairseq_root = os.path.join(av_hubert_root, "fairseq")
    # fairseq_root  → local fork wins over any installed fairseq
    # av_hubert_root → `importlib.import_module("avhubert")` finds the package
    # user_dir must NOT be added: the avhubert modules use relative imports when
    # len(sys.argv)>1 (DBG=False), so bare-import double-registration is avoided.
    for p in (fairseq_root, av_hubert_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    # checkpoint_utils never calls import_user_module, so we must do it manually
    # BEFORE load_model_ensemble_and_task to register all custom models/tasks.
    from argparse import Namespace
    from fairseq.utils import import_user_module
    import_user_module(Namespace(user_dir=user_dir))

    from fairseq import checkpoint_utils
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path], arg_overrides={"user_dir": user_dir}
    )
    model = models[0].to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─── per-sample forward ─────────────────────────────────────────────────────

@torch.no_grad()
def run_one(model, audio_feat, video_frames, noisy_logits_12, device):
    """
    audio_feat       : [T_a, 104]  log-fbank stacked
    video_frames     : [1, T_v, H, W]  preprocessed
    noisy_logits_12  : [T_12, 2048] float32

    Returns logits [T_out, 2048] float32.
    """
    # ── audio: [1, 104, T_a] ──────────────────────────────────────────────
    audio = audio_feat.T.unsqueeze(0).to(device)           # [1, 104, T_a]

    # ── video: [1, 1, T_v, 88, 88] ───────────────────────────────────────
    video = video_frames.unsqueeze(1).to(device)           # [1, 1, T_v, 88, 88]

    # ── noisy logits: upsample 12.5→25 Hz ────────────────────────────────
    noisy_25 = noisy_logits_12.repeat_interleave(2, dim=0)  # [T_25, 2048]
    noisy_25 = noisy_25.unsqueeze(0).to(device)             # [1, T_25, 2048]

    # ── align T dimension (use min of audio/video frame counts) ──────────
    T_a = audio.size(-1)    # audio frames (stacked, so same rate as video)
    T_v = video.size(2)
    T   = min(T_a, T_v)
    audio = audio[:, :, :T]
    video = video[:, :, :T]

    T_25 = noisy_25.size(1)
    if T_25 < T:
        noisy_25 = F.pad(noisy_25, (0, 0, 0, T - T_25))
    else:
        noisy_25 = noisy_25[:, :T]

    padding_mask = torch.zeros(1, T, dtype=torch.bool, device=device)

    source = {"audio": audio, "video": video}
    net_input = {
        "source": source,
        "padding_mask": padding_mask,
        "noisy_logits": noisy_25,
    }

    out = model(**net_input)
    raw = out["encoder_out"]   # [T, 1, V] or [1, T, V]

    if raw.dim() == 3:
        if raw.size(0) == 1:
            raw = raw.squeeze(0)       # [T, V]
        elif raw.size(1) == 1:
            raw = raw.squeeze(1)       # [T, V]
        else:
            # [T, B, V] → squeeze B=1
            raw = raw[:, 0, :]

    pad = out.get("encoder_padding_mask")
    if pad is not None:
        valid = ~pad[0].cpu()
        T_min = min(raw.size(0), valid.size(0))
        raw = raw[:T_min][valid[:T_min]]

    return raw.float().cpu()   # [T_valid, 2048]


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv",        required=True)
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--user_dir",   required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--rank",       type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    noisy_logits_root = os.environ.get("NOISY_LOGITS_ROOT")
    assert noisy_logits_root, "Set NOISY_LOGITS_ROOT env var"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rank {args.rank}/{args.world_size}] Loading model...", flush=True)
    model = load_model(args.ckpt, args.user_dir, device)
    print("Model loaded.", flush=True)

    # parse TSV
    with open(args.tsv) as f:
        lines = f.read().splitlines()
    entries = []
    for line in lines[1:]:
        parts = line.split("\t")
        fid, video_path, audio_path = parts[0], parts[1], parts[2]
        entries.append((fid, video_path, audio_path))

    # shard
    entries = entries[args.rank::args.world_size]
    print(f"[rank {args.rank}] {len(entries)} utterances to process.", flush=True)

    done = skipped = errors = 0
    for i, (fid, video_path, audio_path) in enumerate(entries):
        out_path = os.path.join(args.out, fid + ".npy")
        if os.path.exists(out_path):
            done += 1
            continue

        # load noisy logits
        logit_path = os.path.join(noisy_logits_root, fid + ".npy")
        if not os.path.exists(logit_path):
            skipped += 1
            continue

        try:
            noisy_logits = torch.from_numpy(
                np.load(logit_path).astype(np.float32)
            )   # [T_12, 2048]

            audio_feat   = load_audio_features(audio_path)
            video_frames = preprocess_video(load_video_frames(video_path))

            logits = run_one(model, audio_feat, video_frames, noisy_logits, device)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, logits.half().numpy())
            done += 1

        except Exception as e:
            print(f"[ERROR] {fid}: {e}", flush=True)
            errors += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(entries)}] done={done} skipped={skipped} errors={errors}", flush=True)

    print(f"[rank {args.rank}] Finished. done={done} skipped={skipped} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
