"""
WER evaluation on LRS3 noisy test set after infer_mimi.py.

Usage:
    # Run infer_mimi.py first for each noise condition, then:

    python eval_wer_lrs3.py \
        --predictions /mnt/hard1/bella/EMNLP26/results/lrs3/speech_noise/snr_neg10/predictions.json \
        --tsv /mnt/hard1/bella/EMNLP26/dataset/LRS3/noisy_test/speech_noise/snr_neg10/test_noisy.tsv \
        --clean_audio_root /mnt/hard1/bella/EMNLP26/dataset/LRS3/audio \
        --out_dir /mnt/hard1/bella/EMNLP26/results/lrs3/speech_noise/snr_neg10/wer \
        --whisper_model base.en

TSV format (tab-separated, first line = root dir):
    /
    test/vid/clip \\t /LRS3/video/... \\t /LRS3/noisy_test/.../clip.wav \\t n_frames \\t n_samples

Clean audio is at: <clean_audio_root>/test/{vid}/{clip}.wav
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

try:
    import jiwer
except ImportError:
    raise ImportError("pip install jiwer")

try:
    import whisper
except ImportError:
    raise ImportError("pip install openai-whisper")

try:
    from moshi.models import loaders
except ImportError:
    raise ImportError("moshi not found — pip install moshi or add to PYTHONPATH")

MIMI_SR = 24_000


# ──────────────────────────────────────────────────────────────────────────────
# Audio / token helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = MIMI_SR) -> torch.Tensor:
    """Returns [1, T] mono waveform resampled to target_sr."""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def encode_all_codebooks(mimi, wav: torch.Tensor, device) -> torch.Tensor:
    """wav [1, T] → all 8 codebook tokens [8, T_tok] at 12.5 Hz."""
    mimi.set_num_codebooks(8)
    with torch.no_grad():
        codes = mimi.encode(wav.unsqueeze(0).to(device))  # [1, 8, T_tok]
    mimi.set_num_codebooks(1)
    return codes[0].cpu()   # [8, T_tok]


def load_or_encode(mimi, wav: torch.Tensor, cache_path: str | None, device) -> torch.Tensor:
    """Load [8, T] codes from cache_path if it exists, otherwise encode and save."""
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            return torch.from_numpy(np.load(str(p)).astype(np.int64))
    codes = encode_all_codebooks(mimi, wav, device)   # [8, T]
    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), codes.numpy().astype(np.int16))
    return codes


def decode_all_codebooks(mimi, codes: torch.Tensor, device) -> torch.Tensor:
    """codes [8, T_tok] → waveform [1, T_audio] at 24kHz."""
    mimi.set_num_codebooks(8)
    with torch.no_grad():
        wav = mimi.decode(codes.unsqueeze(0).to(device))  # [1, 1, T_audio]
    mimi.set_num_codebooks(1)
    return wav.squeeze(0).cpu()   # [1, T_audio]


def transcribe(whisper_model, wav: torch.Tensor, sr: int = MIMI_SR) -> str:
    """wav [1, T] → lowercased transcript string."""
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    wav_np = wav.squeeze(0).numpy().astype(np.float32)
    result = whisper_model.transcribe(wav_np, language="en", fp16=False)
    return result["text"].strip().lower()


def wer(hyp: str, ref: str) -> float:
    if not ref.strip():
        return float("nan")
    return jiwer.wer(ref, hyp)


# ──────────────────────────────────────────────────────────────────────────────
# TSV reader for LRS3 noisy test TSV
# ──────────────────────────────────────────────────────────────────────────────

def read_tsv(tsv_path: str, clean_audio_root: str) -> dict:
    """
    Parse LRS3 noisy_test TSV.
    utt_id format: test/{vid}/{clip}  e.g. test/QuIS7J59SYw/00001
    noisy audio col: /LRS3/noisy_test/.../test/{vid}/{clip}.wav
    clean audio: <clean_audio_root>/test/{vid}/{clip}.wav
    Returns dict: utt_id -> {noisy, clean}
    """
    clean_root = Path(clean_audio_root)
    rows = {}
    with open(tsv_path) as f:
        lines = [l.rstrip("\n") for l in f]

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        utt_id   = parts[0]          # e.g. test/QuIS7J59SYw/00001
        noisy_path = parts[2]

        # derive clean path: clean_audio_root / test / vid / clip.wav
        # utt_id = test/QuIS7J59SYw/00001  → test/QuIS7J59SYw/00001.wav
        clean_path = str(clean_root / (utt_id + ".wav"))

        rows[utt_id] = {
            "noisy": noisy_path,
            "clean": clean_path,
        }
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    with open(args.predictions) as f:
        predictions = json.load(f)

    if args.n_samples:
        predictions = predictions[: args.n_samples]

    pred_by_id = {str(p["utt_id"]): p["pred_tokens"] for p in predictions}

    tsv_map = read_tsv(args.tsv, args.clean_audio_root)
    print(f"[INFO] predictions: {len(pred_by_id)} | TSV entries: {len(tsv_map)}")

    print("[INFO] loading Mimi...")
    mimi_weight = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(1)
    mimi.eval()

    print(f"[INFO] loading Whisper ({args.whisper_model})...")
    whisper_model = whisper.load_model(args.whisper_model, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0

    for utt_id, pred_tokens_list in tqdm(pred_by_id.items(), desc="Evaluating"):
        if utt_id not in tsv_map:
            print(f"[WARN] {utt_id} not in TSV")
            skipped += 1
            continue

        paths = tsv_map[utt_id]

        if not Path(paths["noisy"]).exists():
            print(f"[WARN] noisy not found: {paths['noisy']}")
            skipped += 1
            continue
        if not Path(paths["clean"]).exists():
            print(f"[WARN] clean not found: {paths['clean']}")
            skipped += 1
            continue

        try:
            noisy_wav = load_audio(paths["noisy"])
            clean_wav = load_audio(paths["clean"])

            # reference transcript from Whisper on clean audio
            ref = transcribe(whisper_model, clean_wav)

            # encode all 8 codebooks (load from cache if available, save on first run)
            noisy_cache = str(Path(args.noisy_tokens_root) / (utt_id + ".npy")) \
                if args.noisy_tokens_root else None
            clean_cache = str(Path(args.clean_tokens_root) / (utt_id + ".npy")) \
                if args.clean_tokens_root else None

            noisy_codes = load_or_encode(mimi, noisy_wav, noisy_cache, device)  # [8, T]
            clean_codes = load_or_encode(mimi, clean_wav, clean_cache, device)  # [8, T]

            # predicted cb0 from infer_mimi.py at 25 Hz → downsample to 12.5 Hz
            pred_tokens = torch.tensor(pred_tokens_list, dtype=torch.long)
            pred_cb0 = pred_tokens[::2]   # [T_tok]

            # align lengths
            T = min(noisy_codes.size(1), clean_codes.size(1), pred_cb0.size(0))
            noisy_codes = noisy_codes[:, :T]
            clean_codes = clean_codes[:, :T]
            pred_cb0    = pred_cb0[:T]

            # build predicted codes: cb0=pred + cb1-7=noisy
            pred_codes = noisy_codes.clone()
            pred_codes[0] = pred_cb0

            # decode all 3 conditions
            noisy_wav_dec = decode_all_codebooks(mimi, noisy_codes, device)
            clean_wav_dec = decode_all_codebooks(mimi, clean_codes, device)
            pred_wav_dec  = decode_all_codebooks(mimi, pred_codes,  device)

            # transcribe
            hyp_noisy = transcribe(whisper_model, noisy_wav_dec)
            hyp_clean = transcribe(whisper_model, clean_wav_dec)
            hyp_pred  = transcribe(whisper_model, pred_wav_dec)

            # WER
            wer_noisy = wer(hyp_noisy, ref)
            wer_clean = wer(hyp_clean, ref)
            wer_pred  = wer(hyp_pred,  ref)

            # per-codebook match rate (noisy vs clean, all 8 CBs)
            cb_match_noisy = [
                (noisy_codes[cb] == clean_codes[cb]).float().mean().item()
                for cb in range(8)
            ]
            tok_acc_noisy = cb_match_noisy[0]   # cb0 semantic
            tok_acc_pred  = (pred_cb0 == clean_codes[0]).float().mean().item()

            results.append({
                "utt_id":         utt_id,
                "ref":            ref,
                "hyp_noisy":      hyp_noisy,
                "hyp_clean":      hyp_clean,
                "hyp_pred":       hyp_pred,
                "wer_noisy":      wer_noisy,
                "wer_clean":      wer_clean,
                "wer_pred":       wer_pred,
                "tok_acc_noisy":  tok_acc_noisy,
                "tok_acc_pred":   tok_acc_pred,
                "cb_match_noisy": cb_match_noisy,
            })

        except Exception as e:
            print(f"[WARN] {utt_id}: {e}")
            skipped += 1
            continue

    print(f"[INFO] evaluated {len(results)} samples, skipped {skipped}")

    with open(out_dir / "per_sample_wer.json", "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if not any(
        np.isnan(r[k]) for k in ("wer_noisy", "wer_clean", "wer_pred")
    )]

    def mean_pct(key):
        return round(float(np.mean([r[key] for r in valid])) * 100, 2)

    # per-codebook summary
    cb_means = [
        round(float(np.mean([r["cb_match_noisy"][cb] for r in valid])) * 100, 2)
        for cb in range(8)
    ]

    summary = {
        "n_samples":              len(valid),
        "wer_noisy_mimi":         mean_pct("wer_noisy"),
        "wer_clean_mimi_oracle":  mean_pct("wer_clean"),
        "wer_pred_avhubert":      mean_pct("wer_pred"),
        "tok_acc_noisy_vs_clean": mean_pct("tok_acc_noisy"),
        "tok_acc_pred_vs_clean":  mean_pct("tok_acc_pred"),
        "cb_match_noisy_vs_clean": cb_means,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Samples evaluated          : {summary['n_samples']}")
    print(f"  WER  noisy Mimi  (baseline): {summary['wer_noisy_mimi']:.1f}%")
    print(f"  WER  clean Mimi  (oracle)  : {summary['wer_clean_mimi_oracle']:.1f}%")
    print(f"  WER  AV-HuBERT   (ours)   : {summary['wer_pred_avhubert']:.1f}%")
    print(f"  Token acc noisy vs clean  : {summary['tok_acc_noisy_vs_clean']:.1f}%  (cb0)")
    print(f"  Token acc pred  vs clean  : {summary['tok_acc_pred_vs_clean']:.1f}%  (cb0)")
    print()
    print("  Per-codebook match rate (noisy vs clean):")
    for cb, m in enumerate(cb_means):
        tag = "semantic" if cb == 0 else "acoustic"
        print(f"    cb{cb} ({tag}): {m:.1f}%")
    print("=" * 60)

    visualize(results, summary, out_dir)
    print(f"\n[INFO] results saved to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize(results, summary, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"noisy": "#e74c3c", "clean": "#2ecc71", "pred": "#3498db"}

    wer_noisy = np.array([r["wer_noisy"] * 100 for r in results])
    wer_clean = np.array([r["wer_clean"] * 100 for r in results])
    wer_pred  = np.array([r["wer_pred"]  * 100 for r in results])

    # 1. Bar chart: mean WER
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Noisy Mimi\n(baseline)", "AV-HuBERT\n(ours)", "Clean Mimi\n(oracle)"]
    means  = [wer_noisy.mean(), wer_pred.mean(), wer_clean.mean()]
    stds   = [wer_noisy.std(),  wer_pred.std(),  wer_clean.std()]
    colors = [C["noisy"], C["pred"], C["clean"]]
    bars   = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                    capsize=5, width=0.5, edgecolor="black", linewidth=0.7)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Word Error Rate (%)", fontsize=12)
    ax.set_title("WER by Semantic Token Source (LRS3 Test)", fontsize=13)
    ax.set_ylim(0, max(means) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_bar.png", dpi=150)
    plt.close()

    # 2. Per-sample scatter: noisy vs pred
    fig, ax = plt.subplots(figsize=(6, 6))
    improved = wer_pred < wer_noisy
    ax.scatter(wer_noisy[improved],  wer_pred[improved],
               c=C["pred"],  alpha=0.5, s=18, label=f"improved ({improved.mean()*100:.0f}%)")
    ax.scatter(wer_noisy[~improved], wer_pred[~improved],
               c=C["noisy"], alpha=0.4, s=18, label="worse/same")
    lim = max(wer_noisy.max(), wer_pred.max()) + 5
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("WER — Noisy Mimi (%)", fontsize=11)
    ax.set_ylabel("WER — AV-HuBERT (%)", fontsize=11)
    ax.set_title("Per-Sample: Noisy vs AV-HuBERT (LRS3)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_scatter.png", dpi=150)
    plt.close()

    # 3. WER reduction histogram
    delta = wer_noisy - wer_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta, bins=40, color=C["pred"], alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, linestyle="--", label="no change")
    ax.axvline(delta.mean(), color=C["noisy"], lw=2,
               linestyle="-", label=f"mean Δ = {delta.mean():.1f}%")
    ax.set_xlabel("WER Reduction (noisy − pred) %", fontsize=11)
    ax.set_ylabel("# Utterances", fontsize=11)
    ax.set_title("WER Improvement Distribution (LRS3)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_delta_hist.png", dpi=150)
    plt.close()

    # 4. Per-codebook match rate bar chart
    cb_means = summary["cb_match_noisy_vs_clean"]
    fig, ax = plt.subplots(figsize=(9, 4))
    cb_labels = [f"cb{i}\n({'sem' if i==0 else 'ac'})" for i in range(8)]
    cb_colors = ["#e74c3c" if i == 0 else "#95a5a6" for i in range(8)]
    bars = ax.bar(cb_labels, cb_means, color=cb_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7)
    for bar, m in zip(bars, cb_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Match Rate vs Clean (%)", fontsize=12)
    ax.set_title("Per-Codebook Token Match Rate: Noisy vs Clean (LRS3)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "codebook_match.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",      required=True,
                        help="predictions.json from infer_mimi.py")
    parser.add_argument("--tsv",              required=True,
                        help="noisy test TSV (e.g. noisy_test/speech_noise/snr_neg10/test_noisy.tsv)")
    parser.add_argument("--clean_audio_root", required=True,
                        help="root of clean audio, e.g. /LRS3/audio (clips at <root>/test/vid/clip.wav)")
    parser.add_argument("--out_dir",          default="./wer_results")
    parser.add_argument("--whisper_model",    default="base.en",
                        help="tiny.en / base.en / small.en / medium.en")
    parser.add_argument("--n_samples",        type=int, default=None,
                        help="limit number of samples for quick testing")
    parser.add_argument("--noisy_tokens_root", default=None,
                        help="dir to cache/load [8,T] noisy RVQ codes (e.g. test_mimi_tokens/speech_noise/snr_0)")
    parser.add_argument("--clean_tokens_root", default=None,
                        help="dir to cache/load [8,T] clean RVQ codes (e.g. test_mimi_tokens/clean)")
    args = parser.parse_args()
    evaluate(args)
