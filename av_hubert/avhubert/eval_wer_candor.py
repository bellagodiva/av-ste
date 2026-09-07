"""
WER evaluation after infer_mimi.py.

Step 1 — run infer_mimi.py to get predictions.json:
    python infer_mimi.py \
        --config-dir conf --config-name infer \
        common_eval.path=<checkpoint> \
        common_eval.results_path=<out_dir> \
        override.data=<data_dir> \
        override.label_dir=<label_dir> \
        override.modalities="[audio,video]" \
        dataset.gen_subset=test

Step 2 — run this script:
    python eval_wer_mimi.py \
        --predictions <out_dir>/predictions.json \
        --tsv <data_dir>/test.tsv \
        --out_dir <out_dir>/wer \
        [--clean_audio_root /path/to/clean/audio]   # if clean audio is in a different root
        [--noisy_audio_root /path/to/noisy/audio]   # if noisy audio is in a different root
        [--whisper_model base.en]
        [--n_samples 100]

TSV format (tab-separated, first line = root dir):
    /root
    id \\t video_path \\t noisy_audio_path \\t n_frames \\t n_samples

Clean audio path is derived by replacing 'noisy_audio' with 'audio' in the noisy path.
If your TSV already points to clean audio, set --noisy_snr to add noise programmatically
or just pass --clean_audio_root to redirect paths.
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
import re
from pathlib import Path
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
# TSV reader — matches your dataset format
# ──────────────────────────────────────────────────────────────────────────────

def normalize_sample_id(x: str) -> str:
    x = str(x).strip()
    base = Path(x).name

    m = re.search(r"sample_(\d{1,6})", base)
    if m:
        return f"{int(m.group(1)):03d}"

    m = re.fullmatch(r"(\d{1,6})", x)
    if m:
        return f"{int(m.group(1)):03d}"

    m = re.search(r"(\d{1,6})", base)
    if m:
        return f"{int(m.group(1)):03d}"

    raise ValueError(f"Could not normalize sample id from: {x}")


def load_candor_transcripts(transcript_dir: str) -> dict:
    """
    Parse all transcript_cliffhanger.csv files under transcript_dir and return
    a dict mapping (session_uuid, turn_id_str) -> utterance text.
    transcript_dir: path to candor_eval root (contains one subdir per session UUID).
    """
    import csv
    mapping = {}
    root = Path(transcript_dir)
    for csv_path in root.rglob("transcript_cliffhanger.csv"):
        session_uuid = csv_path.parent.parent.name
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (session_uuid, row["turn_id"])
                mapping[key] = row["utterance"].strip()
    return mapping


def lookup_ref_from_filename(clean_path: str, transcript_map: dict) -> str:
    """
    Extract (session_uuid, turn_id) from a clean wav filename like:
        000_832b10fd-545f-4001-9eb3-3804c774272e_turn162_spk5eeaddb2....wav
    and return the ground-truth utterance.
    """
    stem = Path(clean_path).stem   # e.g. 000_832b10fd-..._turn162_spk5eea...
    parts = stem.split("_")
    # find the part starting with 'turn'
    session_uuid = None
    turn_id = None
    for i, p in enumerate(parts):
        if p.startswith("turn") and p[4:].isdigit():
            turn_id = p[4:]
            # session UUID is the part before this (after the numeric prefix)
            # reconstruct: parts[1] through parts[i-1] joined by '_'
            session_uuid = "_".join(parts[1:i])
            break
    if session_uuid is None or turn_id is None:
        return None
    return transcript_map.get((session_uuid, turn_id))


def read_tsv(tsv_path: str, clean_map_path: str = None):
    clean_map = {}
    if clean_map_path:
        with open(clean_map_path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    raw_id, clean_path = parts
                    clean_map[normalize_sample_id(raw_id)] = clean_path

    rows = {}
    with open(tsv_path) as f:
        lines = [l.rstrip("\n") for l in f]

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        utt_id = str(parts[0])
        noisy_path = parts[2]

        sample_id = normalize_sample_id(noisy_path)

        if clean_map_path:
            if sample_id not in clean_map:
                raise KeyError(f"Missing clean map for sample_id={sample_id}, noisy={noisy_path}")
            clean_path = clean_map[sample_id]
        else:
            raise ValueError("CandOR evaluation requires --clean_map")

        row = {
            "noisy": noisy_path,
            "clean": clean_path,
            "sample_id": sample_id,
        }

        rows[utt_id] = row
        rows[sample_id] = row
        rows[f"sample_{sample_id}"] = row

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # load predictions from infer_mimi.py
    with open(args.predictions) as f:
        predictions = json.load(f)

    if args.n_samples:
        predictions = predictions[: args.n_samples]

    # index by utt_id (may be int or string in the JSON)
    pred_by_id = {str(p["utt_id"]): p["pred_tokens"] for p in predictions}

    # load TSV
    tsv_map = read_tsv(
        args.tsv,
        clean_map_path=args.clean_map,
    )

    print(f"[INFO] predictions: {len(pred_by_id)} | TSV entries: {len(tsv_map)}")

    # load Mimi
    print("[INFO] loading Mimi...")
    mimi_weight = loaders.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(1)
    mimi.eval()

    # load Whisper
    print(f"[INFO] loading Whisper ({args.whisper_model})...")
    whisper_model = whisper.load_model(args.whisper_model, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load CandOR ground-truth transcripts if available
    transcript_map = {}
    if args.transcript_dir:
        print(f"[INFO] loading CandOR transcripts from {args.transcript_dir}...")
        transcript_map = load_candor_transcripts(args.transcript_dir)
        print(f"[INFO] loaded {len(transcript_map)} transcript entries")

    results = []
    skipped = 0

    for utt_id, pred_tokens_list in tqdm(pred_by_id.items(), desc="Evaluating"):
        if utt_id not in tsv_map:
            skipped += 1
            continue

        paths = tsv_map[utt_id]

        if not Path(paths["noisy"]).exists():
            print(f"[WARN] noisy audio not found: {paths['noisy']}")
            skipped += 1
            continue
        if not Path(paths["clean"]).exists():
            print(f"[WARN] clean audio not found: {paths['clean']}")
            skipped += 1
            continue

        try:
            # ── load audio ────────────────────────────────────────────────────
            noisy_wav = load_audio(paths["noisy"])
            clean_wav = load_audio(paths["clean"])

            # ── ground-truth transcript: CSV lookup first, else Whisper on clean ──
            ref = None
            if transcript_map:
                ref = lookup_ref_from_filename(paths["clean"], transcript_map)
                if ref:
                    ref = ref.lower()
            if not ref:
                ref = transcribe(whisper_model, clean_wav)

            # ── encode all 8 codebooks for noisy and clean ────────────────────
            # noisy_codes [8, T], clean_codes [8, T]
            noisy_codes = encode_all_codebooks(mimi, noisy_wav, device)
            clean_codes = encode_all_codebooks(mimi, clean_wav, device)

            # predicted cb0 from infer_mimi.py (25 Hz upsampled → take [::2])
            pred_tokens = torch.tensor(pred_tokens_list, dtype=torch.long)
            pred_cb0 = pred_tokens[::2]   # [T_tok] at 12.5 Hz

            # align lengths across all sources
            T = min(noisy_codes.size(1), clean_codes.size(1), pred_cb0.size(0))
            noisy_codes = noisy_codes[:, :T]   # [8, T]
            clean_codes = clean_codes[:, :T]   # [8, T]
            pred_cb0    = pred_cb0[:T]         # [T]

            # build predicted codes: cb0=pred + cb1-7=noisy
            pred_codes = noisy_codes.clone()   # [8, T]
            pred_codes[0] = pred_cb0           # replace semantic token

            # ── decode all 8 codebooks → waveform ────────────────────────────
            # noisy:  cb0=noisy + cb1-7=noisy  (baseline)
            # clean:  cb0=clean + cb1-7=clean  (oracle)
            # pred:   cb0=pred  + cb1-7=noisy  (ours)
            noisy_wav_dec = decode_all_codebooks(mimi, noisy_codes, device)
            clean_wav_dec = decode_all_codebooks(mimi, clean_codes, device)
            pred_wav_dec  = decode_all_codebooks(mimi, pred_codes,  device)

            # ── transcribe ────────────────────────────────────────────────────
            hyp_noisy = transcribe(whisper_model, noisy_wav_dec)
            hyp_clean = transcribe(whisper_model, clean_wav_dec)
            hyp_pred  = transcribe(whisper_model, pred_wav_dec)

            # ── WER ───────────────────────────────────────────────────────────
            wer_noisy = wer(hyp_noisy, ref)
            wer_clean = wer(hyp_clean, ref)
            wer_pred  = wer(hyp_pred,  ref)

            # ── token accuracy vs clean (cb0 only) ───────────────────────────
            tok_acc_noisy = (noisy_codes[0] == clean_codes[0]).float().mean().item()
            tok_acc_pred  = (pred_cb0 == clean_codes[0]).float().mean().item()

            results.append({
                "utt_id":            utt_id,
                "ref":               ref,
                "hyp_noisy":         hyp_noisy,
                "hyp_clean":         hyp_clean,
                "hyp_pred":          hyp_pred,
                "wer_noisy":         wer_noisy,
                "wer_clean":         wer_clean,
                "wer_pred":          wer_pred,
                "tok_acc_noisy":     tok_acc_noisy,
                "tok_acc_pred":      tok_acc_pred,
            })

        except Exception as e:
            print(f"[WARN] {utt_id}: {e}")
            skipped += 1
            continue

    print(f"[INFO] evaluated {len(results)} samples, skipped {skipped}")

    # ── save per-sample results ────────────────────────────────────────────────
    with open(out_dir / "per_sample_wer.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── aggregate ─────────────────────────────────────────────────────────────
    valid = [r for r in results if not any(
        np.isnan(r[k]) for k in ("wer_noisy", "wer_clean", "wer_pred")
    )]

    def mean_pct(key):
        return round(float(np.mean([r[key] for r in valid])) * 100, 2)

    summary = {
        "n_samples":              len(valid),
        "wer_noisy_mimi":         mean_pct("wer_noisy"),
        "wer_clean_mimi_oracle":  mean_pct("wer_clean"),
        "wer_pred_avhubert":      mean_pct("wer_pred"),
        "tok_acc_noisy_vs_clean": mean_pct("tok_acc_noisy"),
        "tok_acc_pred_vs_clean":  mean_pct("tok_acc_pred"),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Samples evaluated         : {summary['n_samples']}")
    print(f"  WER  noisy Mimi  (baseline): {summary['wer_noisy_mimi']:.1f}%")
    print(f"  WER  clean Mimi  (oracle)  : {summary['wer_clean_mimi_oracle']:.1f}%")
    print(f"  WER  AV-HuBERT   (ours)   : {summary['wer_pred_avhubert']:.1f}%")
    print(f"  Token acc noisy vs clean  : {summary['tok_acc_noisy_vs_clean']:.1f}%")
    print(f"  Token acc pred  vs clean  : {summary['tok_acc_pred_vs_clean']:.1f}%")
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

    # ── 1. Bar chart: mean ± std ───────────────────────────────────────────────
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
    ax.set_title("WER by Semantic Token Source", fontsize=13)
    ax.set_ylim(0, max(means) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_bar.png", dpi=150)
    plt.close()

    # ── 2. Per-sample scatter: noisy vs pred (diagonal = no change) ───────────
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
    ax.set_title("Per-Sample: Noisy vs AV-HuBERT", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_scatter.png", dpi=150)
    plt.close()

    # ── 3. WER reduction histogram ────────────────────────────────────────────
    delta = wer_noisy - wer_pred   # positive = improvement
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delta, bins=40, color=C["pred"], alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, linestyle="--", label="no change")
    ax.axvline(delta.mean(), color=C["noisy"], lw=2,
               linestyle="-", label=f"mean Δ = {delta.mean():.1f}%")
    ax.set_xlabel("WER Reduction (noisy − pred) %", fontsize=11)
    ax.set_ylabel("# Utterances", fontsize=11)
    ax.set_title("WER Improvement Distribution (AV-HuBERT vs Noisy Mimi)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_delta_hist.png", dpi=150)
    plt.close()

    # ── 4. Violin: all three distributions ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(
        [wer_noisy, wer_pred, wer_clean],
        positions=[1, 2, 3], showmedians=True, showextrema=True
    )
    for pc, c in zip(parts["bodies"], [C["noisy"], C["pred"], C["clean"]]):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Noisy Mimi", "AV-HuBERT (ours)", "Clean Mimi (oracle)"], fontsize=11)
    ax.set_ylabel("WER (%)", fontsize=12)
    ax.set_title("WER Distribution by Token Source", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "wer_violin.png", dpi=150)
    plt.close()

    # ── 5. Token accuracy bar ─────────────────────────────────────────────────
    tok_noisy = summary["tok_acc_noisy_vs_clean"]
    tok_pred  = summary["tok_acc_pred_vs_clean"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Noisy Mimi\nvs clean", "AV-HuBERT\nvs clean"],
        [tok_noisy, tok_pred],
        color=[C["noisy"], C["pred"]], alpha=0.85,
        edgecolor="black", linewidth=0.7, width=0.4,
    )
    for bar, m in zip(bars, [tok_noisy, tok_pred]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Token Accuracy vs Clean (%)", fontsize=12)
    ax.set_title("Semantic Token Match Rate", fontsize=13)
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "token_accuracy.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",     required=True,
                        help="predictions.json from infer_mimi.py")
    parser.add_argument("--tsv",             required=True,
                        help="test TSV file (same one used for inference)")
    parser.add_argument("--out_dir",    default="./wer_results")
    parser.add_argument("--clean_map",  default=None,
                        help="tab-separated file: id\\tclean_audio_path (for CandOR)")
    parser.add_argument("--transcript_dir",  default=None,
                        help="CandOR candor_eval root dir with per-session transcript CSVs")
    parser.add_argument("--whisper_model",   default="base.en",
                        help="tiny.en / base.en / small.en / medium.en")
    parser.add_argument("--n_samples",       type=int, default=None,
                        help="limit number of samples (for quick testing)")
    args = parser.parse_args()
    evaluate(args)
