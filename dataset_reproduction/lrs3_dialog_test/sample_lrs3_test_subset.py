"""
Sample 125 LRS3 test clips with duration 3-10 seconds and write a subset TSV.

Reproduces the paper's 125-clip dialogue evaluation set exactly: with
--seed 42 (default), the sampled clip IDs match clip_ids.txt in this
directory byte-for-byte (verified). Requires your own licensed copy of
LRS3's test_clean.tsv manifest -- see the README in this directory for why
this repo ships IDs and a script rather than the LRS3 media itself.

Usage:
    python sample_lrs3_test_subset.py \
        --input_tsv  /path/to/your/lrs3/433h/test_clean.tsv \
        --output_tsv /path/to/output/test_clean_125.tsv \
        --n 125 \
        --min_sec 3.0 \
        --max_sec 10.0 \
        --seed 42
"""

import argparse
import random
from pathlib import Path


VIDEO_FPS  = 25
AUDIO_SR   = 16_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv",  required=True)
    ap.add_argument("--output_tsv", required=True)
    ap.add_argument("--n",        type=int,   default=125)
    ap.add_argument("--min_sec",  type=float, default=3.0)
    ap.add_argument("--max_sec",  type=float, default=10.0)
    ap.add_argument("--seed",     type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    input_tsv  = Path(args.input_tsv)
    output_tsv = Path(args.output_tsv)

    with input_tsv.open() as f:
        lines = f.readlines()

    root    = lines[0].rstrip("\n")   # first line is the data root
    entries = lines[1:]

    eligible = []
    for line in entries:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        n_frames = int(parts[3])
        dur = n_frames / VIDEO_FPS
        if args.min_sec <= dur <= args.max_sec:
            eligible.append(line)

    print(f"Eligible clips ({args.min_sec}–{args.max_sec}s): {len(eligible)} / {len(entries)}")

    if len(eligible) < args.n:
        raise ValueError(
            f"Only {len(eligible)} eligible clips but requested {args.n}. "
            "Lower --n or widen the duration range."
        )

    sampled = random.sample(eligible, args.n)
    sampled.sort()   # stable order for reproducibility

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w") as f:
        f.write(root + "\n")
        f.writelines(sampled)

    # print duration stats
    durs = [int(l.split("\t")[3]) / VIDEO_FPS for l in sampled]
    print(f"Sampled : {len(sampled)}")
    print(f"Duration: min={min(durs):.1f}s  max={max(durs):.1f}s  "
          f"mean={sum(durs)/len(durs):.1f}s")
    print(f"Written → {output_tsv}")


if __name__ == "__main__":
    main()
