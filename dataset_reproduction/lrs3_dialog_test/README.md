# LRS3 dialogue test set (125 clips)
LRS3 is distributed under a license agreement that restricts redistribution
of the underlying media, and the raw video/audio ultimately comes from
third-party TED/TEDx recordings. We do not host or redistribute any LRS3
clips here. Instead, this directory contains exactly what's needed to
reconstruct our 125-clip single-turn dialogue evaluation set (Section 4.1)
from **your own** independently obtained LRS3 license:

- **`clip_ids.txt`** -- the 125 LRS3 clip IDs used in the paper (format:
  `test/<video_id>/<segment_id>`, matching LRS3's own directory layout). No
  paths, no media, no content -- just identifiers into a dataset you access
  yourself.
- **`sample_lrs3_test_subset.py`** -- the exact sampling script. Run it
  against your own `test_clean.tsv` manifest with `--seed 42` (the default)
  and it reproduces `clip_ids.txt` 

## Usage

1. Obtain LRS3 and generate your own `test_clean.tsv` manifest (standard
   AV-HuBERT/fairseq TSV format), following LRS3's own access process.
2. Run:
   ```bash
   python sample_lrs3_test_subset.py \
       --input_tsv  /path/to/your/test_clean.tsv \
       --output_tsv /path/to/output/test_clean_125.tsv \
       --seed 42
   ```
3. Confirm it matches:
   ```bash
   tail -n +2 /path/to/output/test_clean_125.tsv | cut -f1 | diff - clip_ids.txt && echo "matches"
   ```
