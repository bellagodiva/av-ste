# AV-STE: Audio-Visual Speech Token Enhancement

> **EMNLP 2026** — *Audio-Visual Speech Token Enhancement for Robust Speech Synthesis under Noise*

AV-STE recovers clean Mimi semantic tokens from noisy speech by fusing audio features with lip-ROI video through an entropy-gated cross-attention module built on top of AV-HuBERT. Enhanced tokens can be fed directly into any Mimi-based TTS or codec LM (e.g., Moshi) in place of the noisy ones.

---

## Overview

```
Noisy audio (16 kHz) ──► AV-HuBERT encoder ──┐
                                               ├──► Entropy-gated cross-attention ──► Enhanced Mimi tokens
Lip video  (25 fps)  ──► AV-HuBERT encoder ──┘          ▲
                                                          │
Noisy audio          ──► Mimi encoder ──► soft logits ───┘
```

- **Input**: noisy 16 kHz WAV + 96×96 grayscale mouth-ROI MP4 at 25 fps
- **Output**: enhanced Mimi cb0 token sequence at 12.5 Hz
- **Token space**: Mimi semantic codebook (2048 entries)

---

## Released checkpoints: scope

Two AV-STE checkpoints are released (see [checkpoints/README.md](checkpoints/README.md)
for the full breakdown and download instructions):

| Checkpoint | Reproduces |
|---|---|
| `avste.pt` | Clean / Non-speech (AudioSet) / Speaker (AudioSet) rows (Table 1, in-domain) |
| `avste_lrs3_interference.pt` | Same-dataset LRS3 speaker-interference row, and the Seamless Interaction (out-of-domain) rows |

Together these two checkpoints reproduce every semantic-token-accuracy row in
the paper's main table. This repo is a working demo and inference tool for
the released checkpoints, not a full reproduction pipeline for every table in
the paper (e.g. WER-side diagnostics, ablations, and training code are not
included).

## Data and code availability

- **Code**: this repository, [github.com/bellagodiva/av-ste](https://github.com/bellagodiva/av-ste).
- **Model weights**: `avste.pt` and `avste_lrs3_interference.pt` on
  [Hugging Face Hub](https://huggingface.co/bellagodiva/av-ste) (see
  [checkpoints/README.md](checkpoints/README.md)). These are parameters we
  trained -- releasing them isn't a redistribution of any licensed dataset.
- **LRS3**: license-gated, and the underlying media is third-party
  (TED/TEDx recordings) -- **we do not host or redistribute any LRS3 clips**.
  For the 125-clip dialogue evaluation set specifically, we instead publish
  the clip IDs and the exact sampling script needed to reconstruct it from
  your own LRS3 license -- see
  [`dataset_reproduction/lrs3_dialog_test/`](dataset_reproduction/lrs3_dialog_test/).
- **AudioSet / Seamless Interaction**: also third-party/license-gated;
  obtain these from their own official sources (linked in
  [Data preparation](#2-data-preparation) below).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_ORG/av-ste.git
cd av-ste
```

### 2. Install dependencies

We recommend a dedicated conda environment:

```bash
conda create -n avste python=3.10 -y
conda activate avste

# PyTorch (adjust cuda version as needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install moshi huggingface_hub openai-whisper
pip install python-speech-features opencv-python
pip install omegaconf hydra-core

# AV-HuBERT / bundled fairseq
pip install -e av_hubert/fairseq/
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### 3. Download the checkpoints

Three files: our two fine-tuned AV-STE checkpoints (`avste.pt` and
`avste_lrs3_interference.pt` -- see [checkpoints/README.md](checkpoints/README.md#which-checkpoint-should-i-use)
for which one to use), and `large_vox_iter5.pt`, the public AV-HuBERT-Large
backbone both were fine-tuned from (needed to build the model architecture;
its weights are immediately overwritten by whichever AV-STE checkpoint you load).

```bash
bash scripts/download_checkpoint.sh
```

See [checkpoints/README.md](checkpoints/README.md) for manual download instructions.


Run the full end-to-end demo on the four bundled LRS3 test clips (speech
noise / ambient noise, each at −10 dB and −5 dB SNR):

```bash
bash examples/run_demo.sh
```

This will:
1. Run AV-STE inference on all four bundled noisy conditions
2. Print **Table A**: semantic-token accuracy (primary metric) + WER (secondary, diagnostic) of Mimi-decoded audio
3. Run **Table B**: feed each condition's tokens to Moshi streaming LM and show its generated speech

Example output for the speech-noise −5 dB clip:

```
  Table A — Speech noise −5 dB — Mimi decode
  Reference: BUT IF THEY DON'T STAY IN PARIS THE INTERNATIONAL PRESSURE WILL BE OVERWHELMING
  ─────────────────────────────────────────────────────────────────────────
  Condition               Tok Acc (primary)  WER (secondary)   Transcript
  ─────────────────────────────────────────────────────────────────────────
  Clean (upper bound)                100.0%             0.0%   But if they don't stay in Paris, the international pressure…
  Noisy │ Mimi baseline               27.6%            61.5%   The problem is they don't stay in Paris. The international…
  Noisy │ AV-STE (ours)               79.3%            38.5%   What if they don't stay in Paris? The international questi…
```

Table B (Moshi's generated response) shows the same story in a more visible
way -- Mimi baseline's corrupted input makes Moshi respond with a complete
non-sequitur, while AV-STE's recovered tokens keep Moshi on-topic:

```
  Table B — Speech noise −5 dB — Moshi streaming generation
  ─────────────────────────────────────────────────────────────────────────
  Condition               Input Tok Acc   Moshi's generated speech
  ─────────────────────────────────────────────────────────────────────────
  Clean (upper bound)          100.0%   That's true, and it's also possible that they could try to find a way to stay…
  Noisy │ Mimi baseline         27.6%   Yeah, I'm sure I've heard of that. What's it about?
  Noisy │ AV-STE (ours)         79.3%   That's true. They should at least try to stay in the city for a few days.
```

**Tok Acc (primary)** = fraction of predicted cb0 tokens matching the clean reference at 12.5 Hz. This is our main metric.
**WER (secondary)** = word error rate of Whisper-large-v3 on the Mimi-decoded audio, reported as a diagnostic only.

At extreme noise levels, Whisper occasionally transcribes nothing for a
clip; when that happens the table shows `(empty transcript -- Whisper heard
nothing)` in the transcript column rather than a blank cell, and WER reads
100%. Token accuracy is unaffected by this, since it never depends on ASR.

Note this table shows a single, deliberately illustrative clip, not an
average -- the paper's Table 1 numbers are averaged over the full 1321-clip
test set, where per-clip results vary a lot with how loud and speech-like
the specific noise draw is. Individual clips can be more extreme in either
direction than the table average; for instance, the bundled ambient/music
−5 dB clip pushes the Mimi baseline all the way to 0% (verified against the
raw cb0 token sequences, not a bug), harsher than the −5 dB row in Table 1.
See [Bundled samples](#bundled-samples) below for why these specific clips
were chosen, and the paper for representative aggregate numbers.

Audio files for all conditions (Mimi-decoded and Moshi-generated) are saved to `outputs/compare_audio/` for qualitative comparison.

### Enhance your own audio

This walks through going from your own noisy audio + face video to enhanced
semantic tokens or reconstructed speech, and optionally on to a
Moshi-generated spoken response.

#### Step 1 — process your video into a lip-ROI clip

AV-STE expects a 96×96 grayscale mouth-crop MP4 at 25 fps, not a raw face
video. `scripts/prepare_lip_roi.py` extracts this for you using dlib face
landmarks:

```bash
# One-time setup: download the dlib 68-point face landmark model
curl -LO https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Extract the lip-ROI clip from your face video
python scripts/prepare_lip_roi.py \
    --input      face_video.mp4 \
    --output     lip_roi.mp4 \
    --landmarks  shape_predictor_68_face_landmarks.dat
```

Your input video needs a visible, front-facing (or near-frontal) face for
landmark detection to work reliably -- this is the same requirement LRS3
itself was filtered for.

Also resample your audio to 16 kHz if it isn't already:

```bash
ffmpeg -i your_audio.wav -ar 16000 -ac 1 noisy_16khz.wav
```

#### Step 2 — pick a checkpoint

See [checkpoints/README.md](checkpoints/README.md#which-checkpoint-should-i-use)
for the full guidance; short version: use `avste.pt` for background/ambient
noise or a competing speaker from a different source, and
`avste_lrs3_interference.pt` for a same-corpus-style competing speaker or
out-of-domain video.

#### Step 3a — enhanced tokens, or reconstructed speech

`infer_avste.py`'s `--output` format is chosen by file extension -- pick
whichever you need:

```bash
# Enhanced semantic tokens only (for your own TTS/codec LM instead of Moshi)
python scripts/infer_avste.py \
    --audio      noisy_16khz.wav \
    --video      lip_roi.mp4 \
    --checkpoint checkpoints/avste.pt \
    --output     outputs/enhanced_tokens.pt \
    --fp16

# Reconstructed speech: enhanced tokens + your noisy audio's own acoustic
# codebooks, decoded through Mimi into a playable .wav
python scripts/infer_avste.py \
    --audio      noisy_16khz.wav \
    --video      lip_roi.mp4 \
    --checkpoint checkpoints/avste.pt \
    --output     outputs/enhanced_speech.wav \
    --fp16
```

`.pt` gives a `LongTensor` of shape `[T]` at 25 Hz (Mimi cb0 token IDs) --
use this if you're feeding tokens into your own downstream system. `.wav`
additionally decodes that into audio, which is convenient for listening but
is a secondary/diagnostic view of the result, not the primary one -- Moshi
(and any Mimi-token-based downstream model) consumes the tokens directly, so
token accuracy is what to optimize for, not how the reconstructed audio
sounds (see the paper's WER discussion). `.json` is a third option (below)
for streaming the tokens through Moshi itself rather than just decoding them.

#### Step 3b — enhanced tokens → Moshi streaming dialogue

Save predictions in the JSON format that `compare_demo.py` reads, then stream through Moshi:

```bash
# 1. Run AV-STE and save predictions as JSON
python scripts/infer_avste.py \
    --audio      noisy_16khz.wav \
    --video      lip_roi.mp4 \
    --checkpoint checkpoints/avste.pt \
    --output     outputs/my_predictions.json \
    --utt_id     my_clip_001 \
    --fp16

# 2. Stream through Moshi (loads Moshi/Mimi from HuggingFace automatically)
python scripts/compare_demo.py \
    --pred_json  outputs/my_predictions.json \
    --utt_id     my_clip_001 \
    --clean      clean_16khz.wav \
    --noisy      noisy_16khz.wav \
    --save_audio
```

All four of `--pred_json`, `--utt_id`, `--clean`, and `--noisy` are required
together for custom-clip mode -- if any one is missing, `compare_demo.py`
does not error, it silently falls back to running the four bundled demo
clips instead of yours. If you don't have a clean reference for your audio,
this script isn't set up for that case; use Step 3a directly instead.

Audio output is written to `outputs/compare_audio/`. No AV-STE checkpoint is needed for step 2.

---

## Bundled samples

The `examples/` directory contains four LRS3 test utterances (speech-noise and ambient/music-noise, each at two SNRs), chosen to clearly showcase AV-STE's recovery under noise. **These are illustrative, not representative** -- they were picked for a dramatic before/after contrast, so single-clip numbers here (e.g. Mimi baseline hitting 0% on the ambient-noise clips) can be more extreme than the paper's Table 1 averages, which are computed over the full 1321-clip test set with per-clip noise varying widely in loudness and spectral overlap with speech. For aggregate, representative numbers, see the paper.

**Speech noise −10 dB** — LRS3 `rP7nmdDA1Fg/00006`: *"I THINK WHAT THAT MEANS IS THAT PEOPLE JUST COULDN'T SEE WHAT WAS IN FRONT OF THEM"*

| File | Description |
|------|-------------|
| `sample_speech_clean.wav` | Clean audio at 16 kHz |
| `sample_speech_lip.mp4` | 96×96 grayscale mouth-ROI at 25 fps |
| `sample_noisy_speech.wav` | + speech/chatter noise at −10 dB SNR |
| `predictions_speech_neg10.json` | AV-STE predicted cb0 tokens (25 Hz) |

**Speech noise −5 dB** — LRS3 `ta2Wvy9FSgA/00003`: *"BUT IF THEY DON'T STAY IN PARIS THE INTERNATIONAL PRESSURE WILL BE OVERWHELMING"*

| File | Description |
|------|-------------|
| `sample_speech_neg5_clean.wav` | Clean audio at 16 kHz |
| `sample_speech_neg5_noisy.wav` | + speech/chatter noise at −5 dB SNR |
| `predictions_speech_neg5.json` | AV-STE predicted cb0 tokens (25 Hz) |



**Ambient/music noise −10 dB** — LRS3 `w1R4F9sSoow/00003`: *"WE HAVE IDEAS FOR HOW TO MAKE THINGS BETTER AND I WANT TO SHARE THREE OF THEM THAT WE'VE PICKED UP IN OUR OWN WORK"*

| File | Description |
|------|-------------|
| `sample_other_neg10_clean.wav` | Clean audio at 16 kHz |
| `sample_other_neg10_noisy.wav` | + ambient/music noise at −10 dB SNR |
| `predictions_other_neg10.json` | AV-STE predicted cb0 tokens (25 Hz) |

**Ambient/music noise −5 dB** — LRS3 `RplnSVTzvnU/00003`: *"WE CAN CREATE A DECENTRALIZED DATABASE THAT HAS THE SAME EFFICIENCY OF A MONOPOLY"*

| File | Description |
|------|-------------|
| `sample_other_neg5_clean.wav` | Clean audio at 16 kHz |
| `sample_other_neg5_noisy.wav` | + ambient/music noise at −5 dB SNR |
| `predictions_other_neg5.json` | AV-STE predicted cb0 tokens (25 Hz) |

---

## Reproducing the paper's experiments

This repo includes the actual training config and launch script
(`scripts/train.sh`) for both released checkpoints. What it does **not**
include is the data-preparation pipeline that builds the manifests and noise
pools `train.sh` expects as input (LRS3 TSV/label generation, AudioSet noise
pool construction, Mimi-logit pre-extraction, same-dataset interferer
selection) -- those scripts live in a larger internal research codebase not
intended for public release. This section documents that preparation
methodology precisely enough to reproduce it independently, in the same
spirit as how the paper describes it, even though the exact scripts aren't
included here.

### 1. Environment setup

Same as [Setup](#setup) above.

### 2. Data preparation

- **LRS3** -- the target-speech training and in-domain evaluation set.
  License-gated; start at the
  [Oxford VGG lip-reading datasets page](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/)
  and follow their current access process (availability has changed over
  time, so treat that page as the source of truth rather than this README).
  Preprocess with AV-HuBERT's standard pipeline: crop the mouth region from
  each face video into 96×96 grayscale frames at 25 fps.
- **AudioSet** -- source of the in-domain non-speech noise and cross-dataset
  speaker interference. We sample clips labeled `Speech`/`Conversation` for
  speaker interference and clips with none of those labels for non-speech
  noise, mixing one randomly-drawn clip per utterance at a uniformly sampled
  SNR in [-10, 10] dB.
- **Same-dataset LRS3 speaker interference** -- each target utterance is
  mixed with 1-4 interfering speakers drawn from LRS3 itself (excluding the
  target's own speaker), summed together and then scaled as one signal to
  hit the target aggregate SNR.
- **Seamless Interaction** (out-of-domain evaluation only, no fine-tuning) --
  a dyadic interaction video corpus, preprocessed the same way as LRS3
  (96×96 grayscale mouth crop, 25 fps, audio resampled to 16 kHz).
- **125-clip LRS3 dialogue evaluation set** -- we don't redistribute LRS3
  media (see [Data and code availability](#data-and-code-availability)); see
  [`dataset_reproduction/lrs3_dialog_test/`](dataset_reproduction/lrs3_dialog_test/)
  for the exact clip IDs and sampling script to reconstruct this set from
  your own LRS3 license.

### 3. Training

```bash
bash scripts/train.sh   # see the script header for required env vars and
                         # what each one needs to point at
```

Both checkpoints are `av_hubert_crossattn_ent` models (4-frame backbone
lookahead, 4-frame cross-attention lookahead, entropy-gated Noise-Adaptive
Modulation, `mimi_mix_loss` criterion), fine-tuned from the public
AV-HuBERT-Large VoxCeleb2 checkpoint (`large_vox_iter5.pt`), config at
[`configs/large_lrs3_433h_crossattn_ent.yaml`](configs/large_lrs3_433h_crossattn_ent.yaml).
Both use Adam (betas 0.9/0.98), a tri-stage LR schedule, and gradient clip
norm 5.0 -- the schedule itself differs between the two stages:

| | `avste.pt` (Stage 1) | `avste_lrs3_interference.pt` (Stage 2) |
|---|---|---|
| Started from | `large_vox_iter5.pt` (from scratch) | `avste.pt` (`checkpoint.finetune_from_model`) |
| Peak LR | 1e-4 | 3e-5 |
| Warmup / decay steps | 4000 / 35000 | 1000 / 15000 |
| Max epochs | 6 | 3 |
| Interference source | AudioSet speech/conversation clips, 1 per sample | Same-dataset LRS3 speakers, 1-4 per sample |

Both stages use the same 20/40/40 clean/non-speech-noise/speaker-interference
sampling ratio and a uniformly sampled SNR in [-10, 10] dB for noisy samples,
implemented as on-the-fly mixing in the data loader (`ONLINE_NOISE_MIX=1`,
see `av_hubert/avhubert/hubert_dataset.py`) rather than a pre-mixed static
dataset -- `scripts/train.sh` sets this up for you, you just need to point it
at your own manifests and noise pools.

### 4. Evaluation

Given noisy audio and lip video, the model predicts an enhanced semantic
(Mimi cb0) token sequence. We report two metrics:

- **Semantic-token accuracy** (primary): fraction of predicted cb0 tokens
  matching the tokens extracted from the corresponding clean audio.
- **WER** (secondary, diagnostic): the enhanced semantic tokens are combined
  with the original noisy audio's acoustic codebooks, decoded to a waveform
  with the Mimi decoder, transcribed with Whisper, and scored against the
  clean-audio ground-truth transcript.

`scripts/infer_avste.py` and `scripts/compare_demo.py` in this repo compute
exactly these metrics, but for one clip at a time -- reproducing the paper's
full Table 1 requires running this same protocol over the entire LRS3
`noisy_test` split (1321 clips × 5 SNRs × noise types), which is part of the
larger pipeline, not this release.

---

## Repository structure

```
av-ste/
├── av_hubert/
│   ├── avhubert/                # AV-HuBERT user_dir: task, data, and model
│   │   └── models/
│   │       └── avhubert_crossattn_ent.py   # AV-STE model (entropy-gated cross-attention)
│   └── fairseq/                 # bundled fairseq (install with pip install -e)
├── configs/                     # fairseq task / model configs
├── scripts/
│   ├── infer_avste.py           # single-clip inference
│   ├── compare_demo.py          # multi-condition comparison table
│   ├── prepare_lip_roi.py       # lip-ROI extraction from face video
│   ├── train.sh                 # train your own checkpoint (both stages)
│   └── download_checkpoint.sh
├── examples/
│   ├── run_demo.sh              # end-to-end demo script
│   └── sample_*.wav / *.mp4    # bundled LRS3 test clips
├── checkpoints/                 # place avste.pt + large_vox_iter5.pt here
├── requirements.txt
└── README.md
```

---

