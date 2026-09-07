#!/usr/bin/env bash
# Train your own AV-STE checkpoint(s), reproducing the paper's two-stage recipe:
#   Stage 1 (STAGE=1): avste.pt          -- fine-tune from AV-HuBERT-Large on
#                                            LRS3 with 20/40/40 clean/AudioSet-
#                                            non-speech/AudioSet-speaker-
#                                            interference augmentation.
#   Stage 2 (STAGE=2): avste_lrs3_interference.pt -- continue Stage 1 with the
#                                            AudioSet interference branch
#                                            replaced by 1-4 same-dataset LRS3
#                                            interferers.
#
# You need, before running this:
#   1. checkpoints/large_vox_iter5.pt          (see checkpoints/README.md)
#   2. An LRS3 433h fairseq manifest at $DATA: {train,valid}.tsv + labels/
#      {train,valid}.mimi (Mimi cb0 token labels), in the standard AV-HuBERT
#      TSV format -- see av_hubert/avhubert/preparation/ for manifest tooling.
#   3. Pre-extracted noisy Mimi logits per utterance under $NOISY_LOGITS_ROOT
#      (the cross-attention fuser's key/value input; required at train and
#      inference time regardless of clean/noisy mixing -- see the paper's
#      Section on Soft Token Cross-Attention).
#   4. For Stage 1: BG_NOISE_ROOT (a directory of AudioSet non-speech .wav
#      clips) and INTERFERER_POOL_TSV (an AudioSet speech/conversation clip
#      manifest, TSV column 3 = audio path).
#   5. For Stage 2: BG_NOISE_ROOT (same as Stage 1) and INTERFERER_POOL_TSV
#      pointing instead at your LRS3 train manifest (so interferers are drawn
#      from LRS3 itself).
#
# None of these are bundled in this repo -- see "Reproducing the paper's
# experiments" in the main README for what each one is and how it's built.
#
# Usage:
#   STAGE=1 DATA=/path/to/lrs3_433h NOISY_LOGITS_ROOT=/path/to/logits \
#     BG_NOISE_ROOT=/path/to/audioset_nonspeech \
#     INTERFERER_POOL_TSV=/path/to/audioset_speech_manifest.tsv \
#     SAVE_DIR=experiment/avste_stage1 \
#     bash scripts/train.sh
#
#   STAGE=2 DATA=/path/to/lrs3_433h NOISY_LOGITS_ROOT=/path/to/logits \
#     BG_NOISE_ROOT=/path/to/audioset_nonspeech \
#     INTERFERER_POOL_TSV=/path/to/lrs3_train_manifest.tsv \
#     FINETUNE_FROM=experiment/avste_stage1/checkpoints/checkpoint_best.pt \
#     SAVE_DIR=experiment/avste_stage2 \
#     bash scripts/train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STAGE="${STAGE:?Set STAGE=1 or STAGE=2}"
DATA="${DATA:?Set DATA to your LRS3 433h manifest directory}"
NOISY_LOGITS_ROOT="${NOISY_LOGITS_ROOT:?Set NOISY_LOGITS_ROOT to your pre-extracted noisy-logits directory}"
BG_NOISE_ROOT="${BG_NOISE_ROOT:?Set BG_NOISE_ROOT to a directory of AudioSet non-speech .wav clips}"
INTERFERER_POOL_TSV="${INTERFERER_POOL_TSV:?Set INTERFERER_POOL_TSV (see usage comment above)}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR, e.g. experiment/avste_stage${STAGE}}"
W2V_PATH="${W2V_PATH:-$ROOT/checkpoints/large_vox_iter5.pt}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_DIR="$ROOT/configs"
CONFIG_NAME="large_lrs3_433h_crossattn_ent"
USER_DIR="$ROOT/av_hubert/avhubert"

mkdir -p "$SAVE_DIR"

case "$STAGE" in
  1)
    # Stage 1: from-scratch fine-tune from the AV-HuBERT-Large backbone.
    # 20/40/40 clean/non-speech/speaker-interference, single AudioSet
    # interferer per noisy sample (matching the paper's "speech mixed with
    # AudioSet conversational speech" -- not the 1-4-interferer setting,
    # that's Stage 2's LRS3-specific augmentation).
    EXTRA_OVERRIDES=()
    export N_INTERFERERS_MIN=1
    export N_INTERFERERS_MAX=1
    ;;
  2)
    # Stage 2: continue from Stage 1, same ratio, AudioSet interference
    # branch replaced by 1-4 same-dataset LRS3 interferers.
    FINETUNE_FROM="${FINETUNE_FROM:?Set FINETUNE_FROM to your Stage-1 checkpoint}"
    EXTRA_OVERRIDES=("checkpoint.finetune_from_model=$FINETUNE_FROM")
    export N_INTERFERERS_MIN=1
    export N_INTERFERERS_MAX=4
    ;;
  *)
    echo "[ERROR] STAGE must be 1 or 2, got: $STAGE" >&2
    exit 1
    ;;
esac

export ONLINE_NOISE_MIX=1
export MIX_PROB_CLEAN=0.20
export MIX_PROB_BG=0.40          # remaining 0.40 goes to speaker interference
export MIX_SNR_MIN=-10.0
export MIX_SNR_MAX=10.0
export BG_NOISE_ROOT
export INTERFERER_POOL_TSV
export NOISY_LOGITS_ROOT

echo "=== AV-STE training, Stage $STAGE ==="
echo "    data:                $DATA"
echo "    save_dir:            $SAVE_DIR"
echo "    w2v_path (backbone): $W2V_PATH"
echo "    interferers per sample: $N_INTERFERERS_MIN-$N_INTERFERERS_MAX"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" fairseq-hydra-train \
    --config-dir "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    common.user_dir="$USER_DIR" \
    task.data="$DATA" \
    task.label_dir="$DATA" \
    model.w2v_path="$W2V_PATH" \
    hydra.run.dir="$SAVE_DIR" \
    "${EXTRA_OVERRIDES[@]}"

echo ""
echo "Done. Checkpoint at $SAVE_DIR/checkpoints/checkpoint_best.pt"
