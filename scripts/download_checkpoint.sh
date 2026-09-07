#!/usr/bin/env bash
# Download the checkpoints AV-STE inference needs:
#   1. avste.pt                    -- our fine-tuned model, from Hugging Face Hub
#   2. avste_lrs3_interference.pt  -- avste.pt further trained on LRS3 speaker
#                                      interference; use this one for same-dataset
#                                      speaker interference / out-of-domain video
#                                      (see checkpoints/README.md)
#   3. large_vox_iter5.pt          -- the public AV-HuBERT-Large backbone both
#                                      were fine-tuned from (config only; weights
#                                      are overwritten at load time)
#
# Usage:
#   bash scripts/download_checkpoint.sh
#
# Alternatively, download a checkpoint manually:
#   pip install huggingface_hub
#   python -c "
#     from huggingface_hub import hf_hub_download
#     path = hf_hub_download(repo_id='YOUR_ORG/av-ste', filename='avste.pt')
#     print(path)
#   "
#   cp <path> checkpoints/avste.pt
#
# Google Drive mirror (backup):
#   https://drive.google.com/file/d/GDRIVE_FILE_ID/view
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$ROOT/checkpoints"

download_hf_checkpoint() {
    local filename="$1"
    local out="$ROOT/checkpoints/$filename"
    if [ -f "$out" ]; then
        echo "[SKIP] $filename already exists: $out"
        return 0
    fi
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub"
        exit 1
    fi
    echo "Downloading $filename from Hugging Face Hub …"
    FILENAME="$filename" OUT="$out" python - <<'PYEOF'
import os, shutil
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="YOUR_ORG/av-ste",   # TODO: update after HF upload
    filename=os.environ["FILENAME"],
    repo_type="model",
)
out = os.environ["OUT"]
os.makedirs(os.path.dirname(out), exist_ok=True)
shutil.copy(path, out)
print(f"Saved to {out}")
PYEOF
}

# ── 1 & 2. AV-STE checkpoints (Hugging Face Hub) ────────────────────────────────
download_hf_checkpoint "avste.pt"
download_hf_checkpoint "avste_lrs3_interference.pt"

# ── 3. large_vox_iter5.pt (official AV-HuBERT model zoo) ────────────────────────
W2V_OUT="$ROOT/checkpoints/large_vox_iter5.pt"
if [ -f "$W2V_OUT" ]; then
    echo "[SKIP] large_vox_iter5.pt already exists: $W2V_OUT"
else
    echo "Downloading large_vox_iter5.pt (AV-HuBERT-Large backbone) …"
    # Verified reachable (HTTP 200, 3.9GB) as of 2026-08-29.
    W2V_URL="https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt"
    if ! curl -fL -o "$W2V_OUT" "$W2V_URL"; then
        rm -f "$W2V_OUT"
        echo "[ERROR] Download failed. Get large_vox_iter5.pt manually from"
        echo "        the AV-HuBERT model zoo and place it at $W2V_OUT"
        echo "        https://github.com/facebookresearch/av_hubert"
        exit 1
    fi
fi

echo "Done. All checkpoints are in $ROOT/checkpoints/"
