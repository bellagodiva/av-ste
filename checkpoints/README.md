# Checkpoints

Three files go in this directory: two fine-tuned AV-STE models (`avste.pt`
and `avste_lrs3_interference.pt`) and `large_vox_iter5.pt`, the public
AV-HuBERT-Large VoxCeleb2 backbone both were fine-tuned from.

## Download

```bash
bash scripts/download_checkpoint.sh
```

Or download manually:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="bgdv99/av-ste", filename="avste.pt")
# same for filename="avste_lrs3_interference.pt"
```

```bash
cp "$path" checkpoints/avste.pt
```

`large_vox_iter5.pt` is the original AV-HuBERT-Large checkpoint pretrained on
LRS3 + VoxCeleb2 -- download it from the
[official AV-HuBERT model zoo](https://github.com/facebookresearch/av_hubert)
and place it at `checkpoints/large_vox_iter5.pt`.

## Checkpoint details

| Checkpoint | Training data | Description |
|------------|--------------|-------------|
| `avste.pt` | LRS3 433 h, with 20/40/40 clean/non-speech/speech-interference augmentation (AudioSet noise) | AV-STE (Soft-CA+NAM), 4-frame lookahead, entropy-gated cross-attention. Reproduces the paper's Clean / Non-speech (AudioSet) / Speaker (AudioSet) results (Table 1). |
| `avste_lrs3_interference.pt` | `avste.pt`, further trained with the target utterance mixed against 1-4 LRS3 interfering speakers with ratio 20/40/40 clean/non-speech/speech-interference | Reproduces the paper's same-dataset LRS3 speaker-interference row and the Seamless Interaction (out-of-domain) rows. |
| `large_vox_iter5.pt` | LRS3 + VoxCeleb2 | Public AV-HuBERT-Large backbone. **Only its architecture config is used** -- `infer_avste.py` builds the model skeleton from it, then immediately overwrites every weight with the AV-STE checkpoint's fine-tuned state dict. You do not need this file to be the "right" checkpoint in any deeper sense, just a valid AV-HuBERT-Large checkpoint file so the loader can read its config. |

All are fairseq model ensemble files, loaded via:

```python
from fairseq import checkpoint_utils
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    ["checkpoints/avste.pt"],  # or checkpoints/avste_lrs3_interference.pt
    arg_overrides={"w2v_path": "checkpoints/large_vox_iter5.pt"},
)
```

The `arg_overrides` step is required: each AV-STE checkpoint internally
remembers the absolute path of `large_vox_iter5.pt` on the machine it was
originally trained on, which won't exist on yours. `scripts/infer_avste.py`
handles this override automatically (default `--w2v_path
checkpoints/large_vox_iter5.pt`, override with a different path if you place
the file elsewhere).
