#!/usr/bin/env python3
"""
Persistent MiMi encoder worker.
Reads [B,T] float32 arrays from stdin, writes [B,T_frames,D] float32 to stdout.
Wire format per message: 4-byte little-endian int32 ndim, ndim×4-byte shape, raw float32 data.
"""
import sys, struct, numpy as np

MIMI_STRIDE = 1920  # product of SEANet downsampling strides: 8×6×5×4

def main():
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    import torch

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda" if torch.cuda.is_available() else "cpu")
    mimi.eval()
    device = next(mimi.parameters()).device

    sys.stderr.write("READY\n")
    sys.stderr.flush()

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        header = stdin.read(4)
        if not header:
            break
        ndim  = struct.unpack('<i', header)[0]
        shape = struct.unpack(f'<{ndim}i', stdin.read(4 * ndim))
        data  = stdin.read(4 * int(np.prod(shape)))
        wav_np = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

        with torch.no_grad():
            wav = torch.from_numpy(wav_np).to(device).unsqueeze(1)  # [B, 1, T]

            # Pad T to a multiple of MIMI_STRIDE so SEANet stride assertion passes
            T = wav.shape[-1]
            pad = (MIMI_STRIDE - T % MIMI_STRIDE) % MIMI_STRIDE
            if pad > 0:
                wav = torch.nn.functional.pad(wav, (0, pad))

            preq = mimi.encoder(wav)      # [B, D, T_frames]
            preq = preq.transpose(1, 2)   # [B, T_frames, D]
            preq_np = preq.cpu().float().numpy()

        # Send result
        stdout.write(struct.pack('<i', preq_np.ndim))
        stdout.write(struct.pack(f'<{preq_np.ndim}i', *preq_np.shape))
        stdout.write(preq_np.tobytes())
        stdout.flush()

if __name__ == "__main__":
    main()
