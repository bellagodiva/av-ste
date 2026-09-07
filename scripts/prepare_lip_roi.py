#!/usr/bin/env python
"""
Extract a 96×96 grayscale mouth-ROI MP4 from a face video.

Requires:
  pip install opencv-python dlib

Download the dlib face landmark model (68-point):
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Usage:
  python scripts/prepare_lip_roi.py \\
      --input  face_video.mp4 \\
      --output lip_roi.mp4 \\
      --landmarks shape_predictor_68_face_landmarks.dat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


MOUTH_POINTS = list(range(48, 68))   # dlib 68-point landmarks, mouth region
ROI_SIZE     = 96                    # output frame size in pixels
FPS_OUT      = 25                    # target frame rate


def crop_mouth(frame: np.ndarray, shape, margin: float = 0.6) -> np.ndarray | None:
    """Return a square ROI centred on the mouth, resized to ROI_SIZE×ROI_SIZE."""
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in MOUTH_POINTS])
    cx, cy = pts.mean(axis=0).astype(int)
    w = int((pts[:, 0].max() - pts[:, 0].min()) * (1 + margin))
    h = int((pts[:, 1].max() - pts[:, 1].min()) * (1 + margin))
    side = max(w, h)
    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(cx + side // 2, frame.shape[1])
    y2 = min(cy + side // 2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def main():
    p = argparse.ArgumentParser(description="Extract mouth-ROI MP4 from face video")
    p.add_argument("--input",     required=True, help="input face video (.mp4 / .avi / ...)")
    p.add_argument("--output",    required=True, help="output mouth-ROI MP4")
    p.add_argument("--landmarks", default="shape_predictor_68_face_landmarks.dat",
                   help="path to dlib 68-point landmark model")
    args = p.parse_args()

    if not Path(args.landmarks).exists():
        print(f"[ERROR] Landmark model not found: {args.landmarks}")
        print("  Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    try:
        import dlib
    except ImportError:
        print("[ERROR] dlib not installed. Run: pip install dlib")
        sys.exit(1)

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.landmarks)

    cap = cv2.VideoCapture(args.input)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or FPS_OUT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, FPS_OUT,
                          (ROI_SIZE, ROI_SIZE), isColor=False)

    n_written = 0
    n_skipped = 0
    prev_roi  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        roi   = None

        if faces:
            shape = predictor(gray, faces[0])
            roi   = crop_mouth(frame, shape)

        if roi is None:
            roi = prev_roi if prev_roi is not None else np.zeros(
                (ROI_SIZE, ROI_SIZE), dtype=np.uint8)
            n_skipped += 1

        out.write(roi)
        prev_roi = roi
        n_written += 1

    cap.release()
    out.release()

    if n_skipped:
        print(f"[WARN] {n_skipped}/{n_written} frames had no face detection; "
              "filled with previous frame.")
    print(f"Done. {n_written} frames → {args.output}")


if __name__ == "__main__":
    main()
