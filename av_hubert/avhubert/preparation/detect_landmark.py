# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, pickle, math, csv
import cv2, dlib
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Quality thresholds — samples that fail any check are skipped entirely
# (no .pkl written) so align_mouth.py will naturally ignore them too.
# ---------------------------------------------------------------------------
MIN_LANDMARK_RATIO = 0.80   # fraction of frames that must have landmarks
MIN_SHARPNESS      = 20.0   # variance-of-Laplacian on full frame (blur check)
MIN_BRIGHTNESS     = 30.0   # mean pixel value (too-dark check)
MAX_BRIGHTNESS     = 235.0  # mean pixel value (over-exposed check)


def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_frame_quality(frames):
    """
    Returns (mean_sharpness, mean_brightness) over all frames.
    frames: list of RGB uint8 ndarrays
    """
    sharpness_vals, brightness_vals = [], []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sharpness_vals.append(variance_of_laplacian(gray))
        brightness_vals.append(float(gray.mean()))
    sharpness  = float(np.mean(sharpness_vals))  if sharpness_vals  else 0.0
    brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0
    return sharpness, brightness


def detect_face_landmarks(face_predictor_path, cnn_detector_path, root_dir,
                           landmark_dir, flist_fn, rank, nshard,
                           rejected_log=None):

    def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    detector    = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)
    predictor   = dlib.shape_predictor(face_predictor_path)
    input_dir   = root_dir
    output_dir  = landmark_dir

    fids = [ln.strip() for ln in open(flist_fn).readlines()]
    num_per_shard = math.ceil(len(fids) / nshard)
    start_id, end_id = num_per_shard * rank, num_per_shard * (rank + 1)
    fids = fids[start_id:end_id]
    print(f"{len(fids)} files in this shard")

    rejected_rows = []
    n_kept = 0

    for fid in tqdm(reversed(fids)):
        video_path = os.path.join(input_dir, fid + '.mp4')
        output_fn  = os.path.join(output_dir, fid + '.pkl')

        if os.path.exists(output_fn):
            n_kept += 1
            continue

        frames = load_video(video_path)
        if len(frames) == 0:
            rejected_rows.append({"fid": fid, "reason": "empty_video",
                                   "sharpness": 0, "brightness": 0, "lm_ratio": 0})
            continue

        # --- quality checks on raw frames ---
        sharpness, brightness = score_frame_quality(frames)

        if sharpness < MIN_SHARPNESS:
            rejected_rows.append({"fid": fid, "reason": "blurry",
                                   "sharpness": round(sharpness, 2),
                                   "brightness": round(brightness, 2), "lm_ratio": 0})
            continue

        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            rejected_rows.append({"fid": fid, "reason": "bad_lighting",
                                   "sharpness": round(sharpness, 2),
                                   "brightness": round(brightness, 2), "lm_ratio": 0})
            continue

        # --- landmark detection ---
        landmarks = []
        for frame in frames:
            landmark = detect_landmark(frame, detector, cnn_detector, predictor)
            landmarks.append(landmark)

        lm_ratio = sum(1 for lm in landmarks if lm is not None) / len(landmarks)

        if lm_ratio < MIN_LANDMARK_RATIO:
            rejected_rows.append({"fid": fid, "reason": "low_landmark_ratio",
                                   "sharpness": round(sharpness, 2),
                                   "brightness": round(brightness, 2),
                                   "lm_ratio": round(lm_ratio, 4)})
            continue

        # --- passed all checks: save landmark pkl ---
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        pickle.dump(landmarks, open(output_fn, 'wb'))
        n_kept += 1

    print(f"\nKept {n_kept}/{len(fids)} samples "
          f"({100*n_kept/max(len(fids),1):.1f}%)")
    print(f"Rejected {len(rejected_rows)} samples")

    # write rejection log
    if rejected_rows:
        log_path = rejected_log or os.path.join(output_dir, f"rejected_rank{rank}.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fid", "reason", "sharpness",
                                               "brightness", "lm_ratio"])
            w.writeheader()
            w.writerows(rejected_rows)
        print(f"Rejection log → {log_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Detect facial landmarks with quality filtering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root',           type=str, help='root dir of input videos')
    parser.add_argument('--landmark',       type=str, help='output landmark dir')
    parser.add_argument('--manifest',       type=str, help='file list (one fid per line)')
    parser.add_argument('--cnn_detector',   type=str, help='path to mmod_human_face_detector.dat')
    parser.add_argument('--face_predictor', type=str, help='path to shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--rank',           type=int, default=0)
    parser.add_argument('--nshard',         type=int, default=1)
    parser.add_argument('--ffmpeg',         type=str, help='ffmpeg binary path')
    parser.add_argument('--rejected-log',   type=str, default=None,
                        help='path for CSV of rejected samples (default: landmark_dir/rejected_rankN.csv)')
    parser.add_argument('--min-landmark-ratio', type=float, default=MIN_LANDMARK_RATIO)
    parser.add_argument('--min-sharpness',      type=float, default=MIN_SHARPNESS)
    parser.add_argument('--min-brightness',     type=float, default=MIN_BRIGHTNESS)
    parser.add_argument('--max-brightness',     type=float, default=MAX_BRIGHTNESS)
    args = parser.parse_args()

    # allow threshold overrides from CLI
    MIN_LANDMARK_RATIO = args.min_landmark_ratio
    MIN_SHARPNESS      = args.min_sharpness
    MIN_BRIGHTNESS     = args.min_brightness
    MAX_BRIGHTNESS     = args.max_brightness

    import skvideo
    skvideo.setFFmpegPath(os.path.dirname(args.ffmpeg))
    import skvideo.io

    detect_face_landmarks(
        args.face_predictor, args.cnn_detector,
        args.root, args.landmark, args.manifest,
        args.rank, args.nshard,
        rejected_log=args.rejected_log,
    )
