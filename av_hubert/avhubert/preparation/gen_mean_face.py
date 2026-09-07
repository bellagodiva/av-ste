import cv2
import numpy as np
import dlib
from pathlib import Path

predictor = dlib.shape_predictor('/mnt/hard1/bella/EMNLP26/source/av_hubert/avhubert/preparation/data/dlib_model/face_predictor/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

videos = list(Path('/mnt/hard1/bella/EMNLP26/dataset/LRS3/video/trainval').rglob('*.mp4'))[:50]
all_landmarks = []
for v in videos:
    cap = cv2.VideoCapture(str(v))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if not dets:
        continue
    shape = predictor(gray, dets[0])
    lm = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float64)
    all_landmarks.append(lm)

print(f'collected {len(all_landmarks)} frames')
mean_face = np.mean(all_landmarks, axis=0)
out = '/mnt/hard1/bella/EMNLP26/source/av_hubert/avhubert/preparation/data/dlib_model/mean_face/20words_mean_face.npy'
np.save(out, mean_face, allow_pickle=False)
print('saved, shape:', np.load(out).shape)
