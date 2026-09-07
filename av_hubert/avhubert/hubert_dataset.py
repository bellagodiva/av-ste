# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile

DBG=True if len(sys.argv) == 1 else False

if DBG:
    import utils as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
else:
    from . import utils as custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    def is_audio_label_aligned(audio_dur, label_durs):
        return True
        #return all([abs(audio_dur - label_dur)<tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, frame_sizes = [], [], []
    dur_from_label_list = []
    is_seq_label = any([x==-1 for x in label_rates])
    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split())/label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            frame_sz = int(items[-2]) # video frame length
            if min_keep is not None and frame_sz < min_keep:
                n_short += 1
            elif max_keep is not None and frame_sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(frame_sz/frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                names.append((video_path, audio_path+':'+audio_id))
                inds.append(ind)
                frame_sizes.append(frame_sz)
            #print(f"PROCESSED {ind+1} lines, loaded {len(names)} samples, skipped {n_short} short, {n_long} long and {n_unaligned} unaligned samples")
    tot = ind + 1
    
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
            f"longest-loaded={max(frame_sizes)}, shortest-loaded={min(frame_sizes)}"
        )
    )
    return root, names, inds, tot, frame_sizes

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return
    audio_rate = 25
    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
        
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )

class MimiLabelProcessor:
    def __call__(self, label):
        label = label.strip()
        if len(label) == 0:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor([int(x) for x in label.split()], dtype=torch.long)

class AVHubertDataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1
    ):
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(manifest_path, max_keep_sample_size, min_keep_sample_size, frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        # One MimiLabelProcessor per label stream (direct int parsing, no dict lookup).
        # The `label_processors` argument from the task is intentionally ignored here
        # because fairseq LabelEncoders add special-token offsets that are wrong for
        # raw Mimi integer tokens.
        self.label_processors = [MimiLabelProcessor() for _ in range(self.num_labels)]
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert len(self.label_processors) == self.num_labels
        if not skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        # optional: pre-extracted noisy Mimi logits for residual corrector training
        # set via task config or environment variable NOISY_LOGITS_ROOT
        self.noisy_logits_root  = os.environ.get("NOISY_LOGITS_ROOT",  None)
        self.teacher_logits_root = os.environ.get("TEACHER_LOGITS_ROOT", None)

        # optional: pre-extracted ArcFace speaker embeddings [512] float32
        # set via environment variable ARCFACE_EMBED_ROOT pointing to dir of {utt_id}.npy
        self.arcface_embed_root = os.environ.get("ARCFACE_EMBED_ROOT", None)

        # optional: pre-extracted AV-HuBERT visual-only features [T, D] float32
        # extracted by running AV-HuBERT with audio zeroed (process_dataset/extract_visual_features.py)
        # used by the AV sync predictor loss (use_av_sync=true in criterion)
        self.visual_feats_root = os.environ.get("VISUAL_FEATS_ROOT", None)

        # filter out samples missing speaker embeddings when ARCFACE_EMBED_ROOT is set
        if self.arcface_embed_root is not None:
            keep_mask = [
                os.path.exists(os.path.join(self.arcface_embed_root, name[1].split(':')[1] + ".npy"))
                for name in self.names
            ]
            n_before = len(self.names)
            self.names  = [n for n, k in zip(self.names,  keep_mask) if k]
            self.sizes  = [s for s, k in zip(self.sizes,  keep_mask) if k]
            if self.store_labels:
                self.label_list = [
                    [lbl for lbl, k in zip(lbls, keep_mask) if k]
                    for lbls in self.label_list
                ]
            logger.info(f"Filtered {n_before - len(self.names)} samples missing speaker embeddings; {len(self.names)} remaining")

        # optional: noisy Mimi tokens (.mimi text file, 12.5 Hz) for cross-attention model
        # set via environment variable NOISY_TOKENS_MIMI pointing to the .mimi file
        self.noisy_tokens_lines = None
        noisy_tokens_mimi = os.environ.get("NOISY_TOKENS_MIMI", None)
        if noisy_tokens_mimi and os.path.exists(noisy_tokens_mimi):
            with open(noisy_tokens_mimi, "r", encoding="utf-8") as f:
                self.noisy_tokens_lines = [l.rstrip("\n") for l in f]
            logger.info(f"Loaded noisy tokens from {noisy_tokens_mimi} ({len(self.noisy_tokens_lines)} lines)")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        #print("RAW LABEL:", repr(label[:200]) if isinstance(label, str) else label)

        if self.label_processors is not None:
            processed = self.label_processors[label_idx](label)
            label = processed

        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            audio_fn = audio_fn.split(':')[0]
            sample_rate, wav_data = wavfile.read(audio_fn)
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            if np.random.rand() < self.noise_prob:
                wav_data = self.add_noise(wav_data)
            audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
            audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(wavfile.read(self.noise_wav[x])[1].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed

    @staticmethod
    def _noisy_logit_key(audio_fn: str, fid: str) -> str:
        """Return the .npy lookup key for noisy logits.

        Seamless AVSR rows have audio paths like:
          .../seamless_noisy_audio_AVSR/train/train_X.wav   → avsr/train/train_X
          .../seamless_noisy_audio_AVSR_2/train/train_X.wav → avsr_2/train/train_X
        All other rows (LRS3, clean seamless) fall back to fid unchanged.
        """
        from pathlib import Path
        p = Path(audio_fn)
        for part in p.parts:
            if part.startswith("seamless_noisy_audio_AVSR"):
                tag = part.replace("seamless_noisy_audio_", "").lower()  # avsr / avsr_2
                return f"{tag}/{p.parent.name}/{p.stem}"
        return fid

    def _load_clean_audio_24k(self, audio_fn: str) -> torch.Tensor:
        """
        Load the clean counterpart of a (possibly noisy) audio file and
        resample to 24 kHz for MiMi.  Returns float32 tensor [T].
        """
        # Derive clean path: replace noisy_audio → audio in the file path
        clean_fn = audio_fn.replace("/noisy_audio/", "/audio/")
        if not os.path.exists(clean_fn):
            clean_fn = audio_fn  # fallback: use whatever we have
        sample_rate, wav_data = wavfile.read(clean_fn)
        wav = torch.from_numpy(wav_data.astype(np.float32))
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        # normalise int16 range to [-1, 1]
        if wav_data.dtype == np.int16:
            wav = wav / 32768.0
        # resample 16 kHz → 24 kHz for MiMi
        if sample_rate != 24_000:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sample_rate, 24_000)
        return wav  # [T_24k]

    def __getitem__(self, index):
        video_feats, audio_feats = self.load_feature(self.names[index])
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if self.normalize and 'audio' in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        labels = self.get_labels(index)
        fid = self.names[index][1].split(':')[1]

        # Load clean audio at 24 kHz for MiMi regression target
        # Skip if SKIP_CLEAN_AUDIO=1 (e.g. crossattn model uses CE only)
        audio_fn = self.names[index][1].split(':')[0]
        if os.environ.get("SKIP_CLEAN_AUDIO", "0") == "1":
            clean_audio = None
        else:
            try:
                clean_audio = self._load_clean_audio_24k(audio_fn)
            except Exception as e:
                logger.warning(f"[WARN] _load_clean_audio_24k failed for {audio_fn}: {e}")
                clean_audio = None

        # load pre-extracted noisy Mimi logits [T_12hz, 2048] if available
        noisy_logits = None
        if self.noisy_logits_root is not None:
            logit_key = self._noisy_logit_key(audio_fn, fid)
            logit_path = os.path.join(self.noisy_logits_root, logit_key + ".npy")
            if not os.path.exists(logit_path):
                raise FileNotFoundError(
                    f"NOISY_LOGITS_ROOT is set but logit file not found: {logit_path}\n"
                    f"Check that NOISY_LOGITS_ROOT={self.noisy_logits_root} is correct."
                )
            try:
                noisy_logits = torch.from_numpy(
                    np.load(logit_path).astype(np.float32)
                )  # [T_12hz, 2048]
            except Exception as e:
                raise RuntimeError(f"Failed to load logit file {logit_path}: {e}")

        # load pre-extracted ArcFace speaker embedding [512] if available
        arcface_embed = None
        if self.arcface_embed_root is not None:
            embed_path = os.path.join(self.arcface_embed_root, fid + ".npy")
            if os.path.exists(embed_path):
                try:
                    arcface_embed = torch.from_numpy(
                        np.load(embed_path).astype(np.float32)
                    )  # [512]
                except Exception:
                    pass

        # load noisy Mimi tokens from .mimi file [T_12hz] if available
        noisy_tokens = None
        if self.noisy_tokens_lines is not None and index < len(self.noisy_tokens_lines):
            line = self.noisy_tokens_lines[index]
            if line.strip():
                noisy_tokens = torch.tensor(
                    [int(x) for x in line.split()], dtype=torch.long
                )  # [T_12hz]

        # load pre-extracted teacher (lookahead=4) logits [T_12hz, 2048] if available
        teacher_logits = None
        if self.teacher_logits_root is not None:
            teacher_path = os.path.join(self.teacher_logits_root, fid + ".npy")
            if not os.path.exists(teacher_path):
                import warnings
                warnings.warn(
                    f"Teacher logit file not found (skipping): {teacher_path}",
                    RuntimeWarning,
                )
            else:
                try:
                    teacher_logits = torch.from_numpy(
                        np.load(teacher_path).astype(np.float32)
                    )  # [T_12hz, 2048]
                except Exception as e:
                    raise RuntimeError(f"Failed to load teacher logit file {teacher_path}: {e}")

        # load pre-extracted visual-only features [T_12hz, D] if available
        visual_feats = None
        if self.visual_feats_root is not None:
            vis_path = os.path.join(self.visual_feats_root, fid + ".npy")
            if os.path.exists(vis_path):
                try:
                    visual_feats = torch.from_numpy(
                        np.load(vis_path).astype(np.float32)
                    )  # [T_12hz, D]
                except Exception as e:
                    raise RuntimeError(f"Failed to load visual feat file {vis_path}: {e}")

        return {"id": index, 'fid': fid, "audio_fn": audio_fn,
                "video_source": video_feats, 'audio_source': audio_feats,
                "label_list": labels, "clean_audio": clean_audio, "noisy_logits": noisy_logits,
                "noisy_tokens": noisy_tokens, "arcface_embed": arcface_embed,
                "teacher_logits": teacher_logits, "visual_feats": visual_feats}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, audio_size, audio_starts)
        else:
            collated_videos = None
        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )
        source = {"audio": collated_audios, "video": collated_videos}
        net_input = {"source": source, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples],
            "audio_path": [s['audio_fn'] for s in samples],
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            if self.is_s2s:
                batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
            else:
                batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list

        # Pad and stack clean audio tensors for MiMi regression target [B, T_24k]
        clean_wavs = [s.get("clean_audio") for s in samples]
        if any(w is not None for w in clean_wavs):
            max_len = max(w.size(0) for w in clean_wavs if w is not None)
            clean_audio = torch.zeros(len(clean_wavs), max_len)
            for i, w in enumerate(clean_wavs):
                if w is not None:
                    clean_audio[i, :w.size(0)] = w
            batch["clean_audio"] = clean_audio

        # Pad and stack noisy Mimi logits [B, T_25hz, 2048] for residual corrector
        if samples[0].get("noisy_logits") is not None:
            logits_list = [s["noisy_logits"] for s in samples]
            # upsample 12.5 Hz → 25 Hz by repeat to match label rate
            logits_list = [l.repeat_interleave(2, dim=0) for l in logits_list]
            max_t = max(l.size(0) for l in logits_list)
            V = logits_list[0].size(1)
            noisy_logits = torch.zeros(len(logits_list), max_t, V)
            for i, l in enumerate(logits_list):
                noisy_logits[i, :l.size(0)] = l
            net_input["noisy_logits"] = noisy_logits  # [B, T_25hz, 2048]
            # discrete noisy tokens for cross-attention model [B, T_25hz]
            net_input["source"]["noisy_tokens"] = noisy_logits.argmax(dim=-1)

        # Pad and stack teacher logits [B, T_25hz, 2048] for KD loss
        if samples[0].get("teacher_logits") is not None:
            tl_list = [s["teacher_logits"] for s in samples]
            # upsample 12.5 Hz → 25 Hz by repeat to match label rate
            tl_list = [l.repeat_interleave(2, dim=0) for l in tl_list]
            max_t = max(l.size(0) for l in tl_list)
            V = tl_list[0].size(1)
            teacher_logits_batch = torch.zeros(len(tl_list), max_t, V)
            for i, l in enumerate(tl_list):
                teacher_logits_batch[i, :l.size(0)] = l
            batch["teacher_logits"] = teacher_logits_batch  # [B, T_25hz, 2048]

        # Pad and stack noisy Mimi tokens from .mimi file [B, T_25hz] for cross-attention model
        if samples[0].get("noisy_tokens") is not None:
            tok_list = [s["noisy_tokens"] for s in samples]
            # upsample 12.5 Hz → 25 Hz by repeat to match label rate
            tok_list = [t.repeat_interleave(2) for t in tok_list]
            max_t = max(t.size(0) for t in tok_list)
            noisy_tokens = torch.zeros(len(tok_list), max_t, dtype=torch.long)
            for i, t in enumerate(tok_list):
                noisy_tokens[i, :t.size(0)] = t
            net_input["source"]["noisy_tokens"] = noisy_tokens  # [B, T_25hz]

        # Stack ArcFace speaker embeddings [B, 512] if available
        arcface_embeds = [s.get("arcface_embed") for s in samples]
        if any(e is not None for e in arcface_embeds):
            emb_dim = next(e for e in arcface_embeds if e is not None).size(0)
            speaker_embed = torch.zeros(len(arcface_embeds), emb_dim)
            for i, e in enumerate(arcface_embeds):
                if e is not None:
                    speaker_embed[i] = e
            net_input["speaker_embed"] = speaker_embed  # [B, 512]

        # Stack visual-only features [B, T_12hz, D] if available
        vis_list = [s.get("visual_feats") for s in samples]
        if any(v is not None for v in vis_list):
            D_vis   = next(v for v in vis_list if v is not None).size(1)
            max_T   = max((v.size(0) if v is not None else 0) for v in vis_list)
            vis_pad = torch.zeros(len(vis_list), max_T, D_vis)
            for i, v in enumerate(vis_list):
                if v is not None:
                    vis_pad[i, :v.size(0)] = v
            net_input["visual_feats"] = vis_pad   # [B, T_12hz, D]

        return batch

    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False) # 
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / 25 # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label_s2s(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = self.label_processors[0].dictionary.pad(), self.label_processors[0].dictionary.eos()
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
        prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
        return (targets_, prev_output_tokens), lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                if self.is_s2s:
                    targets, lengths, ntokens = self.collater_seq_label_s2s(targets, pad)
                else:
                    targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]
