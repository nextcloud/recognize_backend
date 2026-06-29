# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Video classification task processor.

Uses VideoMAE (Large) fine-tuned on Kinetics-400 (400 action classes)
via the Hugging Face ``transformers`` library. We follow VideoMAE's
default 16-frame uniform clip sampling.
"""

import os

import numpy as np

from ._device import torch_device

DEFAULT_MODEL = os.environ.get(
    "RECOGNIZE_VIDEO_MODEL", "MCG-NJU/videomae-large-finetuned-kinetics"
)
NUM_FRAMES = int(os.environ.get("RECOGNIZE_VIDEO_FRAMES", "16"))
TOP_K = int(os.environ.get("RECOGNIZE_VIDEO_TOP_K", "5"))
SCORE_THRESHOLD = float(os.environ.get("RECOGNIZE_VIDEO_THRESHOLD", "0.15"))

PROVIDER_ID = "recognize_backend:video:classification"
PROVIDER_NAME = "Recognize Video Classification (VideoMAE/Kinetics)"
EXPECTED_RUNTIME = 60


def _sample_indices(total_frames: int, num_samples: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        # Repeat last frame to pad
        idxs = list(range(total_frames))
        while len(idxs) < num_samples:
            idxs.append(total_frames - 1)
        return idxs
    step = total_frames / num_samples
    return [min(total_frames - 1, int(step * i + step / 2)) for i in range(num_samples)]


def _read_frames(path: str, num_samples: int) -> np.ndarray:
    import av

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames or 0
        if total == 0:
            # Some containers don't report frame count; decode once to count.
            total = sum(1 for _ in container.decode(video=0))
            container.close()
            container = av.open(path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
        indices = set(_sample_indices(total, num_samples))
        frames: dict[int, np.ndarray] = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames[i] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(indices):
                break
    finally:
        container.close()

    ordered = _sample_indices(max(total, num_samples), num_samples)
    last = None
    out = []
    for idx in ordered:
        arr = frames.get(idx)
        if arr is None:
            arr = last
        if arr is None and frames:
            arr = next(iter(frames.values()))
        if arr is not None:
            out.append(arr)
            last = arr
    if not out:
        raise RuntimeError("Failed to decode any video frames")
    while len(out) < num_samples:
        out.append(out[-1])
    return np.stack(out[:num_samples], axis=0)


def build():
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    device = torch_device()
    processor = VideoMAEImageProcessor.from_pretrained(DEFAULT_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(DEFAULT_MODEL)
    model.eval()
    model.to(device)
    id2label = model.config.id2label

    @torch.no_grad()
    def run(path: str) -> str:
        frames = _read_frames(path, NUM_FRAMES)
        # transformers expects a list of frames per video; pass list-of-arrays.
        inputs = processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))
        labels = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            if score < SCORE_THRESHOLD and labels:
                continue
            labels.append(id2label[idx])
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                uniq.append(label)
        return ", ".join(uniq)

    return run
