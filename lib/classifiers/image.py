# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image classification task processor.

Uses ConvNeXt V2 (Large, 384, pretrained on ImageNet-22k) which gives
fine-grained labels across ~21k WordNet synsets via the Hugging Face
``transformers`` image-classification pipeline.
"""

import os

from ._device import batch_size_for, torch_device
from ._processor import Processor

DEFAULT_MODEL = os.environ.get(
    "RECOGNIZE_IMAGE_MODEL", "facebook/convnextv2-large-22k-384"
)
TOP_K = int(os.environ.get("RECOGNIZE_IMAGE_TOP_K", "50"))
SCORE_THRESHOLD = float(os.environ.get("RECOGNIZE_IMAGE_THRESHOLD", "0.5"))
# Measured ~96 MiB of activations per 384px image for ConvNeXt V2 Large (fp32),
# plus headroom for cuDNN workspaces.
VRAM_PER_IMAGE = 128 * 1024 ** 2
MAX_BATCH_SIZE = 64

PROVIDER_ID = "recognize_backend:image:classification"
PROVIDER_NAME = "Recognize Image Classification (ConvNeXt V2)"
EXPECTED_RUNTIME = 15


def _clean_label(label: str) -> str:
    # ImageNet labels are often a comma-separated synonym list, e.g.
    # "tabby, tabby cat". Keep only the first, most-specific synonym.
    return label.split(",", 1)[0].strip()


def build() -> Processor:
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = torch_device()
    image_processor = AutoImageProcessor.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForImageClassification.from_pretrained(DEFAULT_MODEL)
    model.eval()
    model.to(device)
    id2label = model.config.id2label
    input_size = image_processor.size.get("shortest_edge") or image_processor.size.get("height", 384)

    def load(path: str):
        with Image.open(path) as img:
            # Decode JPEGs at a reduced DCT scale; we only need about twice the
            # model's input size for a clean downscale. No-op for other formats.
            img.draft("RGB", (2 * input_size, 2 * input_size))
            # Honor the EXIF orientation tag so rotated photos are
            # classified upright rather than sideways.
            img = ImageOps.exif_transpose(img).convert("RGB")
            return image_processor(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def infer(items: list) -> list[str]:
        logits = model(pixel_values=torch.stack(items).to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))
        return [
            _labels(scores, indices)
            for scores, indices in zip(top.values.tolist(), top.indices.tolist())
        ]

    def _labels(scores: list[float], indices: list[int]) -> str:
        labels = [
            _clean_label(id2label[idx])
            for score, idx in zip(scores, indices)
            if score >= SCORE_THRESHOLD
        ]
        if not labels and indices:
            labels = [_clean_label(id2label[indices[0]])]
        # Deduplicate, preserve order
        seen = set()
        uniq = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                uniq.append(label)
        return ", ".join(uniq)

    return Processor(
        load=load,
        infer=infer,
        batch_size=batch_size_for(VRAM_PER_IMAGE, MAX_BATCH_SIZE),
    )
