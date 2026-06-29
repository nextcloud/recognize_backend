# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image classification task processor.

Uses ConvNeXt V2 (Large, 384, pretrained on ImageNet-22k) which gives
fine-grained labels across ~21k WordNet synsets via the Hugging Face
``transformers`` image-classification pipeline.
"""

import os

from ._device import torch_device

DEFAULT_MODEL = os.environ.get(
    "RECOGNIZE_IMAGE_MODEL", "facebook/convnextv2-large-22k-384"
)
TOP_K = int(os.environ.get("RECOGNIZE_IMAGE_TOP_K", "50"))
SCORE_THRESHOLD = float(os.environ.get("RECOGNIZE_IMAGE_THRESHOLD", "0.35"))

PROVIDER_ID = "recognize_backend:image:classification"
PROVIDER_NAME = "Recognize Image Classification (ConvNeXt V2)"
EXPECTED_RUNTIME = 15


def _clean_label(label: str) -> str:
    # ImageNet labels are often a comma-separated synonym list, e.g.
    # "tabby, tabby cat". Keep only the first, most-specific synonym.
    return label.split(",", 1)[0].strip()


def build():
    from PIL import Image
    from transformers import pipeline

    device = torch_device()
    classifier = pipeline(
        task="image-classification",
        model=DEFAULT_MODEL,
        device=0 if device == "cuda" else -1,
    )

    def run(path: str) -> str:
        with Image.open(path) as img:
            img = img.convert("RGB")
            predictions = classifier(img, top_k=TOP_K)
        labels = [
            _clean_label(p["label"])
            for p in predictions
            if p.get("score", 0.0) >= SCORE_THRESHOLD
        ]
        if not labels and predictions:
            labels = [_clean_label(predictions[0]["label"])]
        # Deduplicate, preserve order
        seen = set()
        uniq = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                uniq.append(label)
        return ", ".join(uniq)

    return run
