# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image classification task processor.

Uses ConvNeXt V2 (Large, 384, pretrained on ImageNet-22k, fine-tuned on
ImageNet-1k) via Hugging Face ``transformers``.

Raw ImageNet classes are mapped to tags through :file:`imagenet_rules.yml`
(the rule set the original recognize classifier used): each class gives a
label and categories once its probability reaches the rule's threshold, and
classes without a rule are dropped. Since those thresholds were tuned for
recognize's EfficientNetV2, logits are divided by :data:`TEMPERATURE` first
so that ConvNeXt's probabilities are on a comparable scale.

Models without ImageNet-1k labels fall back to emitting raw labels above
:data:`SCORE_THRESHOLD`.
"""

import logging
import math
import os

from ._device import batch_size_for, torch_device
from ._processor import Processor

CALIBRATED_MODEL = "facebook/convnextv2-large-22k-384"
# Fitted by maximizing tag agreement (F1) with recognize's EfficientNetV2 on a
# sample of ~600 photos from a real library; 1.0 gave noticeably fewer tags (25% untagged
# vs. 21% for EfficientNetV2), 0.8 gives 14%.
CALIBRATED_TEMPERATURE = 0.8

DEFAULT_MODEL = os.environ.get("RECOGNIZE_IMAGE_MODEL", CALIBRATED_MODEL)
TEMPERATURE = float(
    os.environ.get(
        "RECOGNIZE_IMAGE_TEMPERATURE",
        CALIBRATED_TEMPERATURE if DEFAULT_MODEL == CALIBRATED_MODEL else 1.0,
    )
)
# Number of top predictions considered, as in the original recognize classifier.
TOP_K = int(os.environ.get("RECOGNIZE_IMAGE_TOP_K", "7"))
# Only used for models whose labels are not covered by the rules.
SCORE_THRESHOLD = float(os.environ.get("RECOGNIZE_IMAGE_THRESHOLD", "0.5"))
RULES_PATH = os.path.join(os.path.dirname(__file__), "imagenet_rules.yml")
# Measured ~96 MiB of activations per 384px image for ConvNeXt V2 Large (fp32),
# plus headroom for cuDNN workspaces.
VRAM_PER_IMAGE = 128 * 1024 ** 2
MAX_BATCH_SIZE = 64

PROVIDER_ID = "recognize_backend:image:classification"
PROVIDER_NAME = "Recognize Image Classification (ConvNeXt V2)"
EXPECTED_RUNTIME = 15

LOGGER = logging.getLogger(os.environ.get("APP_ID", "recognize_backend"))


def _clean_label(label: str) -> str:
    # ImageNet labels are often a comma-separated synonym list, e.g.
    # "tabby, tabby cat". Keep only the first, most-specific synonym.
    return label.split(",", 1)[0].strip()


def load_rules() -> dict:
    import yaml

    with open(RULES_PATH) as f:
        return yaml.safe_load(f)


def _find_rule(rules: dict, name: str, depth: int = 0) -> dict | None:
    rule = rules.get(name)
    if rule and "see" in rule and depth < 10:
        return _find_rule(rules, rule["see"], depth + 1)
    return rule


def apply_rules(rules: dict, predictions: list[tuple[str, float]]) -> list[str]:
    """Map ``(imagenet label, probability)`` pairs to tags.

    Port of recognize's ``src/classifier_imagenet.js``: a class contributes its
    label and categories when its probability reaches the rule's threshold,
    and a category shared by several classes is added when the root of their
    summed squared probabilities reaches the highest of their thresholds.
    """
    matched = [
        (prob, rule)
        for label, prob in predictions
        if (rule := _find_rule(rules, _clean_label(label).lower()))
    ]
    tags = []
    for prob, rule in matched:
        if prob >= rule["threshold"]:
            if rule.get("label"):
                tags.append(rule["label"])
            tags.extend(rule.get("categories") or [])

    cat_prob: dict[str, float] = {}
    cat_threshold: dict[str, float] = {}
    cat_count: dict[str, int] = {}
    for prob, rule in matched:
        categories = ([rule["label"]] if rule.get("label") else []) + (rule.get("categories") or [])
        for category in dict.fromkeys(categories):
            cat_prob[category] = cat_prob.get(category, 0.0) + prob ** 2
            cat_threshold[category] = max(cat_threshold.get(category, 0.0), rule["threshold"])
            cat_count[category] = cat_count.get(category, 0) + 1
    for category, prob in cat_prob.items():
        if cat_count[category] > 1 and math.sqrt(prob) >= cat_threshold[category]:
            tags.append(category)

    return list(dict.fromkeys(tag[0].upper() + tag[1:] for tag in tags if tag))


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

    rules = load_rules()
    # Only use the rules if the model actually predicts ImageNet-1k classes.
    covered = sum(_clean_label(label).lower() in rules for label in id2label.values())
    use_rules = covered >= 0.9 * len(id2label)
    if not use_rules:
        LOGGER.warning(
            "%s labels are not covered by the ImageNet rules (%d of %d), emitting raw labels",
            DEFAULT_MODEL,
            covered,
            len(id2label),
        )

    @torch.no_grad()
    def infer(items: list) -> list[str]:
        logits = model(pixel_values=torch.stack(items).to(device)).logits
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        top = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))
        results = []
        for scores, indices in zip(top.values.tolist(), top.indices.tolist()):
            predictions = [(id2label[idx], score) for score, idx in zip(scores, indices)]
            if use_rules:
                tags = apply_rules(rules, predictions)
            else:
                tags = list(dict.fromkeys(
                    _clean_label(label) for label, score in predictions if score >= SCORE_THRESHOLD
                ))
            results.append(", ".join(tags))
        return results

    return Processor(
        load=load,
        infer=infer,
        batch_size=batch_size_for(VRAM_PER_IMAGE, MAX_BATCH_SIZE),
    )
