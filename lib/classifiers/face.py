# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Face recognition task processor.

Uses InsightFace's ``buffalo_l`` model pack (SCRFD-10G detector +
Glint360k-trained ArcFace embedding) which produces 512-dimensional
face embeddings. One face is emitted per line as a JSON object of the
form::

    {"x": int, "y": int, "width": int, "height": int,
     "score": float, "angle": float, "vector": [float, ...]}

``angle`` is the in-plane roll angle in degrees, derived from the two
eye keypoints (positive = right eye below left eye).
"""

import json
import logging
import math
import os

from nc_py_api.ex_app import persistent_storage

from ._device import batch_size_for, onnx_providers
from ._processor import Processor

MODEL_PACK = os.environ.get("RECOGNIZE_FACE_MODEL", "buffalo_l")
DET_SIZE = int(os.environ.get("RECOGNIZE_FACE_DET_SIZE", "640"))
# JPEGs are decoded at a reduced DCT scale (1/2, 1/4, 1/8) as long as the long
# side stays at least this large, which is much cheaper than a full decode.
DECODE_SIZE = int(os.environ.get("RECOGNIZE_FACE_DECODE_SIZE", "1600"))
# Detection runs per image (det_10g has a fixed batch dimension of 1), so this
# mostly sizes the ArcFace batches, which hold every face of a batch of images.
VRAM_PER_IMAGE = 96 * 1024 ** 2
MAX_BATCH_SIZE = 16
MAX_EMBED_BATCH = 64

PROVIDER_ID = "recognize_backend:image:facerecognition"
PROVIDER_NAME = "Recognize Face Recognition (InsightFace ArcFace)"
EXPECTED_RUNTIME = 15

LOGGER = logging.getLogger(os.environ.get("APP_ID", "recognize_backend"))


def _roll_angle(kps) -> float:
    """Roll angle in degrees from the 5-point landmark eye keypoints."""
    if kps is None or len(kps) < 2:
        return 0.0
    left_eye, right_eye = kps[0], kps[1]
    dx = float(right_eye[0]) - float(left_eye[0])
    dy = float(right_eye[1]) - float(left_eye[1])
    return math.degrees(math.atan2(dy, dx))


def _decode(path: str):
    """Decode an image as a BGR array, or return None if it can't be read."""
    import cv2

    # Load via PIL so we can honor the EXIF orientation tag; cv2.imread
    # ignores it, which would flip/rotate bounding boxes on phone photos.
    try:
        from PIL import Image, ImageOps
        import numpy as np

        with Image.open(path) as pil:
            w, h = pil.size
            scale = DECODE_SIZE / max(w, h)
            if scale < 1:
                # No-op for non-JPEG formats
                pil.draft("RGB", (math.ceil(w * scale), math.ceil(h * scale)))
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(path)


def build() -> Processor:
    import numpy as np
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    root = os.path.join(persistent_storage(), "insightface")
    os.makedirs(root, exist_ok=True)

    providers = onnx_providers()
    app = FaceAnalysis(
        name=MODEL_PACK,
        root=root,
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    app.prepare(ctx_id=ctx_id, det_size=(DET_SIZE, DET_SIZE))
    # onnxruntime only logs a warning and drops to CPU when the CUDA provider
    # cannot be loaded (e.g. CUDA/cuDNN mismatch), so surface that explicitly.
    for task, model in app.models.items():
        active = model.session.get_providers()
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active:
            LOGGER.warning("Face %s model is running on %s instead of CUDA", task, active)
    detector = app.det_model
    recognizer = app.models["recognition"]

    def embed(crops: list) -> list:
        embeddings = []
        for i in range(0, len(crops), MAX_EMBED_BATCH):
            embeddings.extend(recognizer.get_feat(crops[i:i + MAX_EMBED_BATCH]))
        return [e / np.linalg.norm(e) for e in embeddings]

    def infer(imgs: list) -> list[str]:
        # Detect per image, then embed all faces of the batch at once.
        detections = []
        crops = []
        for img in imgs:
            if img is None:
                detections.append([])
                continue
            bboxes, kpss = detector.detect(img, max_num=0, metric="default")
            faces = []
            for i in range(bboxes.shape[0]):
                if kpss is None:
                    continue
                faces.append((bboxes[i], kpss[i]))
                crops.append(
                    face_align.norm_crop(img, landmark=kpss[i], image_size=recognizer.input_size[0])
                )
            detections.append(faces)
        embeddings = iter(embed(crops))

        results = []
        for img, faces in zip(imgs, detections):
            lines = []
            for bbox, kps in faces:
                embedding = next(embeddings)
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                h, w = img.shape[:2]
                lines.append(
                    json.dumps(
                        {
                            "x": x1 / w,
                            "y": y1 / h,
                            "width": (x2 - x1) / w,
                            "height": (y2 - y1) / h,
                            "score": float(bbox[4]),
                            "angle": _roll_angle(kps),
                            "vector": [float(v) for v in embedding],
                        }
                    )
                )
            results.append("\n".join(lines))
        return results

    return Processor(
        load=_decode,
        infer=infer,
        batch_size=batch_size_for(VRAM_PER_IMAGE, MAX_BATCH_SIZE),
    )
