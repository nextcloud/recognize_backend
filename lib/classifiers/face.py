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
import math
import os

from nc_py_api.ex_app import persistent_storage

from ._device import onnx_providers

MODEL_PACK = os.environ.get("RECOGNIZE_FACE_MODEL", "buffalo_l")
DET_SIZE = int(os.environ.get("RECOGNIZE_FACE_DET_SIZE", "640"))

PROVIDER_ID = "recognize_backend:image:facerecognition"
PROVIDER_NAME = "Recognize Face Recognition (InsightFace ArcFace)"
EXPECTED_RUNTIME = 15


def _roll_angle(kps) -> float:
    """Roll angle in degrees from the 5-point landmark eye keypoints."""
    if kps is None or len(kps) < 2:
        return 0.0
    left_eye, right_eye = kps[0], kps[1]
    dx = float(right_eye[0]) - float(left_eye[0])
    dy = float(right_eye[1]) - float(left_eye[1])
    return math.degrees(math.atan2(dy, dx))


def build():
    import cv2
    from insightface.app import FaceAnalysis

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

    def run(path: str) -> str:
        img = cv2.imread(path)
        if img is None:
            from PIL import Image
            import numpy as np

            with Image.open(path) as pil:
                img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = app.get(img)
        lines = []
        for face in faces:
            embedding = face.normed_embedding
            if embedding is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            h, w = img.shape[:2]
            lines.append(
                json.dumps(
                    {
                        "x": x1 / w,
                        "y": y1 / h,
                        "width": (x2 - x1) / w,
                        "height": (y2 - y1) / h,
                        "score": float(face.det_score),
                        "angle": _roll_angle(face.kps),
                        "vector": [float(v) for v in embedding],
                    }
                )
            )
        return "\n".join(lines)

    return run
