# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classifier task processors for the recognize_backend ex-app.

Each module exposes a factory ``build()`` that returns a callable taking
the local file path of a media file and returning a string (the value to
report back for the corresponding task input).
"""

from . import audio, face, image, video

TASK_TYPES = {
    "recognize:audio:classification": audio,
    "recognize:image:classification": image,
    "recognize:image:facerecognition": face,
    "recognize:video:classification": video,
}
