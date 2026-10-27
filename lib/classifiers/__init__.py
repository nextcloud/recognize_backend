# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classifier task processors for the recognize_backend ex-app.

Each module exposes a factory ``build()`` that returns a
:class:`~classifiers._processor.Processor`: ``load`` decodes one local media
file (run in the download thread), ``infer`` turns a batch of loaded items
into one string per item (the value to report back for that task input).
"""

from . import audio, face, image, video

TASK_TYPES = {
    "recognize:audio:classification": audio,
    "recognize:image:classification": image,
    "recognize:image:facerecognition": face,
    "recognize:video:classification": video,
}
