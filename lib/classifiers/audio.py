# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audio classification task processor.

Uses MIT's Audio Spectrogram Transformer fine-tuned on AudioSet
(527 classes) via the Hugging Face ``transformers`` pipeline API.

When AST decides the audio is *music* (any predicted label whose name
contains "music" / "singing" scores above
:data:`MUSIC_DETECTION_THRESHOLD`), we additionally run a dedicated
music-genre classifier (``dima806/music_genres_classification`` by
default) and concatenate its labels onto the AST labels. This adds the
finer-grained genre detail that AST lacks while skipping the genre
model entirely for non-music audio.
"""

import os

from ._device import torch_device

DEFAULT_MODEL = os.environ.get(
    "RECOGNIZE_AUDIO_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593"
)
TOP_K = int(os.environ.get("RECOGNIZE_AUDIO_TOP_K", "5"))
SCORE_THRESHOLD = float(os.environ.get("RECOGNIZE_AUDIO_THRESHOLD", "0.2"))
TARGET_SAMPLE_RATE = 16000

MUSIC_GENRE_MODEL = os.environ.get(
    "RECOGNIZE_MUSIC_GENRE_MODEL", "dima806/music_genres_classification"
)
MUSIC_GENRE_ENABLED = os.environ.get("RECOGNIZE_MUSIC_GENRE_ENABLED", "1") == "1"
MUSIC_GENRE_TOP_K = int(os.environ.get("RECOGNIZE_MUSIC_GENRE_TOP_K", "3"))
MUSIC_GENRE_THRESHOLD = float(os.environ.get("RECOGNIZE_MUSIC_GENRE_THRESHOLD", "0.25"))
MUSIC_DETECTION_THRESHOLD = float(
    os.environ.get("RECOGNIZE_MUSIC_DETECTION_THRESHOLD", "0.3")
)
MUSIC_LABEL_SUBSTRINGS = ("music", "singing")

PROVIDER_ID = "recognize_backend:audio:classification"
PROVIDER_NAME = "Recognize Audio Classification (AST/AudioSet + music genre)"
EXPECTED_RUNTIME = 30


def _is_music(predictions) -> bool:
    for p in predictions:
        label = p.get("label", "").lower()
        if p.get("score", 0.0) < MUSIC_DETECTION_THRESHOLD:
            continue
        if any(s in label for s in MUSIC_LABEL_SUBSTRINGS):
            return True
    return False


def build():
    from transformers import pipeline

    device = torch_device()
    hf_device = 0 if device == "cuda" else -1

    classifier = pipeline(
        task="audio-classification",
        model=DEFAULT_MODEL,
        device=hf_device,
    )

    # Lazily loaded music-genre head; built on first music hit and reused.
    genre_state: dict = {"classifier": None, "tried": False}

    def _get_genre_classifier():
        if not MUSIC_GENRE_ENABLED:
            return None
        if genre_state["classifier"] is not None:
            return genre_state["classifier"]
        if genre_state["tried"]:
            return None
        genre_state["tried"] = True
        try:
            genre_state["classifier"] = pipeline(
                task="audio-classification",
                model=MUSIC_GENRE_MODEL,
                device=hf_device,
            )
        except Exception:
            # Don't bring the whole audio pipeline down because the optional
            # genre head failed to load; just skip genre detection.
            genre_state["classifier"] = None
        return genre_state["classifier"]

    def _load(path: str):
        import librosa

        wav, _ = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
        return wav

    def run(path: str) -> str:
        wav = _load(path)
        # Pull a few extra predictions so we can detect music even when the
        # user-facing TOP_K is small.
        detect_k = max(TOP_K, 10)
        predictions = classifier(wav, top_k=detect_k)

        labels = [
            p["label"]
            for p in predictions[:TOP_K]
            if p.get("score", 0.0) >= SCORE_THRESHOLD
        ]
        if not labels and predictions:
            labels = [predictions[0]["label"]]

        if _is_music(predictions):
            genre = _get_genre_classifier()
            if genre is not None:
                try:
                    genre_preds = genre(wav, top_k=MUSIC_GENRE_TOP_K)
                    for p in genre_preds:
                        if p.get("score", 0.0) >= MUSIC_GENRE_THRESHOLD:
                            labels.append(p["label"])
                    if not any(
                        p.get("score", 0.0) >= MUSIC_GENRE_THRESHOLD
                        for p in genre_preds
                    ) and genre_preds:
                        labels.append(genre_preds[0]["label"])
                except Exception:
                    # If the genre head fails on this clip, fall back silently
                    # to AST-only labels rather than failing the task.
                    pass

        seen = set()
        uniq = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                uniq.append(label)
        return ", ".join(uniq)

    return run
