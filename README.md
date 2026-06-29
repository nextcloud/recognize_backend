<!--
  - SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Nextcloud Recognize Backend

[![REUSE status](https://api.reuse.software/badge/github.com/nextcloud/recognize_backend)](https://api.reuse.software/info/github.com/nextcloud/recognize_backend)

On-premises classification backend for the
[Recognize](https://github.com/nextcloud/recognize) app. Implements the
TaskProcessing task types contributed by
[nextcloud/recognize#1500](https://github.com/nextcloud/recognize/pull/1500):

| Task type                          | Model                                           | Library                |
|------------------------------------|-------------------------------------------------|------------------------|
| `recognize:audio:classification`   | Audio Spectrogram Transformer (AudioSet, 527 c.)| 🤗 `transformers`      |
| `recognize:image:classification`   | ConvNeXt V2 (Large, ImageNet-22k)               | 🤗 `transformers`      |
| `recognize:image:facerecognition`  | InsightFace `buffalo_l` (SCRFD + ArcFace)       | `insightface`          |
| `recognize:video:classification`   | VideoMAE (Large, Kinetics-400)                  | 🤗 `transformers` + PyAV |

The default model for each task type can be swapped with any compatible
Hugging Face / InsightFace model via the environment variables documented
in `appinfo/info.xml` (or the AppAPI deploy daemon UI).

This app is to be deployed via Nextcloud
[AppAPI](https://github.com/cloud-py-api/app_api) like
[`llm2`](https://github.com/nextcloud/llm2) or
[`stt_whisper2`](https://github.com/nextcloud/stt_whisper2). The image is
CUDA-enabled (nvidia/cuda 12.2 base) and falls back to CPU automatically
when no GPU is available.

## Installation

See the [Nextcloud admin documentation](https://docs.nextcloud.com/server/latest/admin_manual/ai/overview.html)
for general AppAPI setup instructions.

## Development installation

1. Create and activate a Python virtual environment:

   ```sh
   python3 -m venv ./venv && . ./venv/bin/activate
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Run the app:

   ```sh
   cd lib
   # Configure your local Nextcloud (see the constants block at the top of lib/main.py)
   python3 main.py
   ```

4. Register the running app with your local Nextcloud:

   ```sh
   make register
   ```

## How it works

* On enable, the app registers one TaskProcessing provider for each of
  the four task types declared by Recognize.
* A background thread polls `taskprocessing/next_task` for those
  provider IDs.
* Each input file from `task['input']['input']` is downloaded over OCS,
  fed into the corresponding classifier, and the resulting string is
  appended to `output`. For image/audio/video classification this is a
  comma-separated label list; for face recognition it is a `\n`-separated
  list of JSON-encoded 512-dim embedding vectors, one per detected face.
* Models are lazy-loaded the first time a task of their type is seen and
  cached in-process. Hugging Face weights are cached in
  `/nc_app_recognize_backend_data/huggingface`; InsightFace packs in
  `/nc_app_recognize_backend_data/insightface`.

## Ethical AI Rating
### Rating: 🟡

Positive:
* the software for training and inference of all bundled models is open source
* the trained models are freely available, and thus can be run on-premises

Negative:
* the training data of the models is not freely available, limiting the ability
  of external parties to check and correct for bias or optimise performance and
  CO2 usage.

Learn more about the Nextcloud Ethical AI Rating
[in our blog](https://nextcloud.com/blog/nextcloud-ethical-ai-rating/).

## License
AGPL-3.0-or-later
