<!--
  - SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

## 1.0.0

Initial release.

* Adds TaskProcessing providers for the task types defined in
  [nextcloud/recognize#1500](https://github.com/nextcloud/recognize/pull/1500):
  * `recognize:audio:classification` (Audio Spectrogram Transformer / AudioSet)
  * `recognize:image:classification` (ConvNeXt V2 / ImageNet-22k)
  * `recognize:image:facerecognition` (InsightFace `buffalo_l`)
  * `recognize:video:classification` (VideoMAE / Kinetics-400)
