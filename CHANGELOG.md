<!--
  - SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

## [1.1.0] - 2026-10-28

#### Added
- enh: use imagenet rules from old recognize code base to improve object detection and classification

### Fixed
- perf: Enhance throughput
- fix: Retry task result reporting and recover from reboot

## [1.0.6] - 2026-10-20

### Fixed
- Build: Use CUDA 12.8 + cuDNN 9 base image and onnxruntime-gpu 1.22 so face recognition actually runs on the GPU

## [1.0.5] - 2026-09-26

### Fixed

- Fix: Reduce default face detection photo size back to 640

## [1.0.4] - 2026-08-06

### Fixed

- fix(face): Harden against non-loadable images


## [1.0.3] - 2026-07-27

### Fixed

- Drop support for nc < 35


## [1.0.2] - 2026-07-27

### Fixed

- Support nc 35

## [1.0.1] - 2026-07-27

### Fixed

- Retrigger release action

## [1.0.0] - 2026-07-27

Initial release.

* Adds TaskProcessing providers for the task types defined in
  [nextcloud/recognize#1500](https://github.com/nextcloud/recognize/pull/1500):
  * `recognize:audio:classification` (Audio Spectrogram Transformer / AudioSet)
  * `recognize:image:classification` (ConvNeXt V2 / ImageNet-22k)
  * `recognize:image:facerecognition` (InsightFace `buffalo_l`)
  * `recognize:video:classification` (VideoMAE / Kinetics-400)
