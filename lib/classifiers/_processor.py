# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interface between the task loop and the classifier modules."""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Processor:
    # Decode/preprocess one local file. Runs in the download thread, so it must
    # not touch the GPU and must return host-side data.
    load: Callable[[str], Any]
    # Run inference on a batch of loaded items, returning one string per item.
    infer: Callable[[list[Any]], list[str]]
    # Maximum number of items passed to ``infer`` at once.
    batch_size: int = 1
