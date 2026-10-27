# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Main entry point of the recognize_backend ex-app.

This is a Nextcloud ExApp that registers TaskProcessing providers for the
classification task types contributed by recognize PR #1500:

* recognize:audio:classification
* recognize:image:classification
* recognize:image:facerecognition
* recognize:video:classification

Models are loaded lazily on the first task per type, then cached for the
process lifetime. The polling loop mirrors stt_whisper2.
"""

import logging
import os
import queue
import threading
import traceback
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter, sleep

import niquests
from fastapi import FastAPI
from nc_py_api import AsyncNextcloudApp, NextcloudApp, NextcloudException
from nc_py_api.ex_app import run_app, set_handlers, setup_nextcloud_logging
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider

from classifiers import TASK_TYPES
from classifiers._device import force_cpu, is_cuda_oom, is_cuda_runtime_error, torch_device
from ocs import get_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(os.environ.get("APP_ID", "recognize_backend"))
LOGGER.setLevel(logging.DEBUG)

ENABLED = Event()
TRIGGER = Event()
WAIT_INTERVAL = int(os.environ.get("TASK_POLLING_INTERVAL", "5"))
WAIT_INTERVAL_WITH_TRIGGER = 5 * 60
# How long to keep retrying to deliver a finished task's result before giving up.
REPORT_RESULT_DEADLINE = 60 * 60
RETRY_BACKOFF_MAX = 60

# Report task progress at most this often (seconds).
PROGRESS_INTERVAL = 5

PROCESSOR_CACHE: dict[str, "Runner"] = {}
PROCESSOR_LOCK = threading.Lock()


def _model_name(module) -> str:
    return getattr(module, "DEFAULT_MODEL", None) or getattr(module, "MODEL_PACK", "?")


def _free_cuda_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # torch may not be importable in face-only paths
        pass


class Runner:
    """A task type's processor, with CUDA out-of-memory back-off and CPU fallback.

    When a batch runs out of GPU memory, the batch size is halved and the batch
    retried. On the first other CUDA/cuDNN runtime error we:

    * call :func:`force_cpu` so every subsequent model build stays on CPU,
    * free the CUDA allocator cache,
    * rebuild this task's processor on CPU,
    * retry the same batch once.
    """

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.module = TASK_TYPES[task_type]
        self._build()

    def _build(self) -> None:
        self.device = torch_device()
        LOGGER.info(
            "Loading %s processor (%s) on %s", self.task_type, _model_name(self.module), self.device
        )
        self.processor = self.module.build()
        LOGGER.info("Using batch size %d for %s", self.processor.batch_size, self.task_type)

    @property
    def batch_size(self) -> int:
        return self.processor.batch_size

    def load(self, path: str):
        return self.processor.load(path)

    def infer(self, items: list) -> list[str]:
        results: list[str] = []
        while len(results) < len(items):
            chunk = items[len(results):len(results) + self.processor.batch_size]
            try:
                results.extend(self.processor.infer(chunk))
                continue
            except Exception as e:
                if self.device != "cuda":
                    raise
                if is_cuda_oom(e) and self.processor.batch_size > 1:
                    self.processor.batch_size = max(1, self.processor.batch_size // 2)
                    LOGGER.warning(
                        "Out of GPU memory in %s, reducing batch size to %d",
                        self.task_type,
                        self.processor.batch_size,
                    )
                    _free_cuda_cache()
                    continue
                if not is_cuda_runtime_error(e):
                    raise
                LOGGER.warning(
                    "CUDA runtime error in %s, switching to CPU permanently: %s",
                    self.task_type,
                    e,
                )
            force_cpu()
            _free_cuda_cache()
            self._build()
        return results


def get_processor(task_type: str) -> Runner:
    """Lazy-load and memoise the processor for a task type."""
    with PROCESSOR_LOCK:
        if task_type not in PROCESSOR_CACHE:
            PROCESSOR_CACHE[task_type] = Runner(task_type)
        return PROCESSOR_CACHE[task_type]


def provider_for(task_type: str) -> TaskProcessingProvider:
    module = TASK_TYPES[task_type]
    return TaskProcessingProvider(
        id=module.PROVIDER_ID,
        name=module.PROVIDER_NAME,
        task_type=task_type,
        expected_runtime=module.EXPECTED_RUNTIME,
    )


def _load_files(task_id: int, file_ids: list, runner: Runner, out: queue.Queue, stop: Event) -> None:
    """Download and decode the task's files in order, feeding ``(index, item)`` into ``out``.

    ``item`` is None if the file could not be downloaded or decoded. A final
    None marks the end. Runs in its own thread so downloads and decoding
    overlap with inference.
    """
    nc = NextcloudApp()

    def put(entry) -> bool:
        while not stop.is_set():
            try:
                out.put(entry, timeout=1)
                return True
            except queue.Full:
                pass
        return False

    try:
        for i, file_id in enumerate(file_ids):
            if stop.is_set():
                return
            item = None
            try:
                path = get_file(nc, task_id, int(file_id))
            except Exception as e:
                LOGGER.error("Failed to download file %s for task %s: %s", file_id, task_id, e)
            else:
                try:
                    item = runner.load(path)
                except Exception as e:
                    LOGGER.error("Failed to load file %s for task %s: %s", file_id, task_id, e)
                finally:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if not put((i, item)):
                return
    finally:
        put(None)


def _infer_batch(runner: Runner, batch: list, file_ids: list, task_id: int) -> list[str]:
    """Run inference on ``[(index, item), ...]``; isolate failures to single files."""
    items = [item for _, item in batch]
    try:
        return runner.infer(items)
    except Exception as e:
        if len(batch) > 1:
            LOGGER.warning("Batch failed for task %s, retrying files one by one: %s", task_id, e)
    results = []
    for i, item in batch:
        try:
            results.extend(runner.infer([item]))
        except Exception as e:
            LOGGER.error(
                "Processor failed on file %s (task %s): %s\n%s",
                file_ids[i],
                task_id,
                e,
                "".join(traceback.format_exception(e)),
            )
            results.append("")
    return results


def process_task(task: dict, task_type: str) -> dict:
    """Run a task: download every input file, classify, return ``{"output": [...]}``."""
    nc = NextcloudApp()
    runner = get_processor(task_type)
    file_ids = task.get("input", {}).get("input")
    if not isinstance(file_ids, list):
        raise ValueError(f"Unexpected input shape: {task.get('input')!r}")

    results = [""] * len(file_ids)
    total = max(1, len(file_ids))
    done = 0
    last_progress = perf_counter()
    # Holds up to one batch of loaded items while the previous one is inferred.
    loaded: queue.Queue = queue.Queue(maxsize=runner.batch_size)
    stop = Event()
    loader = threading.Thread(
        target=_load_files, args=(task["id"], file_ids, runner, loaded, stop), daemon=True
    )
    loader.start()
    try:
        finished = False
        while not finished:
            # Block for the next item, then take whatever else is ready up to
            # the batch size, so the GPU never idles waiting for a full batch.
            batch = []
            entry = loaded.get()
            while True:
                if entry is None:
                    finished = True
                    break
                i, item = entry
                done += 1
                if item is not None:
                    batch.append((i, item))
                if len(batch) >= runner.batch_size:
                    break
                try:
                    entry = loaded.get_nowait()
                except queue.Empty:
                    break
            if batch:
                for (i, _), output in zip(batch, _infer_batch(runner, batch, file_ids, task["id"])):
                    results[i] = output
            if not finished and perf_counter() - last_progress >= PROGRESS_INTERVAL:
                last_progress = perf_counter()
                try:
                    # nc_py_api expects a percentage (0-100)
                    nc.providers.task_processing.set_progress(task["id"], done / total * 100)
                except Exception as e:
                    LOGGER.warning("Failed to report progress for task %s: %s", task["id"], e)
    finally:
        stop.set()
        loader.join()

    return {"output": results}


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (niquests.exceptions.ConnectionError, niquests.exceptions.Timeout)):
        return True
    if isinstance(e, NextcloudException):
        return e.status_code == 408 or e.status_code >= 500
    return False


def report_result(nc: NextcloudApp, task_id: int, output: dict | None = None, error: str | None = None) -> None:
    """Deliver a task result, retrying transient failures.

    nc_py_api's ``report_result`` swallows HTTP errors, and a lost result leaves
    the task in STATUS_RUNNING forever, so we call the endpoint directly and keep
    retrying while Nextcloud is unreachable or returns a server error.
    """
    deadline = perf_counter() + REPORT_RESULT_DEADLINE
    delay = 2
    while True:
        try:
            nc._session.ocs(
                "POST",
                f"/ocs/v2.php/taskprocessing/tasks_provider/{task_id}/result",
                json={"taskId": task_id, "output": output, "errorMessage": error},
            )
            return
        except Exception as e:
            if not _is_transient(e):
                LOGGER.error("Failed to report result for task %s, giving up: %s", task_id, e)
                return
            if perf_counter() + delay > deadline:
                LOGGER.error("Failed to report result for task %s, retries exhausted: %s", task_id, e)
                return
            LOGGER.warning("Failed to report result for task %s, retrying in %ds: %s", task_id, delay, e)
            sleep(delay)
            delay = min(delay * 2, RETRY_BACKOFF_MAX)


def wait_for_initial_enabled_state(nc: NextcloudApp) -> None:
    """Fetch the enabled state, retrying until Nextcloud is reachable.

    ``nc.enabled_state`` returns False when Nextcloud cannot be reached, and
    AppAPI only calls the enabled handler on (re-)enable, not on a container
    restart. If the container starts before Nextcloud or HaRP (e.g. after a host
    reboot), trusting that False would leave the worker idle forever.
    """
    delay = 2
    while not ENABLED.is_set():
        try:
            if nc._session.ocs("GET", "/ocs/v1.php/apps/app_api/ex-app/state"):
                ENABLED.set()
            return
        except Exception as e:
            LOGGER.info("Nextcloud not reachable yet, retrying in %ds: %s", delay, e)
            sleep(delay)
            delay = min(delay * 2, RETRY_BACKOFF_MAX)


def wait_for_task(interval=None):
    global WAIT_INTERVAL
    if interval is None:
        interval = WAIT_INTERVAL
    if TRIGGER.wait(timeout=interval):
        WAIT_INTERVAL = WAIT_INTERVAL_WITH_TRIGGER
    TRIGGER.clear()


def background_thread_task():
    nc = NextcloudApp()
    provider_ids = [TASK_TYPES[t].PROVIDER_ID for t in TASK_TYPES]
    task_type_ids = list(TASK_TYPES.keys())

    wait_for_initial_enabled_state(nc)

    while True:
        while not ENABLED.is_set():
            sleep(5)

        try:
            item = nc.providers.task_processing.next_task(provider_ids, task_type_ids)
            if not isinstance(item, dict):
                wait_for_task()
                continue
            task = item.get("task")
            provider = item.get("provider")
            if task is None or provider is None:
                wait_for_task()
                continue
        except (niquests.exceptions.ConnectionError, niquests.exceptions.Timeout) as e:
            LOGGER.info("Temporary error fetching next task, will retry: %s", e)
            wait_for_task(5)
            continue
        except Exception as e:
            LOGGER.error("%s\n%s", e, "".join(traceback.format_exception(e)))
            wait_for_task(10)
            continue

        provider_id = provider.get("id") or provider.get("name")
        task_type = None
        for tt, module in TASK_TYPES.items():
            if module.PROVIDER_ID == provider_id:
                task_type = tt
                break
        if task_type is None:
            report_result(nc, task["id"], error=f"Unknown provider: {provider_id}")
            continue

        try:
            LOGGER.info("Task %s: type=%s files=%s", task["id"], task_type, len(task.get("input", {}).get("input", []) or []))
            time_start = perf_counter()
            output = process_task(task, task_type)
            LOGGER.info("Task %s done in %.2fs", task["id"], perf_counter() - time_start)
        except Exception as e:
            LOGGER.error("%s\n%s", e, "".join(traceback.format_exception(e)))
            report_result(nc, task["id"], error=str(e))
            continue
        report_result(nc, task["id"], output)


def start_bg_task():
    t = threading.Thread(target=background_thread_task, daemon=True)
    t.start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_nextcloud_logging(
        os.environ.get("APP_ID", "recognize_backend"),
        logging_level=logging.WARNING,
    )
    set_handlers(APP, enabled_handler, trigger_handler=trigger_handler)
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    if enabled:
        ENABLED.set()
        LOGGER.info("Hello from %s", nc.app_cfg.app_name)
        for task_type in TASK_TYPES:
            try:
                await nc.providers.task_processing.register(provider_for(task_type))
                LOGGER.info("Registered provider for %s", task_type)
            except NextcloudException as e:
                LOGGER.error("Failed to register %s: %s", task_type, e)
                return f"Failed to register {task_type}: {e}"
    else:
        ENABLED.clear()
        LOGGER.info("Bye bye from %s", nc.app_cfg.app_name)
        for task_type in TASK_TYPES:
            try:
                await nc.providers.task_processing.unregister(
                    TASK_TYPES[task_type].PROVIDER_ID, True
                )
            except NextcloudException as e:
                LOGGER.warning("Failed to unregister %s: %s", task_type, e)
    return ""


def trigger_handler(providerId: str):
    TRIGGER.set()


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
