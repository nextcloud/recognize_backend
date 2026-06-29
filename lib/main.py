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
from classifiers._device import force_cpu, is_cuda_runtime_error, torch_device
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

PROCESSOR_CACHE: dict[str, callable] = {}
PROCESSOR_LOCK = threading.Lock()


def _model_name(module) -> str:
    return getattr(module, "DEFAULT_MODEL", None) or getattr(module, "MODEL_PACK", "?")


def _free_cuda_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # torch may not be importable in face-only paths
        pass


def _build_with_fallback(task_type: str):
    """Build the processor and wrap it so CUDA failures fall back to CPU.

    On the first CUDA/cuDNN runtime error observed during inference we:

    * call :func:`force_cpu` so every subsequent model build stays on CPU,
    * free the CUDA allocator cache,
    * rebuild this task's processor on CPU,
    * retry the same input once.

    Subsequent calls go straight to the CPU processor.
    """
    module = TASK_TYPES[task_type]
    built_on = torch_device()
    LOGGER.info("Loading %s processor (%s) on %s", task_type, _model_name(module), built_on)
    state = {"fn": module.build(), "device": built_on}

    def run(path: str) -> str:
        try:
            return state["fn"](path)
        except Exception as e:
            if state["device"] != "cuda" or not is_cuda_runtime_error(e):
                raise
            LOGGER.warning(
                "CUDA runtime error in %s, switching to CPU permanently: %s",
                task_type,
                e,
            )
            force_cpu()
            _free_cuda_cache()
            state["fn"] = module.build()
            state["device"] = "cpu"
            return state["fn"](path)

    return run


def get_processor(task_type: str):
    """Lazy-load and memoise the processor for a task type."""
    with PROCESSOR_LOCK:
        if task_type in PROCESSOR_CACHE:
            return PROCESSOR_CACHE[task_type]
        processor = _build_with_fallback(task_type)
        PROCESSOR_CACHE[task_type] = processor
        return processor


def provider_for(task_type: str) -> TaskProcessingProvider:
    module = TASK_TYPES[task_type]
    return TaskProcessingProvider(
        id=module.PROVIDER_ID,
        name=module.PROVIDER_NAME,
        task_type=task_type,
        expected_runtime=module.EXPECTED_RUNTIME,
    )


def process_task(task: dict, task_type: str) -> dict:
    """Run a task: download every input file, classify, return ``{"output": [...]}``."""
    nc = NextcloudApp()
    processor = get_processor(task_type)
    file_ids = task.get("input", {}).get("input")
    if not isinstance(file_ids, list):
        raise ValueError(f"Unexpected input shape: {task.get('input')!r}")

    results: list[str] = []
    total = max(1, len(file_ids))
    downloaded_paths: list[str] = []
    try:
        for i, file_id in enumerate(file_ids):
            try:
                path = get_file(nc, task["id"], int(file_id))
            except Exception as e:
                LOGGER.error("Failed to download file %s for task %s: %s", file_id, task["id"], e)
                results.append("")
                continue
            downloaded_paths.append(path)
            try:
                results.append(processor(path))
            except Exception as e:
                LOGGER.error(
                    "Processor failed on file %s (task %s): %s\n%s",
                    file_id,
                    task["id"],
                    e,
                    "".join(traceback.format_exception(e)),
                )
                results.append("")
            try:
                nc.providers.task_processing.set_progress(task["id"], (i + 1) / total)
            except Exception as e:
                LOGGER.warning("Failed to report progress for task %s: %s", task["id"], e)
    finally:
        for path in downloaded_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    return {"output": results}


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
            try:
                nc.providers.task_processing.report_result(
                    task["id"], None, f"Unknown provider: {provider_id}"
                )
            except Exception:
                pass
            continue

        try:
            LOGGER.info("Task %s: type=%s files=%s", task["id"], task_type, len(task.get("input", {}).get("input", []) or []))
            time_start = perf_counter()
            output = process_task(task, task_type)
            LOGGER.info("Task %s done in %.2fs", task["id"], perf_counter() - time_start)
            nc.providers.task_processing.report_result(task["id"], output)
        except Exception as e:
            LOGGER.error("%s\n%s", e, "".join(traceback.format_exception(e)))
            try:
                nc.providers.task_processing.report_result(task["id"], None, str(e))
            except Exception:
                pass


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
    nc = NextcloudApp()
    if nc.enabled_state:
        ENABLED.set()
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
