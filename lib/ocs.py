# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helpers for downloading task-processing input files via OCS.

Adapted from nextcloud/stt_whisper2.
"""

import tempfile
import typing
from json import loads

from nc_py_api import NextcloudException
from niquests import Response, codes


def _check_error(response: Response, info: str = ""):
    status_code = response.status_code
    if not info:
        info = f"request: {response.request.method} {response.request.url}"
    if 996 <= status_code <= 999:
        phrase = {
            996: "Server error",
            997: "Unauthorised",
            998: "Not found",
        }.get(status_code, "Unknown error")
        raise NextcloudException(status_code, reason=phrase, info=info)
    if status_code < 400:
        return
    raise NextcloudException(status_code, reason=codes(status_code).phrase, info=info)


def _ocs_download(
    nc_session,
    method: str,
    path: str,
    *,
    content: bytes | str | typing.Iterable[bytes] | typing.AsyncIterable[bytes] | None = None,
    json: dict | list | None = None,
    params: dict | None = None,
    files: dict | None = None,
    **kwargs,
) -> str:
    nc_session.init_adapter()
    info = f"request: {method} {path}"
    response: Response = nc_session.adapter.request(
        method, path, data=content, json=json, params=params, files=files, stream=True, **kwargs
    )
    if response.status_code >= 400:
        try:
            print(loads(response.text))
        except Exception:
            pass
    _check_error(response, info)
    if response.status_code == 204:
        raise NextcloudException(204, reason="No content")
    with tempfile.NamedTemporaryFile(delete=False, mode="wb") as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        temp_file.flush()
        return temp_file.name


def get_file(nc, task_id: int, file_id: int) -> str:
    """Download a TaskProcessing input file and return the local path."""
    return _ocs_download(
        nc._session,
        "GET",
        f"/ocs/v2.php/taskprocessing/tasks_provider/{task_id}/file/{file_id}",
    )
