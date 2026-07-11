"""Selected-image download and extraction helpers for EchoSight subsets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from shutil import disk_usage
import tarfile
import time
from typing import Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


STATUS_COMPLETE = "complete"
STATUS_DOWNLOAD_FAILED = "download_failed"
STATUS_INCOMPLETE_DOWNLOAD = "incomplete_download"
STATUS_INSUFFICIENT_DISK = "insufficient_disk"
STATUS_INVALID_IMAGE_PAYLOAD = "invalid_image_payload"
STATUS_INVALID_IMAGE_SUFFIX = "invalid_image_suffix"
STATUS_SKIPPED_TRANSFER_CAP = "skipped_transfer_cap"
STATUS_TRANSFER_CAP_EXCEEDED = "transfer_cap_exceeded"
RESUMABLE_PARTIAL_STATUSES = frozenset(
    {
        STATUS_DOWNLOAD_FAILED,
        STATUS_INCOMPLETE_DOWNLOAD,
        STATUS_INSUFFICIENT_DISK,
        STATUS_TRANSFER_CAP_EXCEEDED,
    }
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVE_SUFFIXES = {".tar"}
PAYLOAD_KIND_IMAGE = "image"
PAYLOAD_KIND_ARCHIVE = "archive"
ARCHIVE_PAYLOAD_KINDS = {PAYLOAD_KIND_ARCHIVE, "tar"}
HTML_XML_PREFIXES = (
    b"<!doctype html",
    b"<html",
    b"<?xml",
    b"<error",
    b"<Error",
)
SENSITIVE_HEADER_TOKENS = {
    "authorization",
    "auth",
    "token",
    "secret",
    "cookie",
    "credential",
}
SENSITIVE_QUERY_KEYS = {
    "api_key",
    "apikey",
    "key",
    "credential",
    "access_key",
    "token",
    "password",
    "secret",
    "session",
    "auth",
    "authorization",
    "signature",
    "sig",
    "expires",
}
SENSITIVE_QUERY_TOKENS = {
    "credential",
    "token",
    "password",
    "secret",
    "session",
    "auth",
    "authorization",
    "signature",
    "sig",
    "expires",
}


@dataclass(frozen=True)
class SelectedDownloadLimits:
    max_transfer_bytes: int = 120_000_000_000
    min_free_bytes: int = 15 * 1024 * 1024 * 1024
    timeout_seconds: float = 120.0
    chunk_size: int = 4 * 1024 * 1024
    max_retries: int = 2
    request_delay_seconds: float = 0.0
    rate_limit_cooldown_seconds: float = 600.0


@dataclass(frozen=True)
class SelectedDownloadRequest:
    dataset: str
    provider: str
    image_id: str
    url: str
    target_path: str
    headers: Mapping[str, str] | None = field(default=None, repr=False)
    payload_kind: str = PAYLOAD_KIND_IMAGE

    def to_report_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "provider": self.provider,
            "image_id": self.image_id,
            "url": _redact_url(self.url),
            "target_path": self.target_path,
            "headers": _safe_headers(self.headers),
            "payload_kind": self.payload_kind,
        }


class HTTPClient(Protocol):
    def open(self, url: str, *, headers=None, timeout=None):
        ...


class UrlopenHTTPClient:
    def open(self, url: str, *, headers=None, timeout=None):
        return urlopen(Request(url, headers=dict(headers or {})), timeout=timeout)


def execute_selected_downloads(
    root: str | Path,
    requests: Sequence[SelectedDownloadRequest],
    limits: SelectedDownloadLimits,
    http_client=None,
) -> dict[str, object]:
    """Download selected image requests under ``root`` with caps and sidecars."""

    _validate_limits(limits)
    root_path = Path(root)
    client = http_client or UrlopenHTTPClient()
    results: list[dict[str, object]] = []
    transferred_bytes_this_run = 0
    stopped_for_cap = False
    processed_download_request = False
    max_attempts = max(limits.max_retries, 0) + 1

    for request in requests:
        if stopped_for_cap:
            result = _skipped_result(root_path, request)
            _write_sidecar_for_request(root_path, request, result)
            results.append(result)
            continue

        if processed_download_request and limits.request_delay_seconds > 0:
            time.sleep(limits.request_delay_seconds)
        processed_download_request = True

        last_result: dict[str, object] | None = None
        previous_attempts: list[dict[str, object]] = []
        for attempt in range(max_attempts):
            result, transferred = _download_once(
                root_path,
                request,
                limits,
                client,
                transferred_bytes_this_run,
                retry_count=attempt,
            )
            transferred_bytes_this_run += transferred
            last_result = result
            if result["status"] == STATUS_TRANSFER_CAP_EXCEEDED:
                stopped_for_cap = True
                break
            if result["status"] == STATUS_COMPLETE or not _should_retry(result):
                break
            if attempt < max_attempts - 1:
                previous_attempts.append(result)
                retry_delay = _retry_delay_seconds(result)
                if retry_delay > 0:
                    time.sleep(retry_delay)

        if last_result is None:
            raise RuntimeError("download loop did not produce a result")
        if previous_attempts:
            last_result = dict(last_result)
            last_result["previous_attempts"] = previous_attempts
            _write_sidecar_for_request(root_path, request, last_result)
        results.append(last_result)

    summary = _download_summary(results, requested=len(requests))
    return {
        "status": STATUS_COMPLETE if summary["failed"] == 0 else "incomplete",
        "summary": summary,
        "transferred_bytes_this_run": transferred_bytes_this_run,
        "results": results,
    }


def extract_selected_tar_members(
    tar_path: str | Path,
    output_root: str | Path,
    member_names: Iterable[str],
    *,
    remove_tar: bool = True,
    min_free_bytes: int | None = None,
) -> dict[str, object]:
    """Extract only selected image members from a tar archive."""

    archive_path = Path(tar_path)
    output_root_path = Path(output_root)
    requested, duplicate_requested = _normalize_requested_member_names(member_names)
    seen_requested: set[str] = set()
    extracted = 0
    failed = len(duplicate_requested)
    skipped = 0
    results: list[dict[str, object]] = [
        {
            "member": name,
            "status": "duplicate_requested_member",
            "source_tar": str(archive_path),
            "error": "selected tar member was requested more than once",
        }
        for name in sorted(duplicate_requested)
    ]

    output_root_path.mkdir(parents=True, exist_ok=True)
    if min_free_bytes is None:
        min_free_bytes = SelectedDownloadLimits().min_free_bytes
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must not be negative")
    _ensure_disk_space(output_root_path, 0, min_free_bytes)

    with tarfile.open(archive_path, "r:*") as archive:
        selected_members: dict[str, tuple[tarfile.TarInfo, str, Path]] = {}
        duplicate_tar_members: set[str] = set()
        seen_tar_members: set[str] = set()
        non_file_members: set[str] = set()
        for member in archive:
            normalized_name = _normalize_member_name(member.name)
            if normalized_name not in requested:
                if member.isfile():
                    skipped += 1
                continue
            seen_requested.add(normalized_name)
            if normalized_name in duplicate_requested:
                continue
            if normalized_name in seen_tar_members:
                selected_members.pop(normalized_name, None)
                non_file_members.discard(normalized_name)
                duplicate_tar_members.add(normalized_name)
                continue
            seen_tar_members.add(normalized_name)
            if normalized_name in duplicate_tar_members:
                continue
            if not member.isfile():
                non_file_members.add(normalized_name)
                continue

            try:
                target_path = _resolve_member_target(output_root_path, normalized_name)
                _ensure_image_suffix(target_path)
                selected_members[normalized_name] = (member, normalized_name, target_path)
            except Exception as exc:
                failed += 1
                failure_status = _tar_extract_failure_status(exc)
                result = {
                    "member": normalized_name,
                    "status": failure_status,
                    "source_tar": str(archive_path),
                    "error": str(exc),
                }
                try:
                    target_path = _resolve_member_target(output_root_path, normalized_name)
                    _write_sidecar(_sidecar_path(target_path), result)
                except ValueError:
                    pass
                results.append(result)

        for non_file_name in sorted(non_file_members - duplicate_tar_members):
            failed += 1
            results.append(
                {
                    "member": non_file_name,
                    "status": "not_a_file",
                    "error": "selected tar member is not a regular file",
                }
            )

        for duplicate_name in sorted(duplicate_tar_members):
            failed += 1
            results.append(
                {
                    "member": duplicate_name,
                    "status": "duplicate_tar_member",
                    "source_tar": str(archive_path),
                    "error": "selected tar member appears more than once in archive",
                }
            )

        selected_bytes = sum(
            max(int(member.size), 0) for member, _, _ in selected_members.values()
        )
        _ensure_disk_space(output_root_path, selected_bytes, min_free_bytes)

        for member, normalized_name, target_path in selected_members.values():
            try:
                extracted_bytes = _extract_member_to_target(
                    archive,
                    member,
                    target_path,
                    output_root_path,
                    min_free_bytes,
                )
                extracted += 1
                result = {
                    "member": normalized_name,
                    "status": STATUS_COMPLETE,
                    "local_path": _relative_path(output_root_path, target_path),
                    "bytes_extracted": extracted_bytes,
                    "source_tar": str(archive_path),
                }
                _write_sidecar(_sidecar_path(target_path), result)
                results.append(result)
            except Exception as exc:
                failed += 1
                failure_status = _tar_extract_failure_status(exc)
                result = {
                    "member": normalized_name,
                    "status": failure_status,
                    "source_tar": str(archive_path),
                    "error": str(exc),
                }
                try:
                    target_path = _resolve_member_target(output_root_path, normalized_name)
                    _write_sidecar(_sidecar_path(target_path), result)
                except ValueError:
                    pass
                results.append(result)

    for missing_name in sorted(requested - seen_requested):
        failed += 1
        results.append(
            {
                "member": missing_name,
                "status": "missing",
                "source_tar": str(archive_path),
                "error": "selected tar member was not present in the archive",
            }
        )

    if remove_tar and failed == 0:
        archive_path.unlink(missing_ok=True)

    summary = {
        "requested": len(requested),
        "extracted": extracted,
        "failed": failed,
        "skipped": skipped,
    }
    return {
        "status": STATUS_COMPLETE if failed == 0 and extracted == len(requested) else "incomplete",
        "summary": summary,
        "results": results,
    }


def _download_once(
    root_path: Path,
    request: SelectedDownloadRequest,
    limits: SelectedDownloadLimits,
    client: HTTPClient,
    transferred_bytes_this_run: int,
    *,
    retry_count: int,
) -> tuple[dict[str, object], int]:
    target_path = _resolve_local_target(root_path, request.target_path)
    part_path = _part_path(target_path)
    sidecar_path = _sidecar_path(target_path)
    _ensure_under_root(root_path, part_path, label="partial")
    _ensure_under_root(root_path, sidecar_path, label="sidecar")

    try:
        _ensure_download_target_matches_payload_kind(request, target_path)
    except ValueError as exc:
        result = _result(
            root_path,
            request,
            target_path,
            status=STATUS_INVALID_IMAGE_SUFFIX,
            bytes_downloaded=0,
            retry_count=retry_count,
            error=str(exc),
        )
        _write_sidecar(sidecar_path, result)
        return result, 0

    if target_path.exists():
        try:
            _validate_downloaded_payload(target_path, request)
        except ValueError as exc:
            result = _result(
                root_path,
                request,
                target_path,
                status=STATUS_INVALID_IMAGE_PAYLOAD,
                bytes_downloaded=target_path.stat().st_size,
                retry_count=retry_count,
                error=str(exc),
            )
            _write_sidecar(sidecar_path, result)
            return result, 0
        result = _result(
            root_path,
            request,
            target_path,
            status=STATUS_COMPLETE,
            bytes_downloaded=target_path.stat().st_size,
            retry_count=retry_count,
        )
        _write_sidecar(sidecar_path, result)
        return result, 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _ensure_disk_space(target_path.parent, 0, limits.min_free_bytes)
    except OSError as exc:
        result = _result(
            root_path,
            request,
            target_path,
            status=STATUS_INSUFFICIENT_DISK,
            bytes_downloaded=0,
            retry_count=retry_count,
            required_bytes=limits.min_free_bytes,
            free_bytes=_free_bytes(target_path.parent),
            error=str(exc),
        )
        _write_sidecar(sidecar_path, result)
        return result, 0

    existing_bytes = _existing_partial_size(
        part_path,
        sidecar_path,
        request,
        expected_local_path=_relative_path(root_path, target_path),
    )
    transferred = 0
    restarted_after_bad_resume = False
    try:
        while True:
            headers = dict(request.headers or {})
            if existing_bytes:
                headers["Range"] = f"bytes={existing_bytes}-"

            with client.open(
                request.url,
                headers=headers,
                timeout=limits.timeout_seconds,
            ) as response:
                status = _response_status(response)
                if status >= 400:
                    retry_after = _response_header(response, "Retry-After")
                    result = _result(
                        root_path,
                        request,
                        target_path,
                        status=STATUS_DOWNLOAD_FAILED,
                        bytes_downloaded=existing_bytes,
                        retry_count=retry_count,
                        error=f"HTTP status {status}",
                        **_http_status_metadata(status, retry_after, limits),
                    )
                    _write_sidecar(sidecar_path, result)
                    return result, 0

                append_part = existing_bytes > 0 and status == 206
                if existing_bytes > 0 and not append_part:
                    _discard_partial(part_path)
                    existing_bytes = 0
                expected_size = _expected_final_size(response, status)
                if append_part:
                    expected_size, resume_error = _resume_validation_error(
                        response,
                        existing_bytes,
                    )
                    if resume_error:
                        if not restarted_after_bad_resume:
                            _discard_partial(part_path)
                            existing_bytes = 0
                            restarted_after_bad_resume = True
                            continue
                        result = _result(
                            root_path,
                            request,
                            target_path,
                            status=STATUS_DOWNLOAD_FAILED,
                            bytes_downloaded=existing_bytes,
                            retry_count=retry_count,
                            error=resume_error,
                        )
                        _write_sidecar(sidecar_path, result)
                        return result, 0

                if expected_size is not None:
                    expected_transfer = max(expected_size - existing_bytes, 0)
                    if transferred_bytes_this_run + expected_transfer > limits.max_transfer_bytes:
                        result = _result(
                            root_path,
                            request,
                            target_path,
                            status=STATUS_TRANSFER_CAP_EXCEEDED,
                            bytes_downloaded=existing_bytes,
                            retry_count=retry_count,
                            required_bytes=expected_transfer,
                            error="selected-image transfer cap would be exceeded",
                        )
                        _write_sidecar(sidecar_path, result)
                        return result, 0
                    try:
                        _ensure_disk_space(
                            target_path.parent,
                            expected_transfer,
                            limits.min_free_bytes,
                        )
                    except OSError as exc:
                        result = _result(
                            root_path,
                            request,
                            target_path,
                            status=STATUS_INSUFFICIENT_DISK,
                            bytes_downloaded=existing_bytes,
                            retry_count=retry_count,
                            required_bytes=expected_transfer,
                            free_bytes=_free_bytes(target_path.parent),
                            error=str(exc),
                        )
                        _write_sidecar(sidecar_path, result)
                        return result, 0

                if transferred_bytes_this_run >= limits.max_transfer_bytes:
                    result = _result(
                        root_path,
                        request,
                        target_path,
                        status=STATUS_TRANSFER_CAP_EXCEEDED,
                        bytes_downloaded=existing_bytes,
                        retry_count=retry_count,
                        required_bytes=1,
                        error="selected-image transfer cap would be exceeded",
                    )
                    _write_sidecar(sidecar_path, result)
                    return result, 0

                written = 0
                with part_path.open("ab" if append_part else "wb") as output:
                    while True:
                        if expected_size is not None and existing_bytes + written >= expected_size:
                            break

                        remaining_cap = (
                            limits.max_transfer_bytes
                            - transferred_bytes_this_run
                            - transferred
                        )
                        if remaining_cap <= 0:
                            result = _result(
                                root_path,
                                request,
                                target_path,
                                status=STATUS_TRANSFER_CAP_EXCEEDED,
                                bytes_downloaded=existing_bytes + written,
                                retry_count=retry_count,
                                required_bytes=1,
                                error="selected-image transfer cap would be exceeded",
                            )
                            _write_sidecar(sidecar_path, result)
                            return result, transferred

                        read_size = min(limits.chunk_size, remaining_cap)
                        if expected_size is None:
                            try:
                                _ensure_disk_space(
                                    target_path.parent,
                                    read_size,
                                    limits.min_free_bytes,
                                )
                            except OSError as exc:
                                result = _result(
                                    root_path,
                                    request,
                                    target_path,
                                    status=STATUS_INSUFFICIENT_DISK,
                                    bytes_downloaded=existing_bytes + written,
                                    retry_count=retry_count,
                                    required_bytes=read_size,
                                    free_bytes=_free_bytes(target_path.parent),
                                    error=str(exc),
                                )
                                _write_sidecar(sidecar_path, result)
                                return result, transferred

                        chunk = response.read(read_size)
                        if not chunk:
                            break

                        transferred += len(chunk)
                        writable_bytes = min(len(chunk), remaining_cap)
                        if writable_bytes:
                            output.write(chunk[:writable_bytes])
                            written += writable_bytes
                        if len(chunk) > remaining_cap:
                            result = _result(
                                root_path,
                                request,
                                target_path,
                                status=STATUS_TRANSFER_CAP_EXCEEDED,
                                bytes_downloaded=existing_bytes + written,
                                retry_count=retry_count,
                                required_bytes=len(chunk),
                                error="selected-image transfer cap would be exceeded",
                            )
                            _write_sidecar(sidecar_path, result)
                            return result, transferred
                break

        part_size = part_path.stat().st_size
        if expected_size is not None and part_size != expected_size:
            result = _result(
                root_path,
                request,
                target_path,
                status=STATUS_INCOMPLETE_DOWNLOAD,
                bytes_downloaded=part_size,
                retry_count=retry_count,
                error=f"downloaded {part_size} bytes; expected {expected_size} bytes",
            )
            _write_sidecar(sidecar_path, result)
            return result, transferred

        try:
            _validate_downloaded_payload(part_path, request, image_path=target_path)
        except ValueError as exc:
            result = _result(
                root_path,
                request,
                target_path,
                status=STATUS_INVALID_IMAGE_PAYLOAD,
                bytes_downloaded=part_size,
                retry_count=retry_count,
                error=str(exc),
            )
            _write_sidecar(sidecar_path, result)
            return result, transferred

        part_path.replace(target_path)
        result = _result(
            root_path,
            request,
            target_path,
            status=STATUS_COMPLETE,
            bytes_downloaded=target_path.stat().st_size,
            retry_count=retry_count,
        )
        _write_sidecar(sidecar_path, result)
        return result, transferred
    except HTTPError as exc:
        existing = part_path.stat().st_size if part_path.exists() else existing_bytes
        status = int(getattr(exc, "code", 0) or 0)
        retry_after = _response_header(exc, "Retry-After")
        result = _result(
            root_path,
            request,
            target_path,
            status=STATUS_DOWNLOAD_FAILED,
            bytes_downloaded=existing,
            retry_count=retry_count,
            error=str(exc),
            **_http_status_metadata(status, retry_after, limits),
        )
        _write_sidecar(sidecar_path, result)
        return result, transferred
    except Exception as exc:
        existing = part_path.stat().st_size if part_path.exists() else existing_bytes
        result = _result(
            root_path,
            request,
            target_path,
            status=STATUS_DOWNLOAD_FAILED,
            bytes_downloaded=existing,
            retry_count=retry_count,
            error=str(exc),
        )
        _write_sidecar(sidecar_path, result)
        return result, transferred


def _extract_member_to_target(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target_path: Path,
    output_root_path: Path,
    min_free_bytes: int,
) -> int:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("selected tar member cannot be read")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = _part_path(target_path)
    _ensure_safe_part_path(output_root_path, part_path)
    written = 0
    with source, part_path.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            _ensure_disk_space(target_path.parent, len(chunk), min_free_bytes)
            output.write(chunk)
            written += len(chunk)

    try:
        _validate_image_payload(part_path, image_path=target_path)
    except ValueError:
        raise
    part_path.replace(target_path)
    return written


def _tar_extract_failure_status(exc: Exception) -> str:
    if isinstance(exc, OSError) and "insufficient disk space" in str(exc):
        return STATUS_INSUFFICIENT_DISK
    return "extract_failed"


def _validate_limits(limits: SelectedDownloadLimits) -> None:
    if limits.max_transfer_bytes < 0:
        raise ValueError("max_transfer_bytes must not be negative")
    if limits.min_free_bytes < 0:
        raise ValueError("min_free_bytes must not be negative")
    if limits.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if limits.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if limits.max_retries < 0:
        raise ValueError("max_retries must not be negative")
    if limits.request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must not be negative")
    if limits.rate_limit_cooldown_seconds < 0:
        raise ValueError("rate_limit_cooldown_seconds must not be negative")


def _http_status_metadata(
    status: int,
    retry_after: str | None,
    limits: SelectedDownloadLimits,
) -> dict[str, object]:
    metadata: dict[str, object] = {"http_status": status}
    if status != 429:
        return metadata
    if retry_after is not None:
        metadata["retry_after"] = retry_after
    metadata["retry_delay_seconds"] = _bounded_retry_after_seconds(retry_after, limits)
    return metadata


def _bounded_retry_after_seconds(
    retry_after: str | None,
    limits: SelectedDownloadLimits,
) -> float:
    cooldown = float(limits.rate_limit_cooldown_seconds)
    parsed = _parse_retry_after_seconds(retry_after)
    if parsed is None:
        return cooldown
    return min(max(parsed, 0.0), cooldown)


def _parse_retry_after_seconds(retry_after: str | None) -> float | None:
    if retry_after is None:
        return None
    value = str(retry_after).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed_date = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
    return (parsed_date - datetime.now(timezone.utc)).total_seconds()


def _resolve_local_target(root: str | Path, local_path: str | Path) -> Path:
    local_text = str(local_path)
    windows_path = PureWindowsPath(local_text)
    posix_path = PurePosixPath(local_text.replace("\\", "/"))
    child_parts = tuple(part for part in posix_path.parts if part not in {"", "."})

    if not local_text:
        raise ValueError("target_path must not be empty")
    if windows_path.drive or windows_path.root or posix_path.is_absolute():
        raise ValueError(f"unsafe absolute target_path: {local_text}")
    if any(part == ".." for part in posix_path.parts):
        raise ValueError(f"unsafe parent traversal in target_path: {local_text}")
    if not child_parts:
        raise ValueError(f"target_path must name a child path under root: {local_text}")

    target_path = Path(root).joinpath(*child_parts)
    _ensure_under_root(root, target_path, label="target")
    return target_path


def _resolve_member_target(root: str | Path, member_name: str) -> Path:
    target_path = Path(root).joinpath(*PurePosixPath(member_name).parts)
    _ensure_under_root(root, target_path, label="tar member")
    return target_path


def _normalize_member_name(member_name: str) -> str:
    member_text = str(member_name)
    windows_path = PureWindowsPath(member_text)
    posix_path = PurePosixPath(member_text.replace("\\", "/"))
    if not member_text:
        raise ValueError("tar member name must not be empty")
    if windows_path.drive or windows_path.root or posix_path.is_absolute():
        raise ValueError(f"unsafe absolute tar member name: {member_text}")
    if any(part == ".." for part in posix_path.parts):
        raise ValueError(f"unsafe parent traversal in tar member name: {member_text}")
    parts = tuple(part for part in posix_path.parts if part not in {"", "."})
    if not parts:
        raise ValueError(f"tar member name must name a child path: {member_text}")
    return PurePosixPath(*parts).as_posix()


def _normalize_requested_member_names(
    member_names: Iterable[str],
) -> tuple[set[str], set[str]]:
    requested: set[str] = set()
    duplicates: set[str] = set()
    for name in member_names:
        normalized = _normalize_member_name(name)
        if normalized in requested:
            duplicates.add(normalized)
        requested.add(normalized)
    return requested, duplicates


def _ensure_under_root(root: str | Path, path: str | Path, *, label: str) -> None:
    root_resolved = Path(root).resolve(strict=False)
    path_resolved = Path(path).resolve(strict=False)
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes selected-image root: {path}") from exc


def _ensure_safe_part_path(root: str | Path, part_path: Path) -> None:
    _ensure_under_root(root, part_path, label="partial")
    if part_path.is_symlink():
        raise ValueError(f"partial path is an existing symlink: {part_path}")


def _ensure_image_suffix(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"selected image must end with one of {supported}: {path.name}")


def _ensure_archive_suffix(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in ARCHIVE_SUFFIXES:
        supported = ", ".join(sorted(ARCHIVE_SUFFIXES))
        raise ValueError(f"selected archive must end with one of {supported}: {path.name}")


def _download_payload_kind(request: SelectedDownloadRequest) -> str:
    payload_kind = str(request.payload_kind or PAYLOAD_KIND_IMAGE).strip().lower()
    if payload_kind == PAYLOAD_KIND_IMAGE:
        return PAYLOAD_KIND_IMAGE
    if payload_kind in ARCHIVE_PAYLOAD_KINDS:
        return PAYLOAD_KIND_ARCHIVE
    raise ValueError(f"unsupported selected download payload_kind: {request.payload_kind}")


def _ensure_download_target_matches_payload_kind(
    request: SelectedDownloadRequest,
    path: Path,
) -> None:
    payload_kind = _download_payload_kind(request)
    if payload_kind == PAYLOAD_KIND_ARCHIVE:
        _ensure_archive_suffix(path)
        return
    _ensure_image_suffix(path)


def _validate_image_payload(path: Path, *, image_path: Path | None = None) -> None:
    _ensure_image_suffix(image_path or path)
    prefix = path.read_bytes()[:512].lstrip().lower()
    if any(prefix.startswith(marker.lower()) for marker in HTML_XML_PREFIXES):
        raise ValueError("retained image begins with an obvious HTML/XML error payload")


def _validate_downloaded_payload(
    path: Path,
    request: SelectedDownloadRequest,
    *,
    image_path: Path | None = None,
) -> None:
    if _download_payload_kind(request) == PAYLOAD_KIND_IMAGE:
        _validate_image_payload(path, image_path=image_path)


def _ensure_disk_space(path: Path, incoming_bytes: int, min_free_bytes: int) -> None:
    free_bytes = _free_bytes(path)
    incoming = max(incoming_bytes, 0)
    if free_bytes <= min_free_bytes:
        raise OSError(
            f"insufficient disk space: free {free_bytes} bytes; "
            f"required more than {min_free_bytes} bytes"
        )
    required = incoming + min_free_bytes
    if free_bytes < required:
        raise OSError(
            f"insufficient disk space: free {free_bytes} bytes; required {required} bytes"
        )


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(disk_usage(probe).free)


def _part_path(target_path: str | Path) -> Path:
    target = Path(target_path)
    return target.with_suffix(target.suffix + ".part")


def _sidecar_path(target_path: str | Path) -> Path:
    target = Path(target_path)
    return target.with_suffix(target.suffix + ".selected_download.json")


def _existing_partial_size(
    part_path: Path,
    sidecar_path: Path,
    request: SelectedDownloadRequest,
    *,
    expected_local_path: str | None = None,
) -> int:
    if not part_path.exists() and not part_path.is_symlink():
        return 0
    if part_path.is_symlink():
        _discard_partial(part_path)
        return 0

    existing_bytes = part_path.stat().st_size
    if existing_bytes <= 0:
        return 0
    if not _partial_sidecar_matches(
        sidecar_path,
        request,
        existing_bytes,
        expected_local_path=expected_local_path,
    ):
        _discard_partial(part_path)
        return 0
    return existing_bytes


def _partial_sidecar_matches(
    sidecar_path: Path,
    request: SelectedDownloadRequest,
    existing_bytes: int,
    *,
    expected_local_path: str | None = None,
) -> bool:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False
    if payload.get("status") not in RESUMABLE_PARTIAL_STATUSES:
        return False
    if payload.get("dataset") != request.dataset:
        return False
    if payload.get("provider") != request.provider:
        return False
    if payload.get("image_id") != request.image_id:
        return False
    if payload.get("source_url") != _redact_url(request.url):
        return False
    if expected_local_path is None:
        expected_local_path = str(request.target_path).replace("\\", "/")
    matched_path = False
    for path_key in ("local_path", "target_path"):
        recorded_path = payload.get(path_key)
        if recorded_path is None:
            continue
        if recorded_path != expected_local_path:
            return False
        matched_path = True
    if not matched_path:
        return False
    recorded_bytes = payload.get("bytes_downloaded")
    if type(recorded_bytes) is not int:
        return False
    return recorded_bytes == existing_bytes


def _discard_partial(part_path: Path) -> None:
    try:
        part_path.unlink()
    except FileNotFoundError:
        pass


def _write_sidecar_for_request(
    root_path: Path,
    request: SelectedDownloadRequest,
    result: Mapping[str, object],
) -> None:
    target_path = _resolve_local_target(root_path, request.target_path)
    _write_sidecar(_sidecar_path(target_path), result)


def _write_sidecar(sidecar_path: Path, result: Mapping[str, object]) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )
    temp_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(sidecar_path)


def _result(
    root_path: Path,
    request: SelectedDownloadRequest,
    target_path: Path,
    *,
    status: str,
    bytes_downloaded: int,
    retry_count: int,
    error: str | None = None,
    required_bytes: int | None = None,
    free_bytes: int | None = None,
    http_status: int | None = None,
    retry_after: str | None = None,
    retry_delay_seconds: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset": request.dataset,
        "provider": request.provider,
        "image_id": request.image_id,
        "local_path": _relative_path(root_path, target_path),
        "status": status,
        "bytes_downloaded": bytes_downloaded,
        "source_url": _redact_url(request.url),
        "headers": _safe_headers(request.headers),
        "retry_count": retry_count,
    }
    if error is not None:
        payload["error"] = _redact_sensitive_text(error, request)
    if required_bytes is not None:
        payload["required_bytes"] = required_bytes
    if free_bytes is not None:
        payload["free_bytes"] = free_bytes
    if http_status is not None:
        payload["http_status"] = http_status
    if retry_after is not None:
        payload["retry_after"] = retry_after
    if retry_delay_seconds is not None:
        payload["retry_delay_seconds"] = retry_delay_seconds
    return payload


def _skipped_result(root_path: Path, request: SelectedDownloadRequest) -> dict[str, object]:
    target_path = _resolve_local_target(root_path, request.target_path)
    return _result(
        root_path,
        request,
        target_path,
        status=STATUS_SKIPPED_TRANSFER_CAP,
        bytes_downloaded=0,
        retry_count=0,
        error="selected-image transfer cap was reached by an earlier request",
    )


def _safe_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in dict(headers or {}).items():
        if _is_sensitive_header_key(str(key)):
            continue
        safe[str(key)] = str(value)
    return safe


def _redact_sensitive_text(text: str, request: SelectedDownloadRequest) -> str:
    redacted = str(text)
    for key, value in dict(request.headers or {}).items():
        if _is_sensitive_header_key(str(key)) and value:
            redacted = redacted.replace(str(value), "<redacted>")
    parts = urlsplit(request.url)
    for value in (parts.username, parts.password, parts.fragment):
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if _is_sensitive_query_key(key) and value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    safe_query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if _is_sensitive_query_key(key):
            safe_query.append((key, "<redacted>"))
        else:
            safe_query.append((key, value))
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(safe_query), ""))


def _normalized_key_parts(key: str) -> tuple[str, set[str]]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    normalized = re.sub(r"[^a-z0-9]+", "_", snake.lower()).strip("_")
    return normalized, set(part for part in normalized.split("_") if part)


def _is_sensitive_header_key(key: str) -> bool:
    normalized, tokens = _normalized_key_parts(key)
    if normalized in {"apikey", "api_key", "access_key", "key", "cookie", "set_cookie"}:
        return True
    if tokens & SENSITIVE_HEADER_TOKENS:
        return True
    if {"api", "key"}.issubset(tokens):
        return True
    if {"access", "key"}.issubset(tokens):
        return True
    return False


def _is_sensitive_query_key(key: str) -> bool:
    normalized, tokens = _normalized_key_parts(key)
    if normalized in SENSITIVE_QUERY_KEYS:
        return True
    if any(token in normalized for token in SENSITIVE_QUERY_TOKENS):
        return True
    if tokens & SENSITIVE_QUERY_TOKENS:
        return True
    if {"api", "key"}.issubset(tokens):
        return True
    if {"access", "key"}.issubset(tokens):
        return True
    return False


def _download_summary(
    results: Iterable[Mapping[str, object]],
    *,
    requested: int,
) -> dict[str, int]:
    complete = 0
    failed = 0
    skipped = 0
    for result in results:
        status = result.get("status")
        if status == STATUS_COMPLETE:
            complete += 1
        elif status == STATUS_SKIPPED_TRANSFER_CAP:
            skipped += 1
        else:
            failed += 1
    return {
        "requested": requested,
        "complete": complete,
        "failed": failed,
        "skipped": skipped,
    }


def _should_retry(result: Mapping[str, object]) -> bool:
    return result.get("status") in {
        STATUS_DOWNLOAD_FAILED,
        STATUS_INCOMPLETE_DOWNLOAD,
    }


def _retry_delay_seconds(result: Mapping[str, object]) -> float:
    if result.get("http_status") != 429:
        return 0.0
    value = result.get("retry_delay_seconds")
    if value is None:
        return 0.0
    return float(value)


def _relative_path(root_path: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root_path.resolve(strict=False)).as_posix()


def _response_status(response) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def _response_header(response, name: str) -> str | None:
    wanted = name.lower()
    headers = getattr(response, "headers", None)
    if headers is not None:
        if hasattr(headers, "get"):
            value = headers.get(name) or headers.get(name.lower())
            if value is not None:
                return str(value)
        if hasattr(headers, "items"):
            for key, value in headers.items():
                if str(key).lower() == wanted:
                    return str(value)
    if hasattr(response, "info"):
        info = response.info()
        if hasattr(info, "get"):
            value = info.get(name) or info.get(name.lower())
            if value is not None:
                return str(value)
    return None


def _response_content_length(response) -> int | None:
    raw = _response_header(response, "Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def _expected_final_size(response, status: int) -> int | None:
    if status == 206:
        content_range = _parse_content_range(_response_header(response, "Content-Range"))
        if content_range is None:
            return None
        return content_range[2]
    if status == 200:
        return _response_content_length(response)
    return None


def _resume_validation_error(
    response,
    existing_bytes: int,
) -> tuple[int | None, str | None]:
    content_range = _parse_content_range(_response_header(response, "Content-Range"))
    if content_range is None:
        return None, "missing or malformed Content-Range for resumed response"

    start, end, total = content_range
    if start != existing_bytes:
        return total, f"resumed Content-Range starts at {start}; expected {existing_bytes}"
    if end != total - 1:
        return total, f"resumed Content-Range ends at {end}; expected {total - 1}"

    content_length = _response_content_length(response)
    expected_length = end - start + 1
    if content_length is not None and content_length != expected_length:
        return total, (
            f"Content-Length {content_length} does not match resumed range "
            f"length {expected_length}"
        )
    return total, None


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    prefix = "bytes "
    if not value.startswith(prefix) or "/" not in value:
        return None
    range_part, total_part = value[len(prefix) :].split("/", 1)
    if "-" not in range_part:
        return None
    start_part, end_part = range_part.split("-", 1)
    try:
        start = int(start_part)
        end = int(end_part)
        total = int(total_part)
    except ValueError:
        return None
    if start < 0 or end < start or total <= end:
        return None
    return start, end, total


__all__ = [
    "SelectedDownloadLimits",
    "SelectedDownloadRequest",
    "execute_selected_downloads",
    "extract_selected_tar_members",
]
