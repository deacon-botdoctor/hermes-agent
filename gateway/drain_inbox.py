"""Durable inbound mailbox for messages accepted during gateway drain."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, IO, Iterator

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.status import _release_file_lock, _try_acquire_file_lock
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_FILE_NAME = "drain-queued-messages.jsonl"
_LOCK_FILE_NAME = ".drain-queued-messages.lock"
_REPLAY_LOCK_FILE_NAME = ".drain-queued-messages.replay.lock"
_LOCK_TIMEOUT_SECONDS = 10.0
_QUEUED = "queued"
_CLAIMED = "claimed"
_AMBIGUOUS = "ambiguous"
_COMPLETED = "completed"
_HANDLED = "handled"
_FAILED = "failed"
_MAX_METADATA_BYTES = 64 * 1024
_MAX_PENDING_RECORDS = 64
_MAX_QUARANTINED_RECORDS = 64
_lock = threading.Lock()
_replay_lock = threading.Lock()
_replay_lock_pid = os.getpid()


class _PostReplaceError(OSError):
    pass


def _capacity_record_count(
    rows: list[tuple[str, dict[str, Any] | None]],
) -> int:
    return sum(
        1
        for _, parsed in rows
        if parsed is None
        or str(parsed.get("state") or _QUEUED) != _AMBIGUOUS
    )


def _has_record_capacity(
    rows: list[tuple[str, dict[str, Any] | None]],
) -> bool:
    return (
        _capacity_record_count(rows) < _MAX_PENDING_RECORDS
        and len(rows)
        < _MAX_PENDING_RECORDS + _MAX_QUARANTINED_RECORDS
    )


def _replace_with_ambiguous(
    path: Path,
    rows: list[tuple[str, dict[str, Any] | None]],
    index: int,
    record: dict[str, Any],
) -> None:
    ambiguous = dict(record)
    ambiguous["state"] = _AMBIGUOUS
    ambiguous["recovery_disposition"] = _AMBIGUOUS
    replacements = [
        (
            json.dumps(ambiguous, ensure_ascii=False, default=str)
            if row_index == index
            else raw
        )
        for row_index, (raw, _) in enumerate(rows)
    ]
    for attempt in range(2):
        try:
            _replace_rows(path, replacements)
            return
        except _PostReplaceError:
            try:
                visible = _read_rows(path)
            except Exception:
                if attempt:
                    raise
                continue
            if any(
                parsed is not None
                and parsed.get("queue_id") == ambiguous["queue_id"]
                and parsed.get("state") == _AMBIGUOUS
                and parsed.get("recovery_disposition") == _AMBIGUOUS
                for _, parsed in visible
            ):
                return
            if attempt:
                raise
        except Exception:
            if attempt:
                raise


@dataclass(frozen=True)
class ProducerReplayLease:
    handle: IO[str]
    path: Path
    producer_token: str
    previous_token: str


def inbox_path(hermes_home: Path | None = None) -> Path:
    """Return the profile-scoped durable drain inbox path."""
    return (hermes_home or get_hermes_home()) / "state" / _FILE_NAME


def _event_identity(event: MessageEvent, session_key: str) -> str:
    message_id = (
        "" if event.message_id is None else str(event.message_id).strip()
    )
    platform_update_id = (
        ""
        if event.platform_update_id is None
        else str(event.platform_update_id).strip()
    )
    if message_id:
        payload = {
            "session_key": session_key,
            "message_id": message_id,
        }
    elif platform_update_id:
        payload = {
            "session_key": session_key,
            "platform_update_id": platform_update_id,
        }
    else:
        raise ValueError("stable platform message identity is required")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def event_queue_id(event: MessageEvent, session_key: str) -> str:
    """Return the stable durable-inbox identity for an event."""
    return _event_identity(event, session_key)


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("event metadata must be an object")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("event metadata must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("event metadata is too large")
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise ValueError("event metadata must be an object")
    return normalized


def _record_for_event(
    event: MessageEvent,
    session_key: str,
    reason: str,
) -> dict[str, Any]:
    pre_dispatch_attempted = getattr(
        event,
        "_hermes_pre_gateway_dispatch_attempted",
        False,
    )
    if not isinstance(pre_dispatch_attempted, bool):
        raise ValueError("pre-dispatch receipt must be a boolean")
    if not isinstance(event.internal, bool):
        raise ValueError("internal receipt must be a boolean")
    if not isinstance(event.durable_ingress, bool):
        raise ValueError("durable-ingress receipt must be a boolean")
    return {
        "schema": _SCHEMA_VERSION,
        "queue_id": _event_identity(event, session_key),
        "state": _QUEUED,
        "pre_dispatch_attempted": pre_dispatch_attempted,
        "queued_at": datetime.now().astimezone().isoformat(),
        "reason": reason,
        "session_key": session_key,
        "source": event.source.to_dict(),
        "text": event.text,
        "message_type": event.message_type.value,
        "message_id": event.message_id,
        "platform_update_id": event.platform_update_id,
        "media_urls": list(event.media_urls or []),
        "media_types": list(event.media_types or []),
        "reply_to_message_id": event.reply_to_message_id,
        "reply_to_text": event.reply_to_text,
        "reply_to_author_id": event.reply_to_author_id,
        "reply_to_author_name": event.reply_to_author_name,
        "reply_to_is_own_message": event.reply_to_is_own_message,
        "auto_skill": event.auto_skill,
        "channel_prompt": event.channel_prompt,
        "channel_context": event.channel_context,
        "internal": event.internal and event.durable_ingress,
        "metadata": _normalize_metadata(event.metadata),
        "timestamp": event.timestamp.isoformat(),
    }


def _validate_private_file(
    path: Path,
    file_stat: os.stat_result,
) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise PermissionError(f"Durable inbox state is not a regular file: {path}")
    if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
        raise PermissionError(f"Durable inbox state has the wrong owner: {path}")
    if os.name != "nt" and stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise PermissionError(f"Durable inbox state is not private: {path}")


def _open_trusted_directory(path: Path) -> os.stat_result:
    if os.name == "nt":  # HERMES_DURABLE_DRAIN_COMPATIBILITY_v1
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise PermissionError(f"Durable inbox directory is a symlink: {path}")
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if reparse_flag and (
            getattr(path_stat, "st_file_attributes", 0) & reparse_flag
        ):
            raise PermissionError(
                f"Durable inbox directory is a reparse point: {path}"
            )
        if not stat.S_ISDIR(path_stat.st_mode):
            raise PermissionError(
                f"Durable inbox directory is not a directory: {path}"
            )
        return path_stat

    # Linux PrivateTmp mount namespaces can deny a read-open of their synthetic
    # root even though metadata traversal is allowed. O_PATH is sufficient for
    # fstat-based trust validation and avoids requiring directory read access.
    flags = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        directory_stat = os.fstat(fd)
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise PermissionError(f"Durable inbox directory is a symlink: {path}")
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise PermissionError(
                f"Durable inbox directory is not a directory: {path}"
            )
        if (
            directory_stat.st_dev,
            directory_stat.st_ino,
        ) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise PermissionError(
                f"Durable inbox directory changed while opening: {path}"
            )
        return directory_stat
    finally:
        os.close(fd)


def _ensure_trusted_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    chain = list(reversed((absolute, *absolute.parents)))
    parent_stat: os.stat_result | None = None
    target_stat: os.stat_result | None = None

    for component in chain:
        try:
            component_stat = _open_trusted_directory(component)
        except FileNotFoundError:
            if (
                parent_stat is not None
                and os.name != "nt"
                and stat.S_IMODE(parent_stat.st_mode) & 0o022
                and not parent_stat.st_mode & stat.S_ISVTX
            ):
                raise PermissionError(
                    "Durable inbox directory trust chain is writable by other users: "
                    f"{component.parent}"
                )
            try:
                os.mkdir(component, mode=0o700)
            except FileExistsError:
                pass
            component_stat = _open_trusted_directory(component)

        if parent_stat is not None and os.name != "nt":
            parent_mode = stat.S_IMODE(parent_stat.st_mode)
            if parent_mode & 0o022:
                if (
                    not parent_stat.st_mode & stat.S_ISVTX
                    or not hasattr(os, "getuid")
                    or component_stat.st_uid != os.getuid()  # windows-footgun: ok — guarded above
                ):
                    raise PermissionError(
                        "Durable inbox directory trust chain is writable by "
                        f"other users: {component.parent}"
                    )
        # systemd PrivateTmp presents its protected namespace prefix (commonly
        # / and /home) as uid 65534. It is safe to traverse while it is not
        # group/other-writable; the actual Hermes state directory must still be
        # owned by the gateway user below.
        synthetic_namespace_ancestor = (
            os.name != "nt"
            and component_stat.st_uid == 65534
            and not stat.S_IMODE(component_stat.st_mode) & 0o022
        )
        if (
            os.name != "nt"
            and hasattr(os, "getuid")
            and component_stat.st_uid not in {0, os.getuid()}  # windows-footgun: ok — guarded above
            and not synthetic_namespace_ancestor
        ):
            raise PermissionError(
                f"Durable inbox directory trust chain has an untrusted owner: {component}"
            )
        parent_stat = component_stat
        target_stat = component_stat

    if target_stat is None:
        raise PermissionError(f"Durable inbox directory is unavailable: {path}")
    if hasattr(os, "getuid") and target_stat.st_uid != os.getuid():
        raise PermissionError(f"Durable inbox directory has the wrong owner: {path}")
    if os.name != "nt" and stat.S_IMODE(target_stat.st_mode) & 0o022:
        raise PermissionError(
            f"Durable inbox directory is writable by other users: {path}"
        )


def _open_private_file(
    path: Path,
    flags: int,
    *,
    create: bool,
) -> tuple[int, bool]:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    created = False
    if create:
        try:
            fd = os.open(
                path,
                flags | os.O_CREAT | os.O_EXCL | no_follow,
                0o600,
            )
            created = True
        except FileExistsError:
            fd = os.open(path, flags | no_follow)
    else:
        fd = os.open(path, flags | no_follow)

    try:
        if created and os.name != "nt":
            os.fchmod(fd, 0o600)
        file_stat = os.fstat(fd)
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise PermissionError(f"Durable inbox state is a symlink: {path}")
        if (
            file_stat.st_dev,
            file_stat.st_ino,
        ) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise PermissionError(f"Durable inbox state changed while opening: {path}")
        _validate_private_file(path, file_stat)
    except BaseException:
        os.close(fd)
        raise
    return fd, created


def _read_rows(path: Path) -> list[tuple[str, dict[str, Any] | None]]:
    try:
        fd, _ = _open_private_file(path, os.O_RDONLY, create=False)
    except FileNotFoundError:
        return []
    rows: list[tuple[str, dict[str, Any] | None]] = []
    with os.fdopen(
        fd,
        "r",
        encoding="utf-8",
        errors="surrogateescape",
    ) as handle:
        contents = handle.read()
    raw_lines = contents.splitlines()
    trailing_fragment = bool(contents) and not contents.endswith("\n")
    for index, raw in enumerate(raw_lines):
        if trailing_fragment and index == len(raw_lines) - 1:
            logger.warning(
                "Incomplete durable drain-inbox tail isolated from recovery"
            )
            continue
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("Malformed durable drain-inbox row retained for recovery")
            value = None
        rows.append((raw, value if isinstance(value, dict) else None))
    return rows


@contextmanager
def _locked_inbox(path: Path) -> Iterator[IO[str]]:
    """Serialize inbox read-modify-write cycles across threads and processes."""
    _ensure_trusted_directory(path.parent)
    lock_path = path.parent / _LOCK_FILE_NAME
    with _lock:
        fd, _ = _open_private_file(lock_path, os.O_RDWR, create=True)
        handle = os.fdopen(fd, "a+", encoding="utf-8")
        acquired = False
        try:
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while not _try_acquire_file_lock(handle):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for durable inbox lock: {lock_path}")
                time.sleep(0.01)
            acquired = True
            yield handle
        finally:
            if acquired:
                _release_file_lock(handle)
            handle.close()


def acquire_replay_lease(
    path: Path | None = None,
    *,
    cancellation_event: threading.Event | None = None,
) -> IO[str]:
    global _replay_lock, _replay_lock_pid
    current_pid = os.getpid()
    if _replay_lock_pid != current_pid:
        _replay_lock = threading.Lock()
        _replay_lock_pid = current_pid
    path = path or inbox_path()
    _ensure_trusted_directory(path.parent)
    lock_path = path.parent / _REPLAY_LOCK_FILE_NAME
    while not _replay_lock.acquire(timeout=0.01):
        if cancellation_event is not None and cancellation_event.is_set():
            raise InterruptedError("durable replay lease acquisition cancelled")
    handle: IO[str] | None = None
    try:
        if cancellation_event is not None and cancellation_event.is_set():
            raise InterruptedError("durable replay lease acquisition cancelled")
        fd, _ = _open_private_file(lock_path, os.O_RDWR, create=True)
        handle = os.fdopen(fd, "a+", encoding="utf-8")
        while not _try_acquire_file_lock(handle):
            if cancellation_event is not None and cancellation_event.is_set():
                raise InterruptedError("durable replay lease acquisition cancelled")
            time.sleep(0.01)
        if cancellation_event is not None and cancellation_event.is_set():
            raise InterruptedError("durable replay lease acquisition cancelled")
        return handle
    except BaseException:
        if handle is not None:
            handle.close()
        _replay_lock.release()
        raise


def release_replay_lease(
    lease: IO[str] | ProducerReplayLease,
) -> None:
    handle = lease.handle if isinstance(lease, ProducerReplayLease) else lease
    try:
        _release_file_lock(handle)
    finally:
        handle.close()
        _replay_lock.release()


def acquire_producer_replay_lease(
    producer_token: str,
    path: Path | None = None,
    *,
    cancellation_event: threading.Event | None = None,
) -> ProducerReplayLease:
    if not producer_token:
        raise ValueError("producer token is required")
    path = path or inbox_path()
    replay_lease = acquire_replay_lease(
        path,
        cancellation_event=cancellation_event,
    )
    try:
        with _locked_inbox(path) as handle:
            handle.seek(0)
            previous_token = handle.read().strip()
            if cancellation_event is not None and cancellation_event.is_set():
                raise InterruptedError("durable producer handoff cancelled")
            handle.seek(0)
            handle.truncate()
            handle.write(producer_token)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        release_replay_lease(replay_lease)
        raise
    lease = ProducerReplayLease(
        handle=replay_lease,
        path=path,
        producer_token=producer_token,
        previous_token=previous_token,
    )
    if cancellation_event is not None and cancellation_event.is_set():
        cancel_producer_replay_lease(lease)
        raise InterruptedError("durable producer handoff cancelled")
    return lease


def cancel_producer_replay_lease(lease: ProducerReplayLease) -> None:
    while True:
        try:
            with _locked_inbox(lease.path) as handle:
                while True:
                    try:
                        handle.seek(0)
                        current_token = handle.read().strip()
                        if current_token not in {
                            lease.producer_token,
                            lease.previous_token,
                        }:
                            raise RuntimeError(
                                "durable producer ownership changed during rollback"
                            )
                        handle.seek(0)
                        handle.truncate()
                        handle.write(lease.previous_token)
                        handle.flush()
                        os.fsync(handle.fileno())
                        break
                    except Exception:
                        logger.exception(
                            "Failed to restore durable inbox producer ownership"
                        )
                        time.sleep(0.05)
            break
        except Exception:
            logger.exception("Failed to lock durable inbox for producer restoration")
            time.sleep(0.05)
    release_replay_lease(lease)


def claim_producer(producer_token: str, path: Path | None = None) -> None:
    """Make one gateway process the only accepted inbox producer."""
    replay_lease = acquire_producer_replay_lease(producer_token, path)
    release_replay_lease(replay_lease)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the platform supports it."""
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def persist_event_result(
    event: MessageEvent,
    session_key: str,
    *,
    reason: str,
    path: Path | None = None,
    producer_token: str | None = None,
) -> tuple[str | None, str]:
    """Write *event* durably before the gateway acknowledges acceptance."""
    if event.source is None:
        return None, _FAILED
    if str(event.text or "").lstrip().startswith("/"):
        return None, _FAILED
    if not str(event.text or "").strip() and not event.media_urls:
        return None, _FAILED

    try:
        record = _record_for_event(event, session_key, reason)
        queue_id = str(record["queue_id"])
        path = path or inbox_path()
        with _locked_inbox(path) as handle:
            if producer_token is not None:
                handle.seek(0)
                if handle.read().strip() != producer_token:
                    raise RuntimeError("durable inbox producer ownership changed")
            rows = _read_rows(path)
            for index, (_, parsed) in enumerate(rows):
                if parsed is None or parsed.get("queue_id") != queue_id:
                    continue
                state = str(parsed.get("state") or _QUEUED)
                if state != _QUEUED:
                    if state in {
                        _CLAIMED,
                        _AMBIGUOUS,
                        _COMPLETED,
                        _HANDLED,
                    }:
                        return queue_id, state
                    raise RuntimeError(f"invalid duplicate drain-inbox state: {state}")
                existing_receipt = parsed.get("pre_dispatch_attempted", False)
                if not isinstance(existing_receipt, bool):
                    raise ValueError("pre-dispatch receipt must be a boolean")
                if existing_receipt or not record["pre_dispatch_attempted"]:
                    return queue_id, _QUEUED
                updated = dict(parsed)
                updated.update(record)
                updated["queued_at"] = parsed.get(
                    "queued_at",
                    record["queued_at"],
                )
                updated["reason"] = parsed.get("reason", record["reason"])
                replacement = json.dumps(
                    updated,
                    ensure_ascii=False,
                    default=str,
                )
                replacements = [
                    replacement if row_index == index else raw
                    for row_index, (raw, _) in enumerate(rows)
                ]
                claimed = dict(updated)
                claimed["state"] = _CLAIMED
                claimed["state_changed_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                claimed_rows = [
                    (
                        json.dumps(
                            claimed,
                            ensure_ascii=False,
                            default=str,
                        )
                        if row_index == index
                        else raw
                    )
                    for row_index, (raw, _) in enumerate(rows)
                ]
                _replace_rows(path, claimed_rows)
                try:
                    _replace_rows(path, replacements)
                except _PostReplaceError:
                    return queue_id, _QUEUED
                except Exception:
                    return queue_id, _CLAIMED
                return queue_id, _QUEUED
            if not _has_record_capacity(rows):
                raise OverflowError("durable drain inbox is at capacity")
            claimed = dict(record)
            claimed["state"] = _CLAIMED
            claimed["state_changed_at"] = (
                datetime.now().astimezone().isoformat()
            )
            prior_rows = [raw for raw, _ in rows]
            _replace_rows(
                path,
                [
                    *prior_rows,
                    json.dumps(
                        claimed,
                        ensure_ascii=False,
                        default=str,
                    ),
                ],
            )
            try:
                _replace_rows(
                    path,
                    [
                        *prior_rows,
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        ),
                    ],
                )
            except _PostReplaceError:
                return queue_id, _QUEUED
            except Exception:
                return queue_id, _CLAIMED
    except Exception as exc:
        logger.warning(
            "Failed to persist drain-time message for %s: %s",
            session_key or "?",
            exc,
        )
        return None, _FAILED
    logger.info(
        "Persisted drain-time message %s for %s (%s)",
        queue_id,
        session_key or "?",
        reason,
    )
    return queue_id, _QUEUED
def claim_pre_dispatch_event_result(
    event: MessageEvent,
    session_key: str,
    *,
    reason: str,
    path: Path | None = None,
    producer_token: str | None = None,
) -> tuple[str | None, str, bool]:
    try:
        record = _record_for_event(event, session_key, reason)
        queue_id = str(record["queue_id"])
        path = path or inbox_path()
        with _locked_inbox(path) as handle:
            if producer_token is not None:
                handle.seek(0)
                if handle.read().strip() != producer_token:
                    raise RuntimeError("durable inbox producer ownership changed")
            rows = _read_rows(path)
            for index, (_, parsed) in enumerate(rows):
                if parsed is None or parsed.get("queue_id") != queue_id:
                    continue
                state = str(parsed.get("state") or _QUEUED)
                attempted = parsed.get("pre_dispatch_attempted", False)
                if not isinstance(attempted, bool):
                    raise ValueError("pre-dispatch receipt must be a boolean")
                if state != _QUEUED or attempted:
                    return queue_id, state, False
                claimed = dict(parsed)
                claimed["state"] = _CLAIMED
                claimed["state_changed_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                replacements = [
                    (
                        json.dumps(claimed, ensure_ascii=False, default=str)
                        if row_index == index
                        else raw
                    )
                    for row_index, (raw, _) in enumerate(rows)
                ]
                _replace_rows(path, replacements)
                return queue_id, _CLAIMED, True
            if not _has_record_capacity(rows):
                raise OverflowError("durable drain inbox is at capacity")
            record["state"] = _CLAIMED
            record["state_changed_at"] = datetime.now().astimezone().isoformat()
            _replace_rows(
                path,
                [
                    *(raw for raw, _ in rows),
                    json.dumps(record, ensure_ascii=False, default=str),
                ],
            )
            return queue_id, _CLAIMED, True
    except Exception as exc:
        logger.warning(
            "Failed to claim pre-dispatch message for %s: %s",
            session_key or "?",
            exc,
        )
        return None, _FAILED, False


def record_pre_dispatch_attempt_result(
    event: MessageEvent,
    session_key: str,
    *,
    path: Path | None = None,
    producer_token: str | None = None,
) -> tuple[str | None, str]:
    try:
        record = _record_for_event(
            event,
            session_key,
            "startup-restore-pre-dispatch",
        )
        if record["pre_dispatch_attempted"] is not True:
            raise ValueError("pre-dispatch attempt is not complete")
        queue_id = str(record["queue_id"])
        path = path or inbox_path()
        with _locked_inbox(path) as handle:
            if producer_token is not None:
                handle.seek(0)
                if handle.read().strip() != producer_token:
                    raise RuntimeError("durable inbox producer ownership changed")
            rows = _read_rows(path)
            for index, (_, parsed) in enumerate(rows):
                if parsed is None or parsed.get("queue_id") != queue_id:
                    continue
                state = str(parsed.get("state") or _QUEUED)
                attempted = parsed.get("pre_dispatch_attempted", False)
                if not isinstance(attempted, bool):
                    raise ValueError("pre-dispatch receipt must be a boolean")
                if state != _CLAIMED or attempted:
                    return queue_id, state
                updated = dict(parsed)
                updated.update(record)
                updated["state"] = _CLAIMED
                updated["queued_at"] = parsed.get(
                    "queued_at",
                    record["queued_at"],
                )
                updated["reason"] = parsed.get("reason", record["reason"])
                updated["state_changed_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                replacements = [
                    (
                        json.dumps(updated, ensure_ascii=False, default=str)
                        if row_index == index
                        else raw
                    )
                    for row_index, (raw, _) in enumerate(rows)
                ]
                try:
                    _replace_rows(path, replacements)
                except Exception:
                    try:
                        _replace_rows(path, replacements)
                    except Exception:
                        _replace_with_ambiguous(
                            path,
                            rows,
                            index,
                            updated,
                        )
                        return queue_id, _AMBIGUOUS
                return queue_id, _CLAIMED
            raise RuntimeError("pre-dispatch claim is missing")
    except Exception as exc:
        logger.warning(
            "Failed to record pre-dispatch attempt for %s: %s",
            session_key or "?",
            exc,
        )
        return None, _FAILED


def finalize_pre_dispatch_event_result(
    event: MessageEvent,
    session_key: str,
    *,
    handled: bool,
    reason: str,
    path: Path | None = None,
    producer_token: str | None = None,
) -> tuple[str | None, str]:
    final_state = _HANDLED if handled else _QUEUED
    control_command = str(event.text or "").lstrip().startswith("/")
    try:
        record = _record_for_event(event, session_key, reason)
        record["state"] = final_state
        queue_id = str(record["queue_id"])
        path = path or inbox_path()
        with _locked_inbox(path) as handle:
            if producer_token is not None:
                handle.seek(0)
                if handle.read().strip() != producer_token:
                    raise RuntimeError("durable inbox producer ownership changed")
            rows = _read_rows(path)
            for index, (_, parsed) in enumerate(rows):
                if parsed is None or parsed.get("queue_id") != queue_id:
                    continue
                state = str(parsed.get("state") or _QUEUED)
                if state == final_state:
                    return queue_id, state
                if state != _CLAIMED:
                    return queue_id, state
                if control_command and not handled:
                    return queue_id, _CLAIMED
                updated = dict(parsed)
                updated.update(record)
                updated["queued_at"] = parsed.get(
                    "queued_at",
                    record["queued_at"],
                )
                updated["reason"] = parsed.get("reason", record["reason"])
                rejection_disposition = (
                    handled
                    and reason == "startup-restore-unauthorized-rejected"
                )
                if rejection_disposition:
                    updated["recovery_disposition"] = "unauthorized-rejected"
                claimed = dict(updated)
                claimed["state"] = _CLAIMED
                claimed["state_changed_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                replacements = [
                    (
                        json.dumps(updated, ensure_ascii=False, default=str)
                        if row_index == index
                        else raw
                    )
                    for row_index, (raw, _) in enumerate(rows)
                ]
                claimed_rows = [
                    (
                        json.dumps(claimed, ensure_ascii=False, default=str)
                        if row_index == index
                        else raw
                    )
                    for row_index, (raw, _) in enumerate(rows)
                ]
                if (
                    rejection_disposition
                    and parsed.get("recovery_disposition")
                    != "unauthorized-rejected"
                ):
                    try:
                        _replace_rows(path, claimed_rows)
                    except Exception:
                        try:
                            _replace_rows(path, claimed_rows)
                        except Exception:
                            _replace_with_ambiguous(
                                path,
                                rows,
                                index,
                                claimed,
                            )
                            return queue_id, _AMBIGUOUS
                try:
                    _replace_rows(path, replacements)
                except _PostReplaceError:
                    try:
                        _replace_rows(path, claimed_rows)
                    except Exception:
                        return queue_id, final_state
                    return queue_id, _CLAIMED
                except Exception:
                    return queue_id, _CLAIMED
                return queue_id, final_state
            raise RuntimeError("pre-dispatch claim is missing")
    except Exception as exc:
        logger.warning(
            "Failed to finalize pre-dispatch message for %s: %s",
            session_key or "?",
            exc,
        )
        return None, _FAILED


def persist_event(
    event: MessageEvent,
    session_key: str,
    *,
    reason: str,
    path: Path | None = None,
    producer_token: str | None = None,
) -> str | None:
    queue_id, state = persist_event_result(
        event,
        session_key,
        reason=reason,
        path=path,
        producer_token=producer_token,
    )
    return queue_id if state == _QUEUED else None


def event_state(
    event: MessageEvent,
    session_key: str,
    path: Path | None = None,
) -> str | None:
    receipt = event_receipt(event, session_key, path)
    if receipt is None:
        return None
    return str(receipt.get("state") or _QUEUED)


def event_receipt(
    event: MessageEvent,
    session_key: str,
    path: Path | None = None,
) -> dict[str, Any] | None:
    queue_id = _event_identity(event, session_key)
    path = path or inbox_path()
    with _locked_inbox(path):
        for _, parsed in _read_rows(path):
            if parsed is not None and parsed.get("queue_id") == queue_id:
                return dict(parsed)
    return None


def pending_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Return parseable pending records in arrival order."""
    path = path or inbox_path()
    with _locked_inbox(path):
        rows = _read_rows(path)
        return [
            parsed
            for _, parsed in rows
            if parsed is not None and parsed.get("queue_id")
        ]


def _replace_rows(path: Path, rows: list[str]) -> None:
    _ensure_trusted_directory(path.parent)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{_FILE_NAME}.",
        suffix=".tmp",
    )
    installed = False
    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            errors="surrogateescape",
        ) as handle:
            if os.name != "nt":
                os.fchmod(handle.fileno(), 0o600)
                if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
                    raise PermissionError(f"Durable inbox is not private: {path}")
            if rows:
                handle.write("\n".join(rows) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        installed = True
        try:
            _fsync_directory(path.parent)
        except OSError:
            _fsync_directory(path.parent)
    except BaseException as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        if installed and isinstance(exc, Exception):
            raise _PostReplaceError(str(exc)) from exc
        raise


def _transition_event(
    queue_id: str,
    expected_state: str,
    new_state: str,
    path: Path | None = None,
    *,
    producer_token: str | None = None,
) -> bool:
    if not queue_id:
        return False
    path = path or inbox_path()
    try:
        with _locked_inbox(path) as handle:
            if producer_token is not None:
                handle.seek(0)
                if handle.read().strip() != producer_token:
                    raise RuntimeError("durable inbox producer ownership changed")
            rows = _read_rows(path)
            replaced: list[str] = []
            transitioned = False
            for raw, parsed in rows:
                if (
                    not transitioned
                    and parsed is not None
                    and parsed.get("queue_id") == queue_id
                ):
                    current_state = str(parsed.get("state") or _QUEUED)
                    if current_state != expected_state:
                        return False
                    updated = dict(parsed)
                    updated["state"] = new_state
                    updated["state_changed_at"] = (
                        datetime.now().astimezone().isoformat()
                    )
                    raw = json.dumps(updated, ensure_ascii=False, default=str)
                    transitioned = True
                replaced.append(raw)
            if not transitioned:
                return False
            _replace_rows(path, replaced)
    except Exception as exc:
        logger.warning(
            "Failed to transition drain-inbox row %s from %s to %s: %s",
            queue_id,
            expected_state,
            new_state,
            exc,
        )
        return False
    return True


def claim_event(
    queue_id: str,
    path: Path | None = None,
    *,
    producer_token: str | None = None,
) -> bool:
    return _transition_event(
        queue_id,
        _QUEUED,
        _CLAIMED,
        path,
        producer_token=producer_token,
    )


def complete_event(
    queue_id: str,
    path: Path | None = None,
    *,
    producer_token: str | None = None,
) -> bool:
    return _transition_event(
        queue_id,
        _CLAIMED,
        _COMPLETED,
        path,
        producer_token=producer_token,
    )


def event_from_record(record: dict[str, Any]) -> MessageEvent:
    """Rebuild a normalized event without restoring untrusted raw payloads."""
    if record.get("schema") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported drain-inbox schema: {record.get('schema')!r}")
    pre_dispatch_attempted = record.get("pre_dispatch_attempted", False)
    if not isinstance(pre_dispatch_attempted, bool):
        raise ValueError("pre-dispatch receipt must be a boolean")
    internal = record.get("internal", False)
    if not isinstance(internal, bool):
        raise ValueError("internal receipt must be a boolean")
    metadata = _normalize_metadata(record.get("metadata", {}))
    raw_type = str(record.get("message_type") or MessageType.TEXT.value)
    try:
        message_type = MessageType(raw_type)
    except ValueError:
        message_type = MessageType.TEXT
    raw_timestamp = record.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(str(raw_timestamp))
    except (TypeError, ValueError):
        timestamp = datetime.now()
    queue_id = str(record.get("queue_id") or "")
    message_id = str(record.get("message_id") or "").strip()
    if not message_id and queue_id:
        message_id = f"drain:{queue_id}"
    event = MessageEvent(
        text=str(record.get("text") or ""),
        message_type=message_type,
        source=SessionSource.from_dict(record["source"]),
        message_id=message_id,
        platform_update_id=record.get("platform_update_id"),
        media_urls=list(record.get("media_urls") or []),
        media_types=list(record.get("media_types") or []),
        reply_to_message_id=record.get("reply_to_message_id"),
        reply_to_text=record.get("reply_to_text"),
        reply_to_author_id=record.get("reply_to_author_id"),
        reply_to_author_name=record.get("reply_to_author_name"),
        reply_to_is_own_message=bool(record.get("reply_to_is_own_message", False)),
        auto_skill=record.get("auto_skill"),
        channel_prompt=record.get("channel_prompt"),
        channel_context=record.get("channel_context"),
        internal=internal,
        metadata=metadata,
        timestamp=timestamp,
    )
    setattr(
        event,
        "_hermes_pre_gateway_dispatch_attempted",
        pre_dispatch_attempted,
    )
    return event


def acknowledge(
    queue_id: str,
    path: Path | None = None,
    *,
    expected_state: str | None = None,
    producer_token: str | None = None,
) -> bool:
    """Atomically remove one matching record while retaining malformed rows."""
    if not queue_id:
        return False
    path = path or inbox_path()
    try:
        with _locked_inbox(path) as handle:
            if producer_token is not None:
                handle.seek(0)
                if handle.read().strip() != producer_token:
                    raise RuntimeError("durable inbox producer ownership changed")
            rows = _read_rows(path)
            kept: list[str] = []
            removed = False
            for raw, parsed in rows:
                if (
                    not removed
                    and parsed is not None
                    and parsed.get("queue_id") == queue_id
                ):
                    current_state = str(parsed.get("state") or _QUEUED)
                    if (
                        expected_state is not None
                        and current_state != expected_state
                    ):
                        return False
                    removed = True
                    continue
                kept.append(raw)
            if not removed:
                return False

            _replace_rows(path, kept)
    except Exception as exc:
        logger.warning("Failed to acknowledge drain-inbox row %s: %s", queue_id, exc)
        return False
    return True
