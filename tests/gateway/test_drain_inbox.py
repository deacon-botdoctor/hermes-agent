import asyncio
import errno
import json
import multiprocessing
import os
import stat
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.drain_inbox as drain_inbox
import gateway.run as gateway_run
from gateway.drain_inbox import (
    acquire_producer_replay_lease,
    acquire_replay_lease,
    acknowledge,
    cancel_producer_replay_lease,
    claim_pre_dispatch_event_result,
    claim_event,
    claim_producer,
    complete_event,
    event_from_record,
    event_state,
    finalize_pre_dispatch_event_result,
    inbox_path,
    pending_records,
    persist_event,
    persist_event_result,
    record_pre_dispatch_attempt_result,
    release_replay_lease,
)
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _acknowledge_after_pause(
    path: str,
    queue_id: str,
    ready,
    proceed,
) -> None:
    """Pause an acknowledge at its final replace for a process-overlap test."""
    real_replace = drain_inbox.os.replace

    def paused_replace(source, destination):
        ready.set()
        if not proceed.wait(timeout=10):
            raise TimeoutError("test did not release acknowledge")
        return real_replace(source, destination)

    drain_inbox.os.replace = paused_replace  # ty:ignore[invalid-assignment]
    if not acknowledge(queue_id, Path(path)):
        raise RuntimeError("acknowledge failed")


def _claim_producer_in_process(path: str, producer_token: str, done) -> None:
    claim_producer(producer_token, Path(path))
    done.set()


def _persist_in_process(path: str, done) -> None:
    event = MessageEvent(
        text="third instruction",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="message-44",
        platform_update_id=101,
    )
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="overlapping-new-process",
        path=Path(path),
    )
    if not queue_id:
        raise RuntimeError("persist failed")
    done.set()


def _event(text: str = "finish the long instruction") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="message-42",
        platform_update_id=99,
    )


def test_persisted_event_round_trips_exact_user_instruction(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("A long instruction with unicode: résumé → done")
    session_key = build_session_key(event.source)

    queue_id = persist_event(event, session_key, reason="test-drain")

    assert queue_id
    records = pending_records()
    assert len(records) == 1
    assert records[0]["state"] == "queued"
    assert records[0]["pre_dispatch_attempted"] is False
    replay = event_from_record(records[0])
    assert replay.text == event.text
    assert replay.message_id == event.message_id
    assert replay.platform_update_id == event.platform_update_id
    assert replay.source.to_dict() == event.source.to_dict()
    if os.name != "nt":
        assert inbox_path().stat().st_mode & 0o777 == 0o600


def test_pre_dispatch_receipt_round_trips_and_rejects_non_boolean(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)

    assert persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    record = pending_records()[0]
    assert record["pre_dispatch_attempted"] is True
    replay = event_from_record(record)
    assert getattr(replay, "_hermes_pre_gateway_dispatch_attempted") is True

    invalid = dict(record)
    invalid["pre_dispatch_attempted"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        event_from_record(invalid)


def test_internal_event_metadata_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    event.internal = True
    event.durable_ingress = True
    event.metadata = {
        "webhook_delivery": {
            "deliver": "telegram",
            "deliver_extra": {"chat_id": "12345"},
        }
    }

    assert persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )

    replay = event_from_record(pending_records()[0])
    assert replay.internal is True
    assert replay.metadata == event.metadata


def test_claim_and_complete_transitions_are_durable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )

    assert claim_event(queue_id)  # ty:ignore[invalid-argument-type]
    assert pending_records()[0]["state"] == "claimed"
    assert complete_event(queue_id)  # ty:ignore[invalid-argument-type]
    assert pending_records()[0]["state"] == "completed"


def test_lifecycle_transitions_require_current_producer(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    claim_producer("current-process", path)
    event = _event()
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
        path=path,
        producer_token="current-process",
    )

    assert not claim_event(
        queue_id,  # ty:ignore[invalid-argument-type]
        path,
        producer_token="stale-process",
    )
    assert claim_event(
        queue_id,  # ty:ignore[invalid-argument-type]
        path,
        producer_token="current-process",
    )
    assert not complete_event(
        queue_id,  # ty:ignore[invalid-argument-type]
        path,
        producer_token="stale-process",
    )
    assert complete_event(
        queue_id,  # ty:ignore[invalid-argument-type]
        path,
        producer_token="current-process",
    )
    assert not acknowledge(
        queue_id,  # ty:ignore[invalid-argument-type]
        path,
        producer_token="stale-process",
    )
    assert acknowledge(
        queue_id,  # ty:ignore[invalid-argument-type]
        path,
        producer_token="current-process",
    )


def test_duplicate_platform_event_is_written_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    session_key = build_session_key(event.source)

    first = persist_event(event, session_key, reason="first")
    second = persist_event(event, session_key, reason="duplicate-delivery")

    assert first == second
    assert len(pending_records()) == 1


def test_pre_dispatch_claim_has_one_owner_and_finalizes_handled(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("plugin candidate")
    session_key = build_session_key(event.source)
    barrier = threading.Barrier(3)
    results: list[tuple[str | None, str, bool]] = []

    def claim() -> None:
        barrier.wait()
        results.append(
            claim_pre_dispatch_event_result(
                event,
                session_key,
                reason="pre-dispatch",
            )
        )

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sum(acquired for _, _, acquired in results) == 1
    assert {state for _, state, _ in results} == {"claimed"}
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)
    queue_id, state = finalize_pre_dispatch_event_result(
        event,
        session_key,
        handled=True,
        reason="plugin-handled",
    )
    assert queue_id
    assert state == "handled"
    assert event_state(event, session_key) == "handled"


def test_pre_dispatch_promotion_failure_restores_claimed_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("plugin candidate")
    session_key = build_session_key(event.source)
    queue_id, state, acquired = claim_pre_dispatch_event_result(
        event,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def fail_after_promotion(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        real_replace_rows(path, rows)
        if replace_calls == 1:
            raise drain_inbox._PostReplaceError(
                "simulated promotion directory fsync failure"
            )

    monkeypatch.setattr(drain_inbox, "_replace_rows", fail_after_promotion)

    result_queue_id, result_state = finalize_pre_dispatch_event_result(
        event,
        session_key,
        handled=False,
        reason="startup-gate",
    )

    assert result_queue_id == queue_id
    assert result_state == "claimed"
    assert pending_records()[0]["state"] == "claimed"


def test_pre_dispatch_attempt_phase_remains_claimed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("plugin candidate")
    session_key = build_session_key(event.source)
    queue_id, state, acquired = claim_pre_dispatch_event_result(
        event,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)

    result_queue_id, result_state = record_pre_dispatch_attempt_result(
        event,
        session_key,
    )

    assert result_queue_id == queue_id
    assert result_state == "claimed"
    record = pending_records()[0]
    assert record["state"] == "claimed"
    assert record["pre_dispatch_attempted"] is True


def test_pre_dispatch_attempt_phase_retries_transient_write_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("plugin candidate")
    session_key = build_session_key(event.source)
    queue_id, state, acquired = claim_pre_dispatch_event_result(
        event,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def fail_once(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise OSError("simulated phase write failure")
        return real_replace_rows(path, rows)

    monkeypatch.setattr(drain_inbox, "_replace_rows", fail_once)

    result_queue_id, result_state = record_pre_dispatch_attempt_result(
        event,
        session_key,
    )

    assert result_queue_id == queue_id
    assert result_state == "claimed"
    assert replace_calls == 2
    assert pending_records()[0]["pre_dispatch_attempted"] is True


def test_pre_dispatch_attempt_phase_quarantines_exhausted_writes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("plugin candidate")
    session_key = build_session_key(event.source)
    queue_id, state, acquired = claim_pre_dispatch_event_result(
        event,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def fail_until_quarantine_retry(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls <= 3:
            raise OSError("simulated phase write failure")
        real_replace_rows(path, rows)
        raise drain_inbox._PostReplaceError(
            "simulated quarantine directory fsync failure"
        )

    monkeypatch.setattr(
        drain_inbox,
        "_replace_rows",
        fail_until_quarantine_retry,
    )

    result_queue_id, result_state = record_pre_dispatch_attempt_result(
        event,
        session_key,
    )

    assert result_queue_id == queue_id
    assert result_state == "ambiguous"
    assert replace_calls == 4
    record = pending_records()[0]
    assert record["state"] == "ambiguous"
    assert record["recovery_disposition"] == "ambiguous"
    assert record["pre_dispatch_attempted"] is True


def test_pre_dispatch_control_command_cannot_promote_to_queued(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("/restart")
    session_key = build_session_key(event.source)
    queue_id, state, acquired = claim_pre_dispatch_event_result(
        event,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(event, "_hermes_pre_gateway_dispatch_attempted", True)

    result_queue_id, result_state = finalize_pre_dispatch_event_result(
        event,
        session_key,
        handled=False,
        reason="startup-gate",
    )

    assert result_queue_id == queue_id
    assert result_state == "claimed"
    assert pending_records()[0]["state"] == "claimed"


def test_queued_duplicate_reconciles_normalized_pre_dispatch_receipt(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = _event("original")
    session_key = build_session_key(original.source)
    queue_id = persist_event(original, session_key, reason="busy-path")
    queued_at = pending_records()[0]["queued_at"]
    redelivery = _event("rewritten")
    setattr(redelivery, "_hermes_pre_gateway_dispatch_attempted", True)

    assert persist_event(
        redelivery,
        session_key,
        reason="cold-redelivery",
    ) == queue_id

    record = pending_records()[0]
    assert record["state"] == "queued"
    assert record["text"] == "rewritten"
    assert record["pre_dispatch_attempted"] is True
    assert record["queued_at"] == queued_at
    assert record["reason"] == "busy-path"


@pytest.mark.parametrize("state", ["claimed", "completed"])
def test_nonqueued_duplicate_reports_existing_state(
    tmp_path,
    monkeypatch,
    state,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = _event("original")
    session_key = build_session_key(original.source)
    queue_id = persist_event(original, session_key, reason="busy-path")
    assert claim_event(queue_id)  # ty:ignore[invalid-argument-type]
    if state == "completed":
        assert complete_event(queue_id)  # ty:ignore[invalid-argument-type]
    redelivery = _event("rewritten")
    setattr(redelivery, "_hermes_pre_gateway_dispatch_attempted", True)

    result_queue_id, result_state = persist_event_result(
        redelivery,
        session_key,
        reason="cold-redelivery",
    )

    assert result_queue_id == queue_id
    assert result_state == state

    record = pending_records()[0]
    assert record["state"] == state
    assert record["text"] == "original"
    assert record["pre_dispatch_attempted"] is False


def test_reconciliation_claimed_baseline_failure_never_reports_acceptance(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = _event("original")
    session_key = build_session_key(original.source)
    queue_id = persist_event(original, session_key, reason="busy-path")
    redelivery = _event("rewritten")
    setattr(redelivery, "_hermes_pre_gateway_dispatch_attempted", True)

    fsync_calls = 0

    def fail_first_replacement_fsync(_path):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls <= 2:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        drain_inbox,
        "_fsync_directory",
        fail_first_replacement_fsync,
    )

    result_queue_id, result_state = persist_event_result(
        redelivery,
        session_key,
        reason="cold-redelivery",
    )

    assert result_queue_id is None
    assert result_state == "failed"
    record = pending_records()[0]
    assert record["state"] == "claimed"
    assert record["text"] == "rewritten"
    assert record["pre_dispatch_attempted"] is True


def test_reconciliation_queued_promotion_post_replace_failure_is_accepted(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = _event("original")
    session_key = build_session_key(original.source)
    queue_id = persist_event(original, session_key, reason="busy-path")
    redelivery = _event("rewritten")
    setattr(redelivery, "_hermes_pre_gateway_dispatch_attempted", True)
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def staged_replace_failure(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        row = json.loads(rows[0])
        if replace_calls == 1:
            assert row["state"] == "claimed"
            assert row["pre_dispatch_attempted"] is True
            return real_replace_rows(path, rows)
        real_replace_rows(path, rows)
        raise drain_inbox._PostReplaceError(
            "simulated queued promotion fsync failure"
        )

    monkeypatch.setattr(drain_inbox, "_replace_rows", staged_replace_failure)

    result_queue_id, result_state = persist_event_result(
        redelivery,
        session_key,
        reason="cold-redelivery",
    )

    assert result_queue_id == queue_id
    assert result_state == "queued"
    record = pending_records()[0]
    assert record["state"] == "queued"
    assert record["text"] == "rewritten"
    assert record["pre_dispatch_attempted"] is True


def test_reconciliation_queued_promotion_failure_retains_claimed_receipt(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = _event("original")
    session_key = build_session_key(original.source)
    queue_id = persist_event(original, session_key, reason="busy-path")
    redelivery = _event("rewritten")
    setattr(redelivery, "_hermes_pre_gateway_dispatch_attempted", True)
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def staged_replace_failure(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            return real_replace_rows(path, rows)
        raise OSError("simulated queued promotion failure")

    monkeypatch.setattr(drain_inbox, "_replace_rows", staged_replace_failure)

    result_queue_id, result_state = persist_event_result(
        redelivery,
        session_key,
        reason="cold-redelivery",
    )

    assert result_queue_id == queue_id
    assert result_state == "claimed"
    record = pending_records()[0]
    assert record["state"] == "claimed"
    assert record["text"] == "rewritten"
    assert record["pre_dispatch_attempted"] is True


def test_duplicate_platform_event_ignores_delivery_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first_event = _event()
    redelivery = _event()
    redelivery.timestamp = first_event.timestamp + timedelta(seconds=30)
    session_key = build_session_key(first_event.source)

    first = persist_event(first_event, session_key, reason="first")
    second = persist_event(redelivery, session_key, reason="redelivery")

    assert first == second
    assert len(pending_records()) == 1


def test_acknowledge_retains_malformed_rows_for_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    with inbox_path().open("a", encoding="utf-8") as handle:
        handle.write("{not-json\n")

    assert acknowledge(queue_id)  # ty:ignore[invalid-argument-type]
    assert pending_records() == []  # HERMES_DURABLE_DRAIN_COMPATIBILITY_TEST_v1
    assert inbox_path().read_text(encoding="utf-8") == "{not-json\n"


def test_control_command_is_not_replayed_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("/restart")

    assert persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    ) is None
    assert not inbox_path().exists()


def test_whitespace_prefixed_control_command_is_not_replayed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("  /restart")

    assert persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    ) is None
    assert not inbox_path().exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes required")
def test_new_inbox_is_private_even_when_process_umask_is_permissive(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: None)
    previous_umask = os.umask(0)
    try:
        queue_id = persist_event(
            _event(),
            build_session_key(_event().source),
            reason="test-drain",
        )
    finally:
        os.umask(previous_umask)

    assert queue_id
    assert inbox_path().stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes required")
def test_existing_nonprivate_inbox_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o666)

    assert persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    ) is None
    with pytest.raises(PermissionError, match="not private"):
        pending_records()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics required")
def test_existing_symlink_inbox_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    path.parent.mkdir(parents=True)
    target = path.parent / "target.jsonl"
    target.write_text("unchanged\n", encoding="utf-8")
    target.chmod(0o600)
    path.symlink_to(target)

    assert persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    ) is None
    with pytest.raises(OSError):
        pending_records()
    assert target.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes required")
def test_other_writable_state_directory_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    path.parent.mkdir(parents=True, mode=0o777)
    path.parent.chmod(0o777)

    assert persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    ) is None
    with pytest.raises(PermissionError, match="writable by other users"):
        pending_records()
    assert not path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes required")
def test_other_writable_home_parent_is_rejected(tmp_path, monkeypatch):
    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o777)
    unsafe_parent.chmod(0o777)
    hermes_home = unsafe_parent / "hermes-home"
    hermes_home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    ) is None
    with pytest.raises(PermissionError, match="trust chain"):
        pending_records()
    assert not inbox_path().exists()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "getuid"),
    reason="POSIX ownership required",
)
def test_owner_writable_untrusted_ancestor_is_rejected(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    real_open_trusted_directory = drain_inbox._open_trusted_directory

    def foreign_owner(component):
        component_stat = real_open_trusted_directory(component)
        if component == tmp_path:
            return SimpleNamespace(
                st_mode=component_stat.st_mode | stat.S_IWUSR,
                st_uid=os.getuid() + 1,  # windows-footgun: ok — POSIX-only test
            )
        return component_stat

    monkeypatch.setattr(
        drain_inbox,
        "_open_trusted_directory",
        foreign_owner,
    )

    assert persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    ) is None
    with pytest.raises(PermissionError, match="untrusted owner"):
        pending_records()
    assert not inbox_path().exists()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "getuid"),
    reason="POSIX ownership required",
)
def test_readonly_synthetic_namespace_ancestors_are_trusted(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    real_open_trusted_directory = drain_inbox._open_trusted_directory

    namespace_prefix = {
        Path(tmp_path.anchor),
        Path(tmp_path.anchor) / tmp_path.parts[1],
    }

    def synthetic_namespace_owner(component):
        component_stat = real_open_trusted_directory(component)
        if component in namespace_prefix:
            return SimpleNamespace(
                st_mode=(component_stat.st_mode & ~0o022) | stat.S_IWUSR,
                st_uid=65534,
            )
        return component_stat

    monkeypatch.setattr(
        drain_inbox,
        "_open_trusted_directory",
        synthetic_namespace_owner,
    )

    assert persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory descriptors required")
def test_trusted_directory_prefers_metadata_only_open(tmp_path, monkeypatch):
    fake_o_path = 1 << 29
    real_open = os.open
    observed_flags = []

    monkeypatch.setattr(drain_inbox.os, "O_PATH", fake_o_path, raising=False)

    def open_without_fake_flag(path, flags):
        observed_flags.append(flags)
        return real_open(path, flags & ~fake_o_path)

    monkeypatch.setattr(drain_inbox.os, "open", open_without_fake_flag)

    directory_stat = drain_inbox._open_trusted_directory(tmp_path)

    assert stat.S_ISDIR(directory_stat.st_mode)
    assert observed_flags[0] & fake_o_path


def test_failed_fsync_does_not_leave_a_replayable_row(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fail_fsync(_fd):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    queue_id = persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    )

    assert queue_id is None
    assert pending_records() == []


def test_claimed_baseline_post_replace_failure_remains_cross_process_safe(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fsync_calls = 0

    def fail_first_replacement_fsync(_path):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls <= 2:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        drain_inbox,
        "_fsync_directory",
        fail_first_replacement_fsync,
    )

    queue_id, state = persist_event_result(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    )

    assert queue_id is None
    assert state == "failed"
    record = pending_records()[0]
    assert record["state"] == "claimed"


def test_queued_promotion_pre_replace_failure_retains_claimed_baseline(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def staged_replace_failure(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            return real_replace_rows(path, rows)
        raise OSError("simulated queued promotion failure")

    monkeypatch.setattr(drain_inbox, "_replace_rows", staged_replace_failure)

    queue_id, state = persist_event_result(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    )

    assert queue_id
    assert state == "claimed"
    assert pending_records()[0]["state"] == "claimed"


def test_queued_promotion_post_replace_failure_is_durably_accepted(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def staged_replace_failure(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        real_replace_rows(path, rows)
        if replace_calls == 2:
            raise drain_inbox._PostReplaceError(
                "simulated queued promotion fsync failure"
            )

    monkeypatch.setattr(drain_inbox, "_replace_rows", staged_replace_failure)

    queue_id, state = persist_event_result(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    )

    assert queue_id
    assert state == "queued"
    assert pending_records()[0]["state"] == "queued"


def test_failed_directory_fsync_quarantines_new_inbox(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real_fsync = os.fsync
    calls = 0

    def fail_second_fsync(fd):
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("simulated directory storage failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_second_fsync)
    queue_id = persist_event(
        _event(),
        build_session_key(_event().source),
        reason="test-drain",
    )

    assert queue_id is None
    assert pending_records()[0]["state"] == "claimed"


def test_durable_inbox_capacity_rejects_new_unique_rows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(drain_inbox, "_MAX_PENDING_RECORDS", 2)
    first = _event("first")
    second = _event("second")
    second.message_id = "message-43"
    third = _event("third")
    third.message_id = "message-44"
    session_key = build_session_key(first.source)

    first_id = persist_event(first, session_key, reason="test-drain")
    assert first_id
    assert persist_event(second, session_key, reason="test-drain")
    assert persist_event(first, session_key, reason="redelivery") == first_id
    assert persist_event(third, session_key, reason="test-drain") is None
    assert len(pending_records()) == 2


def test_ambiguous_receipts_use_bounded_reserved_capacity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(drain_inbox, "_MAX_PENDING_RECORDS", 1)
    monkeypatch.setattr(drain_inbox, "_MAX_QUARANTINED_RECORDS", 1)
    ambiguous = _event("ambiguous")
    session_key = build_session_key(ambiguous.source)
    queue_id, state, acquired = claim_pre_dispatch_event_result(
        ambiguous,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(ambiguous, "_hermes_pre_gateway_dispatch_attempted", True)
    real_replace_rows = drain_inbox._replace_rows
    replace_calls = 0

    def fail_twice(path, rows):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls <= 2:
            raise OSError("simulated phase write failure")
        return real_replace_rows(path, rows)

    monkeypatch.setattr(drain_inbox, "_replace_rows", fail_twice)
    assert record_pre_dispatch_attempt_result(
        ambiguous,
        session_key,
    ) == (queue_id, "ambiguous")
    monkeypatch.setattr(drain_inbox, "_replace_rows", real_replace_rows)

    queued = _event("queued")
    queued.message_id = "message-after-ambiguous"
    assert persist_event(queued, session_key, reason="test-drain")
    assert [record["state"] for record in pending_records()] == [
        "ambiguous",
        "queued",
    ]

    queue_id, state, acquired = claim_pre_dispatch_event_result(
        queued,
        session_key,
        reason="pre-dispatch",
    )
    assert queue_id
    assert state == "claimed"
    assert acquired is True
    setattr(queued, "_hermes_pre_gateway_dispatch_attempted", True)
    replace_calls = 0
    monkeypatch.setattr(drain_inbox, "_replace_rows", fail_twice)
    assert record_pre_dispatch_attempt_result(
        queued,
        session_key,
    ) == (queue_id, "ambiguous")
    monkeypatch.setattr(drain_inbox, "_replace_rows", real_replace_rows)

    rejected = _event("rejected")
    rejected.message_id = "message-after-full-quarantine"
    assert persist_event(rejected, session_key, reason="test-drain") is None
    assert [record["state"] for record in pending_records()] == [
        "ambiguous",
        "ambiguous",
    ]


def test_failed_copy_on_write_preserves_earlier_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = _event("first")
    session_key = build_session_key(first.source)
    first_queue_id = persist_event(first, session_key, reason="test-drain")
    second = _event("second")
    second.message_id = "message-43"

    def fail_fsync(_fd):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    assert persist_event(
        second,
        session_key,
        reason="test-drain",
    ) is None
    assert [record["queue_id"] for record in pending_records()] == [
        first_queue_id
    ]


def test_cross_device_replace_failure_never_truncates_live_inbox(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = _event("first")
    session_key = build_session_key(first.source)
    first_queue_id = persist_event(first, session_key, reason="test-drain")
    original_contents = inbox_path().read_bytes()
    second = _event("second")
    second.message_id = "message-43"

    def fail_replace(_source, _destination):
        raise OSError(errno.EXDEV, "simulated cross-device replace")

    monkeypatch.setattr(os, "replace", fail_replace)

    assert persist_event(
        second,
        session_key,
        reason="test-drain",
    ) is None
    assert inbox_path().read_bytes() == original_contents
    assert [record["queue_id"] for record in pending_records()] == [
        first_queue_id
    ]


def test_incomplete_trailing_append_does_not_block_earlier_recovery(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event("first")
    session_key = build_session_key(event.source)
    queue_id = persist_event(event, session_key, reason="test-drain")
    with inbox_path().open("ab") as handle:
        handle.write(b'\n{"schema":1,"queue_id":"partial')

    assert [record["queue_id"] for record in pending_records()] == [queue_id]
    assert claim_event(queue_id)  # ty:ignore[invalid-argument-type]
    assert b'"queue_id":"partial' not in inbox_path().read_bytes()


def test_blank_message_id_uses_platform_update_id_for_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = _event("first")
    first.message_id = ""
    second = _event("second")
    second.message_id = ""
    second.platform_update_id = 100
    session_key = build_session_key(first.source)

    first_queue_id = persist_event(first, session_key, reason="test-drain")
    second_queue_id = persist_event(second, session_key, reason="test-drain")

    assert first_queue_id != second_queue_id
    assert len(pending_records()) == 2


def test_zero_platform_update_id_is_stable_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    event.message_id = None
    event.platform_update_id = 0

    assert persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    assert len(pending_records()) == 1


def test_event_without_stable_platform_identity_is_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = _event()
    first.message_id = None
    first.platform_update_id = None
    second = _event()
    second.message_id = None
    second.platform_update_id = None
    second.timestamp = first.timestamp + timedelta(seconds=30)
    session_key = build_session_key(first.source)

    assert persist_event(first, session_key, reason="first") is None
    assert persist_event(second, session_key, reason="redelivery") is None
    assert pending_records() == []


@pytest.mark.skipif(os.name == "nt", reason="fork-based process overlap test")
def test_old_process_ack_cannot_erase_new_process_append(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    first = _event("first instruction")
    second = MessageEvent(
        text="second instruction",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="message-43",
        platform_update_id=100,
    )
    first_queue_id = persist_event(
        first,
        build_session_key(first.source),
        reason="old-process",
    )
    assert first_queue_id
    assert persist_event(
        second,
        build_session_key(second.source),
        reason="old-process",
    )

    context = multiprocessing.get_context("fork")
    acknowledge_ready = context.Event()
    allow_acknowledge = context.Event()
    append_done = context.Event()
    acknowledge_process = context.Process(
        target=_acknowledge_after_pause,
        args=(
            str(path),
            first_queue_id,
            acknowledge_ready,
            allow_acknowledge,
        ),
    )
    append_process = context.Process(
        target=_persist_in_process,
        args=(str(path), append_done),
    )

    acknowledge_process.start()
    assert acknowledge_ready.wait(timeout=10)
    append_process.start()
    append_done.wait(timeout=1)
    allow_acknowledge.set()
    acknowledge_process.join(timeout=10)
    append_process.join(timeout=10)

    assert acknowledge_process.exitcode == 0
    assert append_process.exitcode == 0
    assert [record["text"] for record in pending_records()] == [
        "second instruction",
        "third instruction",
    ]


def test_new_producer_claim_rejects_old_process_append(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    session_key = build_session_key(_event().source)
    claim_producer("old-process", path)
    assert persist_event(
        _event("accepted before handoff"),
        session_key,
        reason="old-process",
        path=path,
        producer_token="old-process",
    )

    claim_producer("new-process", path)
    late = _event("late old-process message")
    late.message_id = "message-late"
    current = _event("new-process message")
    current.message_id = "message-current"

    assert persist_event(
        late,
        session_key,
        reason="old-process",
        path=path,
        producer_token="old-process",
    ) is None
    assert persist_event(
        current,
        session_key,
        reason="new-process",
        path=path,
        producer_token="new-process",
    )
    assert [record["text"] for record in pending_records(path)] == [
        "accepted before handoff",
        "new-process message",
    ]


def test_pending_record_read_failure_is_not_reported_as_empty(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fail_read(_path):
        raise OSError("simulated read failure")

    monkeypatch.setattr(drain_inbox, "_read_rows", fail_read)

    with pytest.raises(OSError, match="simulated read failure"):
        pending_records()


@pytest.mark.asyncio
async def test_runner_persistence_does_not_block_event_loop(monkeypatch):
    runner, _adapter = make_restart_runner()
    started = threading.Event()
    release = threading.Event()

    def blocking_persist(*_args, **_kwargs):
        started.set()
        if not release.wait(timeout=2):
            raise TimeoutError("test did not release persistence")
        return "queue-id"

    monkeypatch.setattr(gateway_run, "persist_drain_event", blocking_persist)
    task = asyncio.create_task(
        runner._persist_drain_event(
            _event(),
            build_session_key(_event().source),
            reason="lock-contention",
        )
    )
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.001)
    assert started.is_set()

    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    assert await task is True


@pytest.mark.asyncio
async def test_runner_uses_process_inbox_across_profile_scopes(tmp_path, monkeypatch):
    process_home = tmp_path / "gateway-home"
    profile_home = tmp_path / "profile-home"
    runner, _adapter = make_restart_runner()
    runner._drain_inbox_path = inbox_path(process_home)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    event = _event()

    assert await runner._persist_drain_event(
        event,
        build_session_key(event.source),
        reason="multiplex-profile-drain",
    )

    assert [row["text"] for row in pending_records(runner._drain_inbox_path)] == [
        event.text
    ]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_runner_replays_and_acknowledges_durable_drain_event(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    session_key = build_session_key(event.source)
    persist_event(event, session_key, reason="test-drain")
    with inbox_path().open("a", encoding="utf-8") as handle:
        handle.write("{not-json\n")

    runner, adapter = make_restart_runner()
    committed_message_ids: set[str] = set()

    async def persist_marker(_session_id, message_id, **_kwargs):
        committed_message_ids.add(str(message_id or ""))
        return True

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(
            side_effect=lambda _session_id, message_id: (
                message_id in committed_message_ids
            )
        ),
        persist_replay_marker=AsyncMock(side_effect=persist_marker),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._schedule_resume_pending_sessions = lambda: 0  # ty:ignore[invalid-assignment]

    await runner._finish_startup_restore()

    assert runner._startup_restore_in_progress is False
    runner._run_startup_resume_event.assert_awaited_once()  # ty:ignore[unresolved-attribute]
    replay, replay_session_key = runner._run_startup_resume_event.await_args.args[1:]  # ty:ignore[unresolved-attribute]
    assert runner._run_startup_resume_event.await_args.args[0] is adapter  # ty:ignore[unresolved-attribute]
    assert replay.text == event.text
    assert replay_session_key == session_key
    assert replay._hermes_startup_restore_replay is True
    assert committed_message_ids == {event.message_id}
    assert pending_records() == []
    assert inbox_path().read_text(encoding="utf-8") == "{not-json\n"


@pytest.mark.asyncio
async def test_runner_dedupes_committed_platform_message_before_replay(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )

    runner, _adapter = make_restart_runner()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=True),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 1
    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("has_native_message_id", [True, False])
async def test_replay_marker_dedupes_after_handler_ack_gap(
    tmp_path,
    monkeypatch,
    has_native_message_id,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    if not has_native_message_id:
        event.message_id = None
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    assert queue_id

    committed_message_ids: set[str] = set()
    first_runner, _adapter = make_restart_runner()
    first_runner._async_session_store = SimpleNamespace(
        _store=first_runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
    )

    async def persist_marker(_session_id, message_id, **_kwargs):
        committed_message_ids.add(str(message_id or ""))
        return True

    first_runner._async_session_store.persist_replay_marker = AsyncMock(
        side_effect=persist_marker
    )
    first_runner._async_session_store.replay_marker_status = AsyncMock(
        side_effect=lambda _session_id, message_id: (
            message_id in committed_message_ids
        )
    )
    first_runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    monkeypatch.setattr(
        gateway_run,
        "acknowledge_drain_event",
        lambda *_args, **_kwargs: False,
    )

    assert await first_runner._drain_persisted_drain_inbox() == 0
    expected_message_id = (
        "message-42" if has_native_message_id else f"drain:{queue_id}"
    )
    assert committed_message_ids == {expected_message_id}
    assert len(pending_records()) == 1

    second_runner, _adapter = make_restart_runner()
    second_runner._async_session_store = SimpleNamespace(
        _store=second_runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(
            side_effect=lambda _session_id, message_id: (
                message_id in committed_message_ids
            )
        ),
    )
    second_runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    monkeypatch.setattr(gateway_run, "acknowledge_drain_event", acknowledge)

    assert await second_runner._drain_persisted_drain_inbox() == 1
    second_runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_unauthorized_durable_event_remains_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    runner, _adapter = make_restart_runner()
    runner._is_user_authorized = lambda _source: False  # ty:ignore[invalid-assignment]
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(),
        replay_marker_status=AsyncMock(),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 0
    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert len(pending_records()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "pre_dispatch_attempted"),
    [
        ("test-drain", False),
        ("startup-restore-pre-dispatch", False),
        ("startup-restore-pre-dispatch", True),
    ],
)
async def test_revoked_claimed_event_remains_ambiguous_without_notice(
    tmp_path,
    monkeypatch,
    reason,
    pre_dispatch_attempted,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    if reason == "startup-restore-pre-dispatch":
        _queue_id, state, acquired = claim_pre_dispatch_event_result(
            event,
            build_session_key(event.source),
            reason=reason,
        )
        assert state == "claimed"
        assert acquired
        if pre_dispatch_attempted:
            setattr(
                event,
                "_hermes_pre_gateway_dispatch_attempted",
                True,
            )
            assert record_pre_dispatch_attempt_result(
                event,
                build_session_key(event.source),
            )[1] == "claimed"
    else:
        queue_id = persist_event(
            event,
            build_session_key(event.source),
            reason=reason,
        )
        assert claim_event(queue_id)  # ty:ignore[invalid-argument-type]
    runner, adapter = make_restart_runner()
    runner._is_user_authorized = lambda _source: False  # ty:ignore[invalid-assignment]
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=True, message_id="warning")
    )

    assert await runner._drain_persisted_drain_inbox() == 0
    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    adapter._send_with_retry.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records()[0]["state"] == "claimed"
    assert pending_records()[0]["reason"] == reason
    assert "recovery_disposition" not in pending_records()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["claimed", "completed", "handled"])
async def test_post_startup_redelivery_cleans_terminal_receipt(
    tmp_path,
    monkeypatch,
    state,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    event = _event(f"{state} redelivery")
    event.message_id = f"{state}-redelivery"
    session_key = build_session_key(event.source)
    claim_producer("current-process", path)
    if state == "handled":
        queue_id, claimed_state, acquired = claim_pre_dispatch_event_result(
            event,
            session_key,
            reason="startup-restore-pre-dispatch",
            path=path,
            producer_token="current-process",
        )
        assert claimed_state == "claimed"
        assert acquired
        queue_id, final_state = finalize_pre_dispatch_event_result(
            event,
            session_key,
            handled=True,
            reason="startup-restore-plugin-handled",
            path=path,
            producer_token="current-process",
        )
        assert final_state == "handled"
    else:
        queue_id = persist_event(
            event,
            session_key,
            reason="test-drain",
            path=path,
            producer_token="current-process",
        )
        assert queue_id
        assert claim_event(
            queue_id,
            path,
            producer_token="current-process",
        )
        if state == "completed":
            assert complete_event(
                queue_id,
                path,
                producer_token="current-process",
            )

    runner, adapter = make_restart_runner()
    runner._drain_inbox_path = path
    runner._drain_inbox_producer_token = "current-process"
    committed_message_ids: set[str] = set()

    async def persist_marker(_session_id, message_id, **_kwargs):
        committed_message_ids.add(message_id)
        return True

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        replay_marker_status_for_session_key=AsyncMock(
            side_effect=lambda _session_key, message_id: (
                message_id in committed_message_ids
            )
        ),
        persist_replay_marker=AsyncMock(side_effect=persist_marker),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    if state == "completed":
        runner._drain_replay_outcomes[queue_id] = "completed"  # ty:ignore[invalid-assignment]

    assert await runner._handle_startup_gate_message(
        event,
        session_key,
        runner._handle_message,
        False,
    )
    task = getattr(runner, "_post_startup_drain_task", None)
    if task is not None:
        await task

    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records(path) == []
    if state == "claimed":
        assert adapter.sent == [runner._DRAIN_UNCERTAIN_NOTICE]  # ty:ignore[unresolved-attribute]
        assert committed_message_ids == {event.message_id}
        adapter.sent.clear()  # ty:ignore[unresolved-attribute]
        assert await runner._handle_startup_gate_message(
            event,
            session_key,
            runner._handle_message,
            False,
        )
        assert adapter.sent == []  # ty:ignore[unresolved-attribute]
    elif state == "completed":
        assert adapter.sent == [runner._DRAIN_COMPLETED_NOTICE]  # ty:ignore[unresolved-attribute]
    else:
        assert adapter.sent == []  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_authenticated_internal_event_replays_without_user_allowlist(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    event.internal = True
    event.durable_ingress = True
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    runner, _adapter = make_restart_runner()
    runner._is_user_authorized = lambda _source: False  # ty:ignore[invalid-assignment]
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 1
    replay = runner._run_startup_resume_event.await_args.args[1]  # ty:ignore[unresolved-attribute]
    assert replay.internal is True
    assert pending_records() == []


@pytest.mark.asyncio
async def test_handler_failure_warns_without_retrying_side_effect(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    runner, adapter = make_restart_runner()
    possible_side_effects: list[str] = []
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=True, message_id="warning-1")
    )

    async def fail_after_side_effect(_adapter, replay, _session_key):
        possible_side_effects.append(replay.text)
        raise RuntimeError("simulated handler crash")

    runner._run_startup_resume_event = AsyncMock(side_effect=fail_after_side_effect)  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 1
    assert possible_side_effects == [event.text]
    assert pending_records() == []
    warning = adapter._send_with_retry.await_args.kwargs["content"]  # ty:ignore[unresolved-attribute]
    assert "could not confirm completion" in warning
    assert "did not retry" in warning
    assert event.text not in warning
    runner._async_session_store.persist_replay_marker.assert_awaited_once()
    marker_args = (
        runner._async_session_store.persist_replay_marker.await_args
    )
    assert marker_args.args == ("session-1", event.message_id)
    assert isinstance(marker_args.kwargs["timestamp"], float)


@pytest.mark.asyncio
async def test_completion_failure_marks_ambiguity_before_ack(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    runner, adapter = make_restart_runner()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=True, message_id="warning")
    )
    monkeypatch.setattr(gateway_run, "complete_drain_event", lambda *_a, **_k: False)

    assert await runner._drain_persisted_drain_inbox() == 1
    runner._async_session_store.persist_replay_marker.assert_awaited_once()
    assert pending_records() == []


@pytest.mark.asyncio
async def test_live_claim_redelivery_does_not_request_resend(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    session_key = build_session_key(event.source)
    persist_event(event, session_key, reason="test-drain")
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    handler_entered = asyncio.Event()
    allow_handler = asyncio.Event()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )

    async def hold_handler(_adapter, _event, _session_key):
        handler_entered.set()
        await allow_handler.wait()

    runner._run_startup_resume_event = AsyncMock(side_effect=hold_handler)  # ty:ignore[invalid-assignment]
    drain_task = asyncio.create_task(runner._drain_persisted_drain_inbox())
    await handler_entered.wait()
    assert pending_records()[0]["state"] == "claimed"

    redelivery = _event("redelivery")
    response = await runner._persist_startup_gate_event(
        redelivery,
        session_key,
    )

    assert "already processing" in response  # ty:ignore[unsupported-operator]
    assert "resend" not in response.lower()  # ty:ignore[unresolved-attribute]

    allow_handler.set()
    assert await drain_task == 1
    assert pending_records() == []


@pytest.mark.asyncio
async def test_producer_handoff_waits_for_live_replay(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(drain_inbox, "_LOCK_TIMEOUT_SECONDS", 0.05)
    path = inbox_path()
    claim_producer("old-process", path)
    event = _event()
    session_key = build_session_key(event.source)
    assert persist_event(
        event,
        session_key,
        reason="test-drain",
        path=path,
        producer_token="old-process",
    )
    runner, _adapter = make_restart_runner()
    runner._drain_inbox_path = path
    runner._drain_inbox_producer_token = "old-process"
    handler_entered = asyncio.Event()
    allow_handler = asyncio.Event()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )

    async def hold_handler(_adapter, _event, _session_key):
        handler_entered.set()
        await allow_handler.wait()

    runner._run_startup_resume_event = AsyncMock(side_effect=hold_handler)  # ty:ignore[invalid-assignment]
    drain_task = asyncio.create_task(runner._drain_persisted_drain_inbox())
    await handler_entered.wait()

    context = multiprocessing.get_context("fork")
    handoff_done = context.Event()
    handoff_process = context.Process(
        target=_claim_producer_in_process,
        args=(str(path), "new-process", handoff_done),
    )
    handoff_process.start()
    try:
        assert not await asyncio.to_thread(handoff_done.wait, 0.2)
        allow_handler.set()
        assert await drain_task == 1
        assert await asyncio.to_thread(handoff_done.wait, 10)
    finally:
        allow_handler.set()
        handoff_process.join(timeout=10)
        if handoff_process.is_alive():
            handoff_process.terminate()
            handoff_process.join(timeout=10)

    assert handoff_process.exitcode == 0
    assert pending_records(path) == []


@pytest.mark.asyncio
async def test_cancelled_replay_lease_acquisition_keeps_previous_producer(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    claim_producer("old-process", path)
    held_lease = acquire_replay_lease(path)
    runner, _adapter = make_restart_runner()
    runner._drain_inbox_path = path
    runner._drain_inbox_producer_token = "cancelled-process"
    drain_task = asyncio.create_task(runner._drain_persisted_drain_inbox())
    await asyncio.sleep(0.05)

    drain_task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await drain_task
    finally:
        release_replay_lease(held_lease)

    event = _event("previous producer remains active")
    event.message_id = "previous-producer-message"
    session_key = build_session_key(event.source)
    assert persist_event(
        event,
        session_key,
        reason="old-process",
        path=path,
        producer_token="old-process",
    )
    rejected = _event("cancelled producer cannot append")
    rejected.message_id = "cancelled-producer-message"
    assert persist_event(
        rejected,
        session_key,
        reason="cancelled-process",
        path=path,
        producer_token="cancelled-process",
    ) is None


def test_cancelled_producer_rollback_holds_lease_until_fsync_succeeds(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = inbox_path()
    claim_producer("old-process", path)
    lease = acquire_producer_replay_lease("cancelled-process", path)
    real_fsync = drain_inbox.os.fsync
    failed_once = threading.Event()
    allow_retry = threading.Event()
    rollback_done = threading.Event()
    append_done = threading.Event()
    append_result = {}
    fsync_calls = 0

    def flaky_fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            failed_once.set()
            raise OSError("simulated rollback fsync failure")
        if not allow_retry.wait(timeout=10):
            raise TimeoutError("rollback retry was not released")
        return real_fsync(fd)

    monkeypatch.setattr(drain_inbox.os, "fsync", flaky_fsync)

    def rollback():
        cancel_producer_replay_lease(lease)
        rollback_done.set()

    event = _event("old producer append")
    event.message_id = "old-producer-append"
    session_key = build_session_key(event.source)

    def append():
        append_result["queue_id"] = persist_event(
            event,
            session_key,
            reason="old-process",
            path=path,
            producer_token="old-process",
        )
        append_done.set()

    rollback_thread = threading.Thread(target=rollback)
    append_thread = threading.Thread(target=append)
    rollback_thread.start()
    try:
        assert failed_once.wait(timeout=10)
        append_thread.start()
        assert not append_done.wait(timeout=0.1)
    finally:
        allow_retry.set()
        rollback_thread.join(timeout=10)
        if append_thread.ident is not None:
            append_thread.join(timeout=10)

    assert rollback_done.is_set()
    assert append_done.is_set()
    assert append_result["queue_id"]


@pytest.mark.asyncio
async def test_replay_marker_lookup_failure_does_not_run_or_ack(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(event, build_session_key(event.source), reason="test-drain")
    runner, _adapter = make_restart_runner()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(
            side_effect=OSError("simulated marker lookup failure")
        ),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 0
    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    runner._async_session_store.persist_replay_marker.assert_not_awaited()
    assert len(pending_records()) == 1


@pytest.mark.asyncio
async def test_replay_marker_write_failure_stays_completed_across_restart(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(event, build_session_key(event.source), reason="test-drain")
    runner, _adapter = make_restart_runner()

    async def fail_marker(*_args, **_kwargs):
        assert pending_records()[0]["state"] == "completed"
        raise OSError("simulated marker write failure")

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(side_effect=fail_marker),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 0
    runner._run_startup_resume_event.assert_awaited_once()  # ty:ignore[unresolved-attribute]
    assert pending_records()[0]["state"] == "completed"

    next_runner, _adapter = make_restart_runner()
    next_runner._async_session_store = SimpleNamespace(
        _store=next_runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    next_runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await next_runner._drain_persisted_drain_inbox() == 1
    next_runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_crash_after_claim_warns_without_handler_until_accepted(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    assert claim_event(queue_id)  # ty:ignore[invalid-argument-type]

    runner, adapter = make_restart_runner()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=False, error="offline")
    )

    assert await runner._drain_persisted_drain_inbox() == 0
    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    runner._async_session_store.persist_replay_marker.assert_not_awaited()
    assert pending_records()[0]["state"] == "claimed"

    next_runner, next_adapter = make_restart_runner()
    next_runner._async_session_store = SimpleNamespace(
        _store=next_runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=False),
    )
    next_runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    next_adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=True, message_id="warning-2")
    )

    assert await next_runner._drain_persisted_drain_inbox() == 0
    next_runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records()[0]["state"] == "claimed"

    final_runner, final_adapter = make_restart_runner()
    final_runner._async_session_store = SimpleNamespace(
        _store=final_runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    final_runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]
    final_adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=True, message_id="warning-3")
    )

    assert await final_runner._drain_persisted_drain_inbox() == 1
    final_runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_completed_row_is_cleaned_without_handler_rerun(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    queue_id = persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    assert claim_event(queue_id)  # ty:ignore[invalid-argument-type]
    assert complete_event(queue_id)  # ty:ignore[invalid-argument-type]
    runner, _adapter = make_restart_runner()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )
    runner._run_startup_resume_event = AsyncMock()  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 1
    runner._run_startup_resume_event.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_outbound_failure_commits_completed_replay(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, adapter = make_restart_runner()
    side_effects: list[str] = []

    async def handler(event):
        side_effects.append(event.text)
        return "response"

    adapter.set_message_handler(handler)
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=False, error="simulated delivery failure")
    )
    event = _event()
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(return_value=False),
        persist_replay_marker=AsyncMock(return_value=True),
    )

    assert await runner._drain_persisted_drain_inbox() == 1

    assert side_effects == [event.text]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_replay_awaits_task_returned_after_topic_recovery():
    runner, adapter = make_restart_runner()
    handler_done = asyncio.Event()

    async def handler(_event):
        await asyncio.sleep(0)
        handler_done.set()
        return None

    adapter.set_message_handler(handler)
    adapter.set_topic_recovery_fn(lambda _source: "recovered-topic")
    event = _event()
    event.reply_to_message_id = "reply-anchor"
    setattr(event, "_hermes_startup_restore_replay", True)
    original_session_key = build_session_key(event.source)

    await runner._run_startup_resume_event(
        adapter,
        event,
        original_session_key,
    )

    assert handler_done.is_set()
    assert event.source.thread_id == "recovered-topic"
    assert event._hermes_handler_succeeded is True  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_generic_golden_checkin_does_not_require_replay_flag():
    runner, adapter = make_restart_runner()
    event = MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        internal=True,
    )
    task = asyncio.create_task(asyncio.sleep(0))
    adapter.handle_message = AsyncMock(return_value=task)  # ty:ignore[invalid-assignment]

    await runner._run_startup_resume_event(
        adapter,
        event,
        build_session_key(event.source),
    )

    assert not hasattr(event, "_hermes_handler_succeeded")


@pytest.mark.asyncio
async def test_replay_rescans_for_rows_appended_during_drain(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = _event("first accepted message")
    second = _event("late accepted message")
    second.message_id = "message-late"
    session_key = build_session_key(first.source)
    persist_event(first, session_key, reason="test-drain")

    runner, _adapter = make_restart_runner()
    committed_message_ids: set[str] = set()
    replayed: list[str] = []

    async def persist_marker(_session_id, message_id, **_kwargs):
        committed_message_ids.add(str(message_id or ""))
        return True

    async def replay(_adapter, event, _session_key):
        replayed.append(event.text)
        if event.text == first.text:
            assert persist_event(
                second,
                session_key,
                reason="late-append",
            )

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1")
        ),
        replay_marker_status=AsyncMock(
            side_effect=lambda _session_id, message_id: (
                message_id in committed_message_ids
            )
        ),
        persist_replay_marker=AsyncMock(side_effect=persist_marker),
    )
    runner._run_startup_resume_event = AsyncMock(side_effect=replay)  # ty:ignore[invalid-assignment]

    assert await runner._drain_persisted_drain_inbox() == 2
    assert replayed == [first.text, second.text]
    assert pending_records() == []


@pytest.mark.asyncio
async def test_startup_read_failure_keeps_restore_gate_closed():
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._drain_persisted_drain_inbox = AsyncMock(  # ty:ignore[invalid-assignment]
        side_effect=OSError("simulated read failure")
    )
    runner._schedule_resume_pending_sessions = AsyncMock()  # ty:ignore[invalid-assignment]

    with pytest.raises(OSError, match="simulated read failure"):
        await runner._finish_startup_restore()

    assert runner._startup_restore_in_progress is True
    runner._schedule_resume_pending_sessions.assert_not_awaited()  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_startup_drains_prior_process_before_new_arrivals():
    runner, _adapter = make_restart_runner()
    runner._startup_restore_tasks = []
    runner._startup_restore_in_progress = True
    order = []
    runner._drain_persisted_drain_inbox = AsyncMock(  # ty:ignore[invalid-assignment]
        side_effect=lambda _attempted: order.append("durable") or 1
    )
    runner._schedule_resume_pending_sessions = (  # ty:ignore[invalid-assignment]
        lambda: order.append("generic-checkin") or 0
    )

    await runner._finish_startup_restore()

    assert order == [
        "durable",
        "generic-checkin",
        "durable",
    ]
    assert runner._startup_restore_in_progress is False


def test_queue_file_is_jsonl_for_operator_inspection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = _event()
    persist_event(
        event,
        build_session_key(event.source),
        reason="test-drain",
    )

    row = json.loads(inbox_path().read_text(encoding="utf-8"))
    assert row["text"] == event.text
    assert row["queue_id"]
