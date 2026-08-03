"""Tests for the resume_pending session continuity path.

Covers the behaviour introduced to fix the ``Gateway shutting down ...
task will be interrupted`` follow-up bug (spec: PR #11852, builds on
PRs #9850, #9934, #7536):

1. When a gateway restart drain times out and agents are force-interrupted,
   the affected sessions are flagged ``resume_pending=True`` — not
   ``suspended`` — so the next user message on the same session_key
   auto-resumes from the existing transcript instead of getting routed
   through ``suspend_recently_active()`` and converted into a fresh
   session.

2. ``suspended=True`` (from ``/stop`` or stuck-loop escalation) still
   wins over ``resume_pending`` — the forced-wipe path is preserved.

3. The restart-resume system note injected into the next user message is
   a superset of the existing tool-tail auto-continue note (from
   PR #9934), using session-entry metadata rather than just transcript
   shape so it fires even when the interrupted transcript does NOT end
   with a ``tool`` role.

4. The existing ``.restart_failure_counts`` stuck-loop counter from
   PR #7536 remains the single source of escalation — no parallel
   counter is added on ``SessionEntry``.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.drain_inbox import event_queue_id
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import (
    _AGENT_PENDING_SENTINEL,
    _auto_continue_freshness_window,
    _coerce_gateway_timestamp,
    _is_fresh_gateway_interruption,
    _last_transcript_timestamp,
    _should_clear_resume_pending_after_turn,
    GatewayRunner,
    build_resume_recovery_note,
)
from gateway.session import SessionEntry, SessionSource, SessionStore
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_resume_pending_is_cleared_only_after_successful_turn():
    """Interrupted/failed drain results must keep the restart recovery marker.

    Regression for dogfood failure: during gateway restart the interrupted run
    returned an empty final response and was normalized into a user-facing
    fallback, but the gateway cleared ``resume_pending`` before startup could
    auto-resume it.
    """
    assert _should_clear_resume_pending_after_turn({"final_response": "done"}) is True
    assert _should_clear_resume_pending_after_turn({"completed": True}) is True
    assert _should_clear_resume_pending_after_turn({"interrupted": True}) is False
    assert _should_clear_resume_pending_after_turn({"completed": False}) is False
    assert _should_clear_resume_pending_after_turn({"failed": True}) is False
    assert _should_clear_resume_pending_after_turn({"partial": True}) is False
    assert _should_clear_resume_pending_after_turn({"error": "boom"}) is False


def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id)


def _make_store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _build_agent_history(history: list) -> list:
    """Mirror gateway/run.py's ``history → agent_history`` conversion.

    This is the transformation that strips ``timestamp`` off tool/tool_call
    rows before the agent sees them.  Tests that check the freshness gate
    must go through this conversion so they exercise the *real* data the
    note-injection code sees.
    """
    agent_history: list = []
    for msg in history:
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            agent_history.append({k: v for k, v in msg.items() if k != "timestamp"})
        else:
            content = msg.get("content")
            if content:
                agent_history.append({"role": role, "content": content})
    return agent_history


def _simulate_note_injection(
    history: list,
    user_message: str,
    resume_entry: SessionEntry | None,
    *,
    agent_history: list | None = None,
    window_secs: float | None = None,
) -> str:
    """Mirror the note-injection logic in gateway/run.py _run_agent().

    The freshness signal reads ``history[-1].timestamp`` (the raw transcript
    row), NOT ``agent_history[-1].timestamp`` (which has been stripped).
    Tests pass the raw ``history`` — ``agent_history`` is derived from it
    via the real conversion if not supplied explicitly.
    """
    if agent_history is None:
        agent_history = _build_agent_history(history)

    window = (
        float(window_secs)
        if window_secs is not None
        else _auto_continue_freshness_window()
    )
    interruption_is_fresh = _is_fresh_gateway_interruption(
        _last_transcript_timestamp(history),
        window_secs=window,
    )

    message = user_message
    resume_mark_is_fresh = False
    if resume_entry is not None and getattr(resume_entry, "resume_pending", False):
        resume_mark_is_fresh = _is_fresh_gateway_interruption(
            getattr(resume_entry, "last_resume_marked_at", None),
            window_secs=window,
        )
    is_resume_pending = bool(
        resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
        and (interruption_is_fresh or resume_mark_is_fresh)
    )
    has_fresh_tool_tail = bool(
        agent_history
        and agent_history[-1].get("role") == "tool"
        and interruption_is_fresh
    )

    if is_resume_pending:
        reason = getattr(resume_entry, "resume_reason", None) or "restart_timeout"
        # Real production note builder — extracted to module scope in
        # gateway/run.py so tests exercise the actual strings.
        message = build_resume_recovery_note(reason, message)
    elif has_fresh_tool_tail:
        message = (
            "[System note: A new message has arrived. The conversation "
            "history contains pending tool outputs from an interrupted turn. "
            "IGNORE those pending results. Address the user's NEW message "
            "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
            + message
        )

    # Empty-turn safety net: mirrors gateway/run.py — a blank
    # auto-resume turn on a resume_pending session must never reach the model.
    if (
        isinstance(message, str)
        and not message.strip()
        and resume_entry is not None
        and getattr(resume_entry, "resume_pending", False)
    ):
        sn_reason = getattr(resume_entry, "resume_reason", None) or "restart_timeout"
        message = build_resume_recovery_note(sn_reason, "")
    return message


# ---------------------------------------------------------------------------
# SessionEntry field + serialization
# ---------------------------------------------------------------------------


class TestSessionEntryResumeFields:
    def test_defaults(self):
        now = datetime.now()
        entry = SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
        )
        assert entry.resume_pending is False
        assert entry.resume_reason is None
        assert entry.last_resume_marked_at is None


# ---------------------------------------------------------------------------
# SessionStore.mark_resume_pending / clear_resume_pending
# ---------------------------------------------------------------------------


class TestMarkResumePending:
    def test_marks_existing_session(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        assert store.mark_resume_pending(entry.session_key) is True
        refreshed = store._entries[entry.session_key]
        assert refreshed.resume_pending is True
        assert refreshed.resume_reason == "restart_timeout"
        assert refreshed.last_resume_marked_at is not None

    def test_custom_reason_persists(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)

        store.mark_resume_pending(entry.session_key, reason="shutdown_timeout")
        assert store._entries[entry.session_key].resume_reason == "shutdown_timeout"


class TestClearResumePending:

    def test_returns_false_when_not_pending(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        # Not marked
        assert store.clear_resume_pending(entry.session_key) is False


# ---------------------------------------------------------------------------
# SessionStore.get_or_create_session resume_pending behaviour
# ---------------------------------------------------------------------------


class TestGetOrCreateResumePending:

    def test_resume_pending_follows_compression_tip(self, tmp_path):
        """Interrupted platform mappings must not stay pinned to compressed roots."""
        store = _make_store(tmp_path)
        source = _make_source(
            platform=Platform.WEIXIN,
            chat_id="wx-chat",
            user_id="wx-user",
        )
        first = store.get_or_create_session(source)
        original_sid = first.session_id
        store.mark_resume_pending(first.session_key)

        with patch.object(
            store, "_compression_tip_for_session_id", return_value="child-session"
        ) as mock_tip:
            second = store.get_or_create_session(source)

        assert second.session_id == "child-session"
        assert second.resume_pending is True
        mock_tip.assert_called_with(original_sid)


# ---------------------------------------------------------------------------
# SessionStore.suspend_recently_active skip behaviour
# ---------------------------------------------------------------------------


class TestSuspendRecentlyActiveSkipsResumePending:
    def test_resume_pending_entries_not_suspended(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key)

        count = store.suspend_recently_active()
        assert count == 0
        e = store._entries[entry.session_key]
        assert e.suspended is False
        assert e.resume_pending is True


# ---------------------------------------------------------------------------
# Restart-resume system-note injection
# ---------------------------------------------------------------------------


class TestResumePendingSystemNote:
    def _pending_entry(self, reason="restart_timeout") -> SessionEntry:
        now = datetime.now()
        return SessionEntry(
            session_key="agent:main:telegram:dm:1",
            session_id="sid",
            created_at=now,
            updated_at=now,
            resume_pending=True,
            resume_reason=reason,
            last_resume_marked_at=now,
        )

    def test_resume_pending_restart_note_mentions_restart(self):
        entry = self._pending_entry(reason="restart_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="what happened?",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "NEW message" in result
        assert "Do NOT re-execute" in result
        assert "what happened?" in result

    def test_resume_pending_shutdown_note_mentions_shutdown(self):
        entry = self._pending_entry(reason="shutdown_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="ping",
            resume_entry=entry,
        )
        assert "gateway shutdown" in result

    def test_empty_message_interactive_note_checks_in(self):
        note = build_resume_recovery_note("restart_timeout", "", interactive=True)
        assert "ask what they would like to do next" in note
        assert "CONTINUE the interrupted task" not in note
        assert "skip any unfinished work" in note

    def test_empty_message_noninteractive_note_does_not_continue_task(self):
        note = build_resume_recovery_note("restart_timeout", "", interactive=False)
        assert "available for a new instruction" in note
        assert "CONTINUE the interrupted task" not in note
        assert "continue unfinished work" in note

    def test_empty_message_guidance_is_safe_across_surfaces(self):
        interactive_note = build_resume_recovery_note(
            "restart_timeout", "", interactive=True
        )
        noninteractive_note = build_resume_recovery_note(
            "restart_timeout", "", interactive=False
        )
        assert "CONTINUE the interrupted task" not in interactive_note
        assert "CONTINUE the interrupted task" not in noninteractive_note


    def test_resume_pending_fires_without_tool_tail(self):
        """Key improvement over PR #9934: the restart-resume note fires
        even when the transcript's last role is NOT ``tool``."""
        entry = self._pending_entry()
        history = [
            {"role": "user", "content": "run a long thing", "timestamp": time.time() - 10},
            {"role": "assistant", "content": "ok, starting...", "timestamp": time.time()},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=entry)
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "NEW message" in result


    def test_no_resume_pending_preserves_tool_tail_note(self):
        """Regression: the old PR #9934 tool-tail behaviour is unchanged."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 1},
            {"role": "tool", "tool_call_id": "c1", "content": "result",
             "timestamp": time.time()},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

    def test_stale_resume_pending_does_not_inject_restart_note(self):
        """Old restart markers must not revive an unrelated stale task.

        The transcript's last row is from an hour ago — well outside the
        default 1h freshness window (fixture uses window=1800 to exercise
        the stale path without tying the test to the production default).
        """
        entry = self._pending_entry()
        entry.last_resume_marked_at = datetime.now() - timedelta(hours=1)

        history = [
            {"role": "assistant", "content": "old in progress",
             "timestamp": time.time() - 3600},
        ]
        result = _simulate_note_injection(
            history=history,
            user_message="start a new task",
            resume_entry=entry,
            window_secs=1800,
        )
        assert result == "start a new task"


    def test_stale_tool_tail_does_not_inject_auto_continue_note(self):
        """The core bug fix: stale tool-tail must not revive a dead task.

        Uses window_secs=1800 (30 min) to verify the gate fires at 1h —
        keeps the test stable regardless of the production default.
        """
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 3601},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "stale result",
                "timestamp": time.time() - 3600,
            },
        ]
        result = _simulate_note_injection(
            history,
            "start a new task",
            resume_entry=None,
            window_secs=1800,
        )
        assert result == "start a new task"

    def test_stale_tool_tail_with_production_data_shape(self):
        """Regression guard for #16802: exercise the REAL production path
        where ``agent_history`` has been stripped of timestamps.

        The original PR #16802 fix read ``agent_history[-1].get("timestamp")``
        — which is always ``None`` at runtime because the gateway strips
        ``timestamp`` off tool/tool_call rows in ``history → agent_history``.
        This test builds a stale history, runs it through the real
        ``_build_agent_history`` conversion, then asserts:

          1. The stripped ``agent_history`` carries NO timestamp (protects
             against someone "fixing" the original PR by re-adding the
             stripped field — which would break the API contract).
          2. The freshness gate still correctly classifies the transcript
             as stale because the signal is read from ``history`` BEFORE
             the strip.
          3. No auto-continue note is injected.
        """
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 7201},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "stale result",
                "timestamp": time.time() - 7200,  # 2 hours old
            },
        ]
        agent_history = _build_agent_history(history)

        # Invariant 1: strip contract preserved
        assert agent_history[-1]["role"] == "tool"
        assert "timestamp" not in agent_history[-1], (
            "agent_history tool rows must NOT carry a timestamp — the "
            "freshness gate must read from raw history, not agent_history"
        )

        # Invariant 2+3: stale classification, no note injection
        result = _simulate_note_injection(
            history,
            "start a new task",
            resume_entry=None,
            agent_history=agent_history,
        )
        assert result == "start a new task"

    def test_freshness_gate_disabled_via_zero_window(self):
        """window_secs=0 restores pre-fix behaviour (always inject)."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ], "timestamp": time.time() - 86400},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "day-old result",
                "timestamp": time.time() - 86400,  # 24 hours old
            },
        ]
        result = _simulate_note_injection(
            history, "ping", resume_entry=None, window_secs=0,
        )
        assert "[System note:" in result
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

    def test_legacy_history_without_timestamps_still_injects(self):
        """Transcripts predating timestamp persistence must keep the old
        behaviour — freshness unknown → treat as fresh."""
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert "[System note:" in result
        assert "pending tool outputs" in result
        assert "Do NOT re-execute" in result

    def test_no_note_when_nothing_to_resume(self):
        history = [
            {"role": "user", "content": "hello", "timestamp": time.time() - 2},
            {"role": "assistant", "content": "hi", "timestamp": time.time() - 1},
        ]
        result = _simulate_note_injection(history, "ping", resume_entry=None)
        assert result == "ping"

    def test_resume_pending_note_warns_against_reexecuting_restart(self):
        """The resume-pending note tells the model any restart/shutdown
        command in the history already ran and must not be re-executed or
        verified — the cognitive backstop to the source-level tail strip.
        """
        entry = self._pending_entry(reason="restart_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="restarted!",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "back online" in result
        assert "already" in result and "do NOT re-execute or verify" in result
        assert "restarted!" in result

    def test_resume_pending_empty_message_checks_in_without_continuing(self):
        entry = self._pending_entry(reason="restart_timeout")
        result = _simulate_note_injection(
            history=[
                {"role": "assistant", "content": "in progress", "timestamp": time.time()},
            ],
            user_message="",
            resume_entry=entry,
        )
        assert "[System note:" in result
        assert "gateway restart" in result
        assert "CONTINUE the interrupted task" not in result
        assert "ask what they would like to do next" in result
        assert "do NOT re-execute or verify" in result
        # No phantom "NEW message" instruction when there is no new message.
        assert "NEW message" not in result
        # Nothing appended after the closing bracket (no empty user text).
        assert result.rstrip().endswith("]")


# ---------------------------------------------------------------------------
# Freshness helpers
# ---------------------------------------------------------------------------


class TestFreshnessHelpers:


    def test_coerce_iso_string(self):
        iso = "2026-04-18T12:00:00+00:00"
        expected = datetime.fromisoformat(iso).timestamp()
        assert _coerce_gateway_timestamp(iso) == pytest.approx(expected, abs=1e-3)


    def test_coerce_rejects_garbage(self):
        assert _coerce_gateway_timestamp(None) is None
        assert _coerce_gateway_timestamp("") is None
        assert _coerce_gateway_timestamp("not-a-timestamp") is None
        assert _coerce_gateway_timestamp(True) is None  # bool rejected
        assert _coerce_gateway_timestamp(False) is None
        assert _coerce_gateway_timestamp([1, 2, 3]) is None


    def test_is_fresh_window_bounds(self):
        now = 1_700_000_000.0
        # 1h window, 30min old → fresh
        assert _is_fresh_gateway_interruption(
            now - 1800, now=now, window_secs=3600,
        ) is True
        # 1h window, 2h old → stale
        assert _is_fresh_gateway_interruption(
            now - 7200, now=now, window_secs=3600,
        ) is False
        # 1h window, exactly at boundary → fresh (<=)
        assert _is_fresh_gateway_interruption(
            now - 3600, now=now, window_secs=3600,
        ) is True


    def test_last_transcript_timestamp_skips_meta(self):
        history = [
            {"role": "user", "content": "hi", "timestamp": 100.0},
            {"role": "assistant", "content": "hey", "timestamp": 200.0},
            {"role": "session_meta", "content": "tools:{}", "timestamp": 999.0},
            {"role": "system", "content": "ignore", "timestamp": 999.0},
        ]
        assert _last_transcript_timestamp(history) == 200.0


    def test_auto_continue_freshness_window_reads_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "7200")
        assert _auto_continue_freshness_window() == 7200.0

    def test_auto_continue_freshness_window_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_AUTO_CONTINUE_FRESHNESS", raising=False)
        # Default is 1 hour
        assert _auto_continue_freshness_window() == 3600.0


# ---------------------------------------------------------------------------
# Drain-timeout path marks sessions resume_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_timeout_marks_resume_pending():
    """End-to-end: a drain timeout during gateway stop should flag every
    active session as resume_pending BEFORE the interrupt fires, so the
    next startup's suspend_recently_active() does not destroy them."""
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_drain_timeout = 0.05

    running_agent = MagicMock()
    session_key_one = "agent:main:telegram:dm:A"
    session_key_two = "agent:main:telegram:dm:B"
    runner._running_agents = {
        session_key_one: running_agent,
        session_key_two: MagicMock(),
    }

    # Plug a mock session_store that records marks.
    session_store = MagicMock()
    session_store.mark_resume_pending = MagicMock(return_value=True)
    runner.session_store = session_store

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        await runner.stop()

    # Both active sessions were marked with the shutdown_timeout reason.
    calls = session_store.mark_resume_pending.call_args_list
    marked = {args[0][0] for args in calls}
    assert marked == {session_key_one, session_key_two}
    for args in calls:
        assert args[0][1] == "shutdown_timeout"


# ---------------------------------------------------------------------------
# Gateway startup auto-resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_auto_resume_skips_unauthorized_owner():
    """A resume-pending session whose owner is no longer authorized under the
    current allowlist must not receive a synthesized agent turn on restart.

    Auto-resume dispatches a full agent turn without going through the normal
    inbound-message auth gate, so it re-checks _is_user_authorized here
    (issue #23778).  An unauthorized owner is skipped WITHOUT claiming a
    _running_agents slot or persisting one — the slot claim happens only
    after this gate passes.
    """
    runner, adapter = make_restart_runner()
    runner._is_user_authorized = lambda _source: False
    runner._persist_active_agents = MagicMock()
    source = make_restart_source(chat_id="revoked-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:revoked-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 0
    adapter.handle_message.assert_not_called()
    # No slot was claimed and nothing was persisted for the skipped session.
    assert pending_entry.session_key not in runner._running_agents
    runner._persist_active_agents.assert_not_called()


@pytest.mark.asyncio
async def test_reconnect_reschedule_is_platform_scoped():
    """The platform filter limits the pass to that platform's sessions, so
    reconnecting one platform never resumes another's pending session."""
    runner, adapter = make_restart_runner()
    tg_source = make_restart_source(chat_id="tg-chat")
    discord_source = SessionSource(
        platform=Platform.DISCORD, chat_id="dc-chat", chat_type="dm", user_id="u1"
    )
    tg_entry = SessionEntry(
        session_key="agent:main:telegram:dm:tg-chat",
        session_id="sid-tg",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=tg_source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    discord_entry = SessionEntry(
        session_key="agent:main:discord:dm:dc-chat",
        session_id="sid-dc",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=discord_source,
        platform=Platform.DISCORD,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {
        tg_entry.session_key: tg_entry,
        discord_entry.session_key: discord_entry,
    }
    adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}

    scheduled = runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)
    await asyncio.sleep(0)

    # Only the telegram session is resumed; the discord session waits for its
    # own reconnect.
    assert scheduled == 1
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source == tg_source


@pytest.mark.asyncio
async def test_auto_resume_skips_sessions_with_running_agent():
    """A session already being resumed (agent in-flight) is not scheduled
    again — guards against a double resume when a platform reconnects while a
    startup-scheduled resume is still running."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="inflight-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:inflight-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}
    runner._running_agents = {pending_entry.session_key: object()}
    adapter.handle_message = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)

    assert scheduled == 0
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_startup_restore_gate_persists_real_inbound_messages(
    tmp_path,
    monkeypatch,
):
    """Real inbound messages are durable before startup acknowledges them."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-message",
    )

    result = await adapter.handle_message(inbound)

    assert result is None
    assert any("saved" in message for message in adapter.sent)  # ty:ignore[unresolved-attribute]
    from gateway.drain_inbox import pending_records

    records = pending_records()
    assert [record["text"] for record in records] == ["hello"]
    assert records[0]["message_id"] == "startup-message"
    assert records[0]["pre_dispatch_attempted"] is True


@pytest.mark.asyncio
async def test_startup_restore_gate_rejects_unstable_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
    )

    result = await adapter.handle_message(inbound)

    assert result is None
    assert any("could not safely claim" in message for message in adapter.sent)  # ty:ignore[unresolved-attribute]
    from gateway.drain_inbox import pending_records

    assert pending_records() == []


@pytest.mark.asyncio
async def test_startup_restore_gate_does_not_persist_unauthorized_message():
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._is_user_authorized = lambda _source: False  # ty:ignore[invalid-assignment]
    runner._persist_drain_event_result = AsyncMock()  # ty:ignore[invalid-assignment]
    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-message",
    )

    assert await runner._handle_message(inbound) is None
    runner._persist_drain_event_result.assert_not_awaited()  # ty:ignore[unresolved-attribute]


class _EarlyListenerAdapter(BasePlatformAdapter):
    def __init__(self, platform):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.gate_seen = False

    async def connect(self, *, is_reconnect=False):
        assert self._message_handler is not None
        assert self._startup_gate_handler is not None
        event = MessageEvent(
            text="early inbound",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=self.platform,
                chat_id="early-chat",
                user_id="early-user",
            ),
            message_id=f"{self.platform.value}-early",
        )
        assert await self.handle_message(event) is None
        assert self.gate_seen is True
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.QQBOT, Platform.WEIXIN])
async def test_early_listener_has_startup_gate_before_connect(
    platform,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            platform: PlatformConfig(enabled=True, token="test"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _EarlyListenerAdapter(platform)

    async def gate(_event, _session_key, _session_was_busy):
        adapter.gate_seen = True
        return True

    monkeypatch.setattr(runner, "_create_adapter", lambda *_args: adapter)
    monkeypatch.setattr(runner, "_make_startup_gate_handler", lambda _handler: gate)
    monkeypatch.setattr(runner.hooks, "discover_and_load", lambda: None)
    monkeypatch.setattr(runner.hooks, "emit", AsyncMock())

    assert await runner.start() is True
    assert adapter.gate_seen is True


@pytest.mark.asyncio
async def test_startup_restore_busy_session_persists_before_memory_queue(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    source = make_restart_source(chat_id="restore-chat")
    session_key = runner._session_key_for_source(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()  # ty:ignore[invalid-assignment]
    inbound = MessageEvent(
        text="hello while busy",
        message_type=MessageType.TEXT,
        source=source,
        message_id="startup-busy-message",
    )

    await adapter.handle_message(inbound)

    from gateway.drain_inbox import pending_records

    records = pending_records()
    assert [record["text"] for record in records] == ["hello while busy"]
    assert records[0]["pre_dispatch_attempted"] is False
    assert adapter._pending_messages == {}
    assert any("saved" in message for message in adapter.sent)  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_startup_restore_waits_for_resume_before_draining_inbound():
    """Queued inbound turns replay only after startup resume tasks finish."""
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    source = make_restart_source(chat_id="restore-chat")
    session_key = runner._session_key_for_source(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()  # ty:ignore[invalid-assignment]
    inbound = MessageEvent(
        text="hello while busy",
        message_type=MessageType.TEXT,
        source=source,
        message_id="startup-busy-message",
    )

    await adapter.handle_message(inbound)

    from gateway.drain_inbox import pending_records

    records = pending_records()
    assert [record["text"] for record in records] == ["hello while busy"]
    assert records[0]["pre_dispatch_attempted"] is False
    assert adapter._pending_messages == {}
    assert any("saved" in message for message in adapter.sent)  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_startup_gate_persist_before_final_barrier_is_drained():
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._schedule_resume_pending_sessions = lambda: 0  # ty:ignore[invalid-assignment]
    persist_entered = asyncio.Event()
    allow_persist = asyncio.Event()
    order: list[str] = []

    async def persist_before_finish(*_args, **_kwargs):
        order.append("persist-start")
        persist_entered.set()
        await allow_persist.wait()
        order.append("persist-finish")
        return "queue-id", "queued"

    async def drain(_attempted):
        order.append("drain")
        return 0

    runner._persist_drain_event_result = persist_before_finish  # ty:ignore[invalid-assignment]
    runner._drain_persisted_drain_inbox = AsyncMock(side_effect=drain)  # ty:ignore[invalid-assignment]
    inbound = MessageEvent(
        text="before closure",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="before-closure",
    )

    persist_task = asyncio.create_task(
        runner._persist_startup_gate_event(
            inbound,
            runner._session_key_for_source(inbound.source),
        )
    )
    await persist_entered.wait()
    finish_task = asyncio.create_task(runner._finish_startup_restore())
    await asyncio.sleep(0)
    assert not finish_task.done()

    allow_persist.set()
    assert "saved" in await persist_task  # ty:ignore[unsupported-operator]
    await finish_task

    assert order == [
        "persist-start",
        "drain",
        "persist-finish",
        "drain",
    ]
    assert runner._startup_restore_in_progress is False


@pytest.mark.asyncio
async def test_startup_gate_waits_for_final_barrier_then_dispatches_normally():
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._schedule_resume_pending_sessions = lambda: 0  # ty:ignore[invalid-assignment]
    final_drain_entered = asyncio.Event()
    allow_final_drain = asyncio.Event()
    drain_count = 0

    replay = MessageEvent(
        text="replay",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="replay-message",
    )
    setattr(replay, "_hermes_startup_restore_replay", True)

    async def drain(_attempted):
        nonlocal drain_count
        drain_count += 1
        if drain_count == 2:
            assert await asyncio.wait_for(
                runner._persist_startup_gate_event(
                    replay,
                    runner._session_key_for_source(replay.source),
                ),
                timeout=0.1,
            ) is None
            final_drain_entered.set()
            await allow_final_drain.wait()
        return 0

    runner._drain_persisted_drain_inbox = AsyncMock(side_effect=drain)  # ty:ignore[invalid-assignment]
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id", "queued")
    )
    finish_task = asyncio.create_task(runner._finish_startup_restore())
    await final_drain_entered.wait()

    inbound = MessageEvent(
        text="after closure",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="after-closure",
    )
    persist_task = asyncio.create_task(
        runner._persist_startup_gate_event(
            inbound,
            runner._session_key_for_source(inbound.source),
        )
    )
    await asyncio.sleep(0)
    assert not persist_task.done()
    runner._persist_drain_event_result.assert_not_awaited()  # ty:ignore[unresolved-attribute]

    allow_final_drain.set()
    await finish_task
    assert await persist_task is None
    runner._persist_drain_event_result.assert_not_awaited()  # ty:ignore[unresolved-attribute]
    assert runner._startup_restore_in_progress is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected", "unexpected"),
    [
        ("queued", "saved and will run next", None),
        ("claimed", "did not retry", None),
        ("completed", "already completed", "resend"),
        ("failed", "could not safely save", None),
    ],
)
async def test_startup_gate_uses_existing_row_state_wording(
    state,
    expected,
    unexpected,
):
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id" if state != "failed" else None, state)
    )
    inbound = MessageEvent(
        text="stable redelivery",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="stable-redelivery",
    )

    response = await runner._persist_startup_gate_event(
        inbound,
        runner._session_key_for_source(inbound.source),
    )

    assert expected in response  # ty:ignore[unsupported-operator]
    if unexpected is not None:
        assert unexpected not in response.lower()  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_outcome", "expected"),
    [
        ("running", "already processing"),
        ("completed", "already completed"),
    ],
)
async def test_startup_gate_prefers_local_replay_outcome(
    local_outcome,
    expected,
):
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id", "claimed")
    )
    inbound = MessageEvent(
        text="concurrent stable redelivery",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="concurrent-redelivery",
    )
    session_key = runner._session_key_for_source(inbound.source)
    runner._drain_replay_outcomes[
        event_queue_id(inbound, session_key)
    ] = local_outcome

    response = await runner._persist_startup_gate_event(
        inbound,
        session_key,
    )

    assert expected in response  # ty:ignore[unsupported-operator]
    assert "resend" not in response.lower()  # ty:ignore[unresolved-attribute]
    runner._persist_drain_event_result.assert_not_awaited()  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_authenticated_internal_ingress_uses_durable_startup_gate():
    runner, _adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id", "queued")
    )
    inbound = MessageEvent(
        text="authenticated notification",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="authenticated-internal",
        internal=True,
        durable_ingress=True,
    )

    decision = await runner._persist_startup_gate_event_decision(
        inbound,
        runner._session_key_for_source(inbound.source),
    )

    assert decision is not None
    assert decision.durable_accepted is True
    runner._persist_drain_event_result.assert_awaited_once()  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_startup_gate_callback_failure_propagates_to_transport():
    _runner, adapter = make_restart_runner()

    async def fail_gate(_event, _session_key, _session_was_busy):
        raise OSError("simulated persistence failure")

    adapter.set_startup_gate_handler(fail_gate)
    inbound = MessageEvent(
        text="retry me",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-gate-failure",
    )

    with pytest.raises(OSError, match="simulated persistence failure"):
        await adapter.handle_message(inbound)

    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_startup_gate_releases_barrier_before_delivery():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id", "queued")
    )
    delivery_entered = asyncio.Event()
    allow_delivery = asyncio.Event()

    async def hold_delivery(**_kwargs):
        delivery_entered.set()
        await allow_delivery.wait()
        return SendResult(success=True, message_id="saved-notice")

    adapter._send_with_retry = AsyncMock(side_effect=hold_delivery)  # ty:ignore[invalid-assignment]
    inbound = MessageEvent(
        text="save without blocking closure",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-delivery",
    )
    session_key = runner._session_key_for_source(inbound.source)
    gate_task = asyncio.create_task(
        runner._handle_startup_gate_message(
            inbound,
            session_key,
            AsyncMock(),
            True,
        )
    )
    await delivery_entered.wait()

    barrier = runner._get_startup_restore_barrier()
    await asyncio.wait_for(barrier.acquire(), timeout=0.1)
    barrier.release()

    allow_delivery.set()
    assert await gate_task is True


@pytest.mark.asyncio
async def test_startup_gate_failed_notice_delivery_propagates_to_transport():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._claim_pre_dispatch_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=(None, "failed", False)
    )
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=False, error="offline")
    )
    inbound = MessageEvent(
        text="retry after failed persistence",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-failed-notice",
    )

    with pytest.raises(RuntimeError, match="delivery was not accepted"):
        await adapter.handle_message(inbound)

    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_log_only_admission_failure_requires_transport_retry():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._claim_pre_dispatch_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=(None, "failed", False)
    )
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=True, message_id="local-log")
    )
    inbound = MessageEvent(
        text="retry after failed persistence",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-log-only-failure",
        retry_transport_on_admission_failure=True,
    )

    with pytest.raises(RuntimeError, match="requires transport retry"):
        await adapter.handle_message(inbound)

    adapter._send_with_retry.assert_not_awaited()  # ty:ignore[unresolved-attribute]


@pytest.mark.asyncio
async def test_durable_startup_acceptance_survives_notice_delivery_failure():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id", "queued")
    )
    adapter._send_with_retry = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=SendResult(success=False, error="offline")
    )
    inbound = MessageEvent(
        text="run once after startup",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-durable-no-notice",
    )
    session_key = runner._session_key_for_source(inbound.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()  # ty:ignore[invalid-assignment]

    assert await adapter.handle_message(inbound) is None
    runner._persist_drain_event_result.assert_awaited_once()  # ty:ignore[unresolved-attribute]
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_startup_gate_missing_notice_adapter_propagates_to_transport():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._claim_pre_dispatch_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=(None, "failed", False)
    )
    runner.adapters.clear()
    inbound = MessageEvent(
        text="retry without an adapter route",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="startup-missing-adapter",
    )

    with pytest.raises(RuntimeError, match="adapter is unavailable"):
        await adapter.handle_message(inbound)

    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_startup_arrival_does_not_own_guard_ahead_of_replay():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._busy_text_mode = "queue"
    barrier = runner._get_startup_restore_barrier()
    await barrier.acquire()
    source = make_restart_source(chat_id="restore-chat")
    session_key = runner._session_key_for_source(source)
    replay_started = asyncio.Event()
    allow_replay = asyncio.Event()

    async def hold_replay(_event):
        replay_started.set()
        await allow_replay.wait()

    adapter.set_message_handler(hold_replay)
    inbound = MessageEvent(
        text="arrival before closure",
        message_type=MessageType.TEXT,
        source=source,
        message_id="arrival-before-closure",
    )
    replay = MessageEvent(
        text="saved replay",
        message_type=MessageType.TEXT,
        source=source,
        message_id="saved-replay",
    )
    setattr(replay, "_hermes_startup_restore_replay", True)

    inbound_task = asyncio.create_task(adapter.handle_message(inbound))
    await asyncio.sleep(0)
    assert session_key not in adapter._active_sessions

    replay_task = await adapter.handle_message(replay)
    assert replay_task is not None
    await replay_started.wait()

    runner._startup_restore_in_progress = False
    barrier.release()
    await inbound_task

    assert adapter._pending_messages[session_key] is inbound
    assert adapter._pending_messages[session_key] is not replay

    allow_replay.set()
    await replay_task
    for _ in range(100):
        if not adapter._background_tasks:
            break
        await asyncio.sleep(0.001)
    assert not adapter._background_tasks


@pytest.mark.asyncio
async def test_startup_restore_waits_for_resume_before_final_durable_drain():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._persist_drain_event_result = AsyncMock(  # ty:ignore[invalid-assignment]
        return_value=("queue-id", "queued")
    )
    drain_order: list[str] = []
    runner._drain_persisted_drain_inbox = AsyncMock(  # ty:ignore[invalid-assignment]
        side_effect=lambda _attempted: drain_order.append("durable") or 0
    )

    source = make_restart_source(chat_id="restore-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:restore-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    resume_done = asyncio.Event()
    seen: list[str] = []

    async def complete_resume(event: MessageEvent) -> None:
        await resume_done.wait()
        setattr(event, "_hermes_handler_succeeded", True)

    async def fake_handle_message(event: MessageEvent):
        seen.append("resume-start")
        return asyncio.create_task(complete_resume(event))

    adapter.handle_message = fake_handle_message  # ty:ignore[invalid-assignment]

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="startup-message",
    )
    assert await runner._handle_startup_gate_message(
        inbound,
        runner._session_key_for_source(source),
        runner._handle_message,
        True,
    )
    assert any("saved" in message for message in adapter.sent)  # ty:ignore[unresolved-attribute]
    assert seen == []
    runner._persist_drain_event_result.assert_awaited_once()  # ty:ignore[unresolved-attribute]

    finish_task = asyncio.create_task(runner._finish_startup_restore())
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.001)
    assert seen == ["resume-start"]

    resume_done.set()
    await finish_task

    assert seen == ["resume-start"]
    assert drain_order == ["durable", "durable"]
    assert runner._startup_restore_in_progress is False


def test_startup_restore_has_no_in_memory_queue_path():
    from gateway.run import GatewayRunner

    runner, _adapter = make_restart_runner()

    assert not hasattr(runner, "_startup_restore_queue")
    assert not hasattr(GatewayRunner, "_drain_startup_restore_queue")


# ---------------------------------------------------------------------------
# Shutdown banner wording
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_banner_promises_a_checkin():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner._running_agents["agent:main:telegram:dm:999"] = MagicMock()

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1
    msg = adapter.sent[0]
    assert "restarting" in msg
    assert "check in when the gateway is back" in msg
    assert "Send any message" not in msg


@pytest.mark.asyncio
async def test_restart_notifies_home_channel_even_without_active_sessions():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent == [
        "⚠️ Gateway restarting — Your current task will be interrupted. "
        "I'll check in when the gateway is back."
    ]


@pytest.mark.asyncio
async def test_restart_home_channel_notification_dedupes_active_chat():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    runner._running_agents["agent:main:telegram:dm:999"] = MagicMock()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="999",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_shutdown_banner_promises_a_checkin_on_next_start():
    runner, adapter = make_restart_runner()
    runner._restart_requested = False
    runner._running_agents["agent:main:telegram:dm:999"] = MagicMock()

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent == [  # ty:ignore[unresolved-attribute]
        "⚠️ Gateway shutting down — Your current task will be interrupted. "
        "I'll check in the next time the gateway starts."
    ]


@pytest.mark.asyncio
async def test_restart_home_channel_notification_not_deduped_across_threads():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    session_key = "agent:main:telegram:group:999"
    runner.session_store._entries[session_key] = MagicMock(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="999",
            chat_type="group",
            user_id="u1",
            thread_id="topic-7",
        )
    )
    runner._running_agents[session_key] = MagicMock()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="999",
        name="Ops Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 2
    assert adapter.sent_calls[0][2] == {"thread_id": "topic-7"}
    assert adapter.sent_calls[1][2] is None


# ---------------------------------------------------------------------------
# Stuck-loop escalation integration
# ---------------------------------------------------------------------------


class TestStuckLoopEscalation:
    """The existing .restart_failure_counts counter (PR #7536) remains the
    single source of terminal escalation — no parallel counter on
    SessionEntry was added.  After the configured threshold, the startup
    path flips suspended=True which overrides resume_pending."""

    def test_escalation_via_stuck_loop_counter_overrides_resume_pending(
        self, tmp_path, monkeypatch
    ):
        """Simulate a session that keeps getting restart-interrupted and
        hits the stuck-loop threshold: next startup should force it to
        fresh-session despite resume_pending being set."""
        import json

        from gateway.run import GatewayRunner

        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        store.mark_resume_pending(entry.session_key, reason="restart_timeout")

        # Simulate counter already at threshold (3 consecutive interrupted
        # restarts).  _suspend_stuck_loop_sessions will flip suspended=True.
        counts_file = tmp_path / ".restart_failure_counts"
        counts_file.write_text(json.dumps({entry.session_key: 3}))

        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        suspended_count = GatewayRunner._suspend_stuck_loop_sessions(runner)
        assert suspended_count == 1
        assert store._entries[entry.session_key].suspended is True
        # resume_pending is still set on the entry, but suspended wins in
        # get_or_create_session so the next message still gets a new sid.
        second = store.get_or_create_session(source)
        assert second.session_id != entry.session_id
        assert second.auto_reset_reason == "suspended"


@pytest.mark.asyncio
async def test_auto_resume_sets_sentinel_before_task_execution():
    """Auto-resume must claim the session slot before the task starts.

    Regression for #45456: between ``asyncio.create_task()`` and the task's
    first await (where ``_process_message_background`` sets the real
    sentinel), an inbound message could arrive and spin up a duplicate
    AIAgent.  The fix pre-claims the slot so the inbound path sees it as
    occupied.
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="race-chat")
    pending_entry = SessionEntry(
        session_key="agent:main:telegram:dm:race-chat",
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {pending_entry.session_key: pending_entry}

    # Slow mock: hold the task open so we can inspect _running_agents
    # while it's in-flight.
    gate = asyncio.Event()

    async def _slow_handle(event):
        await gate.wait()

    adapter.handle_message = _slow_handle

    scheduled = runner._schedule_resume_pending_sessions()

    assert scheduled == 1
    # The sentinel must be set immediately — before the task starts executing.
    assert pending_entry.session_key in runner._running_agents
    assert runner._running_agents[pending_entry.session_key] is _AGENT_PENDING_SENTINEL
    assert pending_entry.session_key in runner._running_agents_ts

    # Release the task and let it complete.
    gate.set()
    await asyncio.sleep(0.05)

    # After the task completes, the sentinel should be cleaned up.
    assert pending_entry.session_key not in runner._running_agents


@pytest.mark.asyncio
async def test_auto_resume_runs_agent_exactly_once_through_full_path():
    """Full-path regression: the pre-claim must NOT make auto-resume a no-op.

    The two tests above mock ``adapter.handle_message`` outright, so they
    only prove the sentinel is set/cleaned around a stub — they never
    exercise the real dispatch chain.  This drives the production path
    end to end:

        _schedule_resume_pending_sessions
          -> _guarded_handle_message
            -> adapter.handle_message            (real)
              -> _process_message_background      (real)
                -> _handle_message                (real)

    The risk the pre-claim introduces is a *self-bounce*: the resume
    turn's own ``_handle_message`` sees the sentinel it pre-claimed at
    the early running-agent guard, queues the event into
    ``_pending_messages`` and returns ``None`` without running the
    agent.  The adapter's late-arrival drain (in
    ``_process_message_background``'s ``finally``) re-dispatches the
    queued event, and because the guard wrapper's ``finally`` releases
    the pre-claim before the spawned drain task starts, the agent runs
    exactly once.  This test locks that invariant in: the resume agent
    must run once — never zero (regression) and never twice (the bug
    the fix targets).
    """
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="full-path-chat")
    session_key = runner._session_key_for_source(source)
    pending_entry = SessionEntry(
        session_key=session_key,
        session_id="sid",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {session_key: pending_entry}

    # Wire the REAL runner pipeline that _handle_message depends on.
    from gateway.run import GatewayRunner

    runner._handle_message = GatewayRunner._handle_message.__get__(
        runner, GatewayRunner
    )
    runner._release_running_agent_state = (
        GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    )
    runner._check_slash_access = lambda *a, **k: None
    runner._begin_session_run_generation = lambda session_key: 1
    runner._is_session_run_current = lambda session_key, generation: True
    runner._invalidate_session_run_generation = lambda *a, **k: 0
    runner._claim_active_session_slot = lambda session_key, source: (object(), None)
    runner._active_session_leases = {}
    runner._busy_ack_ts = {}
    runner._post_turn_goal_continuation = AsyncMock()
    runner.session_store.get_or_create_session.return_value = None

    # Count how many times an actual agent run is started for this session.
    agent_runs: list[str] = []

    async def _fake_run(event, source, _quick_key, run_generation):
        agent_runs.append(_quick_key)
        return "RESUMED OK"

    runner._handle_message_with_agent = _fake_run

    # Route the adapter's real background pipeline at the real handler,
    # and stub the leaf send/typing calls so delivery is a no-op.
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    adapter._keep_typing = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="1")
    )
    adapter._run_processing_hook = AsyncMock()

    scheduled = runner._schedule_resume_pending_sessions()
    assert scheduled == 1
    # Pre-claim must be visible immediately.
    assert runner._running_agents.get(session_key) is _AGENT_PENDING_SENTINEL

    # Let the guarded task, the background task, and the late-arrival
    # drain task all settle.
    for _ in range(20):
        await asyncio.sleep(0.02)

    # Exactly one agent run for the resumed session — not zero (the
    # pre-claim did not swallow the resume) and not two (no duplicate).
    assert agent_runs == [session_key]
    # No leaked sentinel and no orphaned queued event.
    assert session_key not in runner._running_agents
    assert session_key not in getattr(adapter, "_pending_messages", {})


# ---------------------------------------------------------------------------
# Startup-restore inbound gate must be BOUNDED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_restore_gate_releases_when_resume_turn_outlives_timeout(
    monkeypatch,
    tmp_path,
):
    """A single slow boot-resume turn must not hold the inbound gate shut.

    While ``_startup_restore_in_progress`` is set, every inbound message is
    durably admitted. ``_finish_startup_restore`` replays persisted user work
    and waits boundedly for restart recovery tasks. Without a bound, one
    pathologically slow recovery could keep the gate closed indefinitely.
    """
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0.05")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._background_tasks = set()
    runner._async_session_store = MagicMock(_store=runner.session_store)
    runner._async_session_store.get_or_create_session = AsyncMock(
        return_value=MagicMock(session_id="bounded-gate-session")
    )
    runner._async_session_store.replay_marker_status_for_session_key = AsyncMock(
        return_value=False
    )
    runner._async_session_store.replay_marker_status = AsyncMock(return_value=False)
    runner._async_session_store.persist_replay_marker = AsyncMock(return_value=True)

    seen: list[str] = []
    never_finishes = asyncio.Event()

    async def slow_resume_turn() -> None:
        await never_finishes.wait()

    async def fake_handle_message(event: MessageEvent) -> asyncio.Task:
        async def handle() -> None:
            seen.append(f"inbound:{event.text}")
            event._hermes_handler_succeeded = True

        return asyncio.create_task(handle())

    adapter.handle_message = fake_handle_message

    slow_task = asyncio.create_task(slow_resume_turn())
    runner._startup_restore_tasks = [slow_task]

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="bounded-gate-1",
    )
    decision = await runner._handle_message(inbound)
    assert decision.durable_accepted is True

    # The gate must release on the bound even though the resume turn is
    # still running.
    await asyncio.wait_for(runner._finish_startup_restore(), timeout=5)

    assert seen == ["inbound:hello"], (
        "startup-restore gate never released: queued inbound was not drained "
        "while a slow boot-resume turn was still running"
    )
    assert runner._startup_restore_in_progress is False
    # The slow turn is NOT cancelled — it finishes in the background.
    assert not slow_task.done()

    never_finishes.set()
    await slow_task


@pytest.mark.asyncio
async def test_startup_restore_gate_replays_durable_work_before_prompt_recovery(
    monkeypatch,
    tmp_path,
):
    """The bound must not truncate a normal-speed resume turn.

    Persisted user work is replayed before the generic recovery task, while the
    gate itself remains closed until a prompt recovery task finishes.
    """
    monkeypatch.delenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._background_tasks = set()
    runner._async_session_store = MagicMock(_store=runner.session_store)
    runner._async_session_store.get_or_create_session = AsyncMock(
        return_value=MagicMock(session_id="prompt-gate-session")
    )
    runner._async_session_store.replay_marker_status_for_session_key = AsyncMock(
        return_value=False
    )
    runner._async_session_store.replay_marker_status = AsyncMock(return_value=False)
    runner._async_session_store.persist_replay_marker = AsyncMock(return_value=True)

    seen: list[str] = []
    resume_done = asyncio.Event()

    async def resume_turn() -> None:
        await resume_done.wait()
        seen.append("resume-finished")

    async def fake_handle_message(event: MessageEvent) -> asyncio.Task:
        async def handle() -> None:
            seen.append(f"inbound:{event.text}")
            event._hermes_handler_succeeded = True

        return asyncio.create_task(handle())

    adapter.handle_message = fake_handle_message

    runner._startup_restore_tasks = [asyncio.create_task(resume_turn())]

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="prompt-gate-1",
    )
    decision = await runner._handle_message(inbound)
    assert decision.durable_accepted is True

    finish_task = asyncio.create_task(runner._finish_startup_restore())
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == ["inbound:hello"]
    assert not finish_task.done(), "gate opened before the recovery task finished"

    resume_done.set()
    await finish_task
    assert seen == ["inbound:hello", "resume-finished"]


@pytest.mark.asyncio
async def test_startup_restore_drain_timeout_zero_restores_unbounded_wait(
    monkeypatch,
    tmp_path,
):
    """A non-positive bound opts back into the historical wait-forever gate."""
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "0")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._background_tasks = set()
    runner._async_session_store = MagicMock(_store=runner.session_store)
    runner._async_session_store.get_or_create_session = AsyncMock(
        return_value=MagicMock(session_id="unbounded-gate-session")
    )
    runner._async_session_store.replay_marker_status_for_session_key = AsyncMock(
        return_value=False
    )
    runner._async_session_store.replay_marker_status = AsyncMock(return_value=False)
    runner._async_session_store.persist_replay_marker = AsyncMock(return_value=True)

    seen: list[str] = []
    resume_done = asyncio.Event()

    async def resume_turn() -> None:
        await resume_done.wait()

    async def fake_handle_message(event: MessageEvent) -> asyncio.Task:
        async def handle() -> None:
            seen.append(f"inbound:{event.text}")
            event._hermes_handler_succeeded = True

        return asyncio.create_task(handle())

    adapter.handle_message = fake_handle_message
    runner._startup_restore_tasks = [asyncio.create_task(resume_turn())]

    inbound = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="restore-chat"),
        message_id="unbounded-gate-1",
    )
    decision = await runner._handle_message(inbound)
    assert decision.durable_accepted is True

    finish_task = asyncio.create_task(runner._finish_startup_restore())
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == ["inbound:hello"]
    assert not finish_task.done(), "unbounded gate released early"

    resume_done.set()
    await finish_task
    assert seen == ["inbound:hello"]


def test_startup_restore_drain_timeout_reads_config_bridged_env(monkeypatch):
    """The bound is a config.yaml knob bridged to an internal env var."""
    from gateway.run import (
        _STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT,
        _startup_restore_drain_timeout_secs,
    )

    monkeypatch.delenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", raising=False)
    assert (
        _startup_restore_drain_timeout_secs()
        == _STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT
    )

    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "12.5")
    assert _startup_restore_drain_timeout_secs() == 12.5

    # A malformed value must fall back to the default, never raise.
    monkeypatch.setenv("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", "not-a-number")
    assert (
        _startup_restore_drain_timeout_secs()
        == _STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT
    )


def test_startup_restore_drain_timeout_is_a_documented_config_key():
    """agent.gateway_startup_restore_drain_timeout ships in DEFAULT_CONFIG."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert (
        "gateway_startup_restore_drain_timeout" in DEFAULT_CONFIG["agent"]
    ), "the bound must be a config.yaml knob, not an undocumented env var"
