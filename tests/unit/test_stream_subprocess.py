"""
tests/unit/test_stream_subprocess.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Focused tests for the ``stream_subprocess`` async generator defined in
``app.main``.

Goals
-----
* Blank / whitespace-only subprocess output lines must **not** appear as
  standalone ``data: `` SSE events.
* Normal (non-empty) output must still be forwarded as ``data: <line>\\n\\n``.
* Synthetic markers ``[HINT]``, ``[ERROR]``, and ``[DONE]`` must all be
  emitted as expected.
* The SSE format (``data: `` prefix + double-newline terminator) must remain
  intact for every event that is emitted.

No extra async test plugins are required — each test drives the generator via
``asyncio.run()``.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

from app.main import stream_subprocess


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_stdout(lines: list[str]):
    """Return an async-iterable that yields encoded lines, mimicking
    ``asyncio.StreamReader`` behaviour as seen inside ``stream_subprocess``."""

    async def _aiter():
        for line in lines:
            yield line.encode()

    reader = MagicMock()
    reader.__aiter__ = lambda self: _aiter()
    return reader


def _make_proc(lines: list[str], returncode: int = 0):
    """Return a mock subprocess whose stdout yields *lines* and whose
    ``wait()`` coroutine returns immediately with *returncode*."""
    proc = MagicMock()
    proc.stdout = _make_stdout(lines)
    proc.returncode = returncode

    async def _wait():
        proc.returncode = returncode

    proc.wait = _wait
    return proc


def collect(gen: AsyncIterator[str]) -> list[str]:
    """Drain an async generator synchronously and return all emitted chunks."""

    async def _drain():
        events: list[str] = []
        async for chunk in gen:
            events.append(chunk)
        return events

    return asyncio.run(_drain())


# ── Blank / whitespace-only line filtering ─────────────────────────────────────

class TestBlankLineFiltering:
    """Empty or whitespace-only subprocess lines must not become SSE events."""

    def test_empty_line_not_emitted(self):
        """A bare newline from the subprocess must produce no SSE event."""
        proc = _make_proc(["\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["echo"]))
        # Only the final [DONE] marker should appear; no "data: \n\n"
        assert events == ["data: [DONE]\n\n"]

    def test_whitespace_only_line_not_emitted(self):
        """A line containing only spaces/tabs must produce no SSE event."""
        proc = _make_proc(["   \t  \n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["echo"]))
        assert events == ["data: [DONE]\n\n"]

    def test_multiple_blank_lines_not_emitted(self):
        """Several consecutive blank lines must all be suppressed."""
        proc = _make_proc(["\n", "\n", "  \n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["echo"]))
        assert events == ["data: [DONE]\n\n"]

    def test_blank_lines_interspersed_with_real_output(self):
        """Blank lines between meaningful lines are suppressed; content lines keep
        their SSE event."""
        proc = _make_proc(["hello\n", "\n", "world\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["echo"]))
        assert "data: hello\n\n" in events
        assert "data: world\n\n" in events
        # No pure blank data: events
        assert "data: \n\n" not in events


# ── Normal output streaming ────────────────────────────────────────────────────

class TestNormalOutputStreaming:
    """Non-empty subprocess output must still be forwarded as SSE events."""

    def test_single_line_emitted(self):
        proc = _make_proc(["Building...\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["make"]))
        assert "data: Building...\n\n" in events

    def test_multiple_lines_all_emitted(self):
        lines = ["Step 1\n", "Step 2\n", "Step 3\n"]
        proc = _make_proc(lines, returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["make"]))
        for raw in lines:
            expected = f"data: {raw.rstrip()}\n\n"
            assert expected in events

    def test_sse_prefix_present_on_all_content_events(self):
        """Every content event must start with 'data: '."""
        proc = _make_proc(["line A\n", "line B\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["make"]))
        for event in events:
            assert event.startswith("data: "), f"Event missing 'data: ' prefix: {event!r}"

    def test_sse_double_newline_terminator(self):
        """Every SSE event must end with '\\n\\n'."""
        proc = _make_proc(["progress\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["make"]))
        for event in events:
            assert event.endswith("\n\n"), f"Event missing double-newline: {event!r}"


# ── Synthetic marker preservation ─────────────────────────────────────────────

class TestSyntheticMarkers:
    """[DONE], [ERROR], and [HINT] markers must all survive as-is."""

    def test_done_emitted_on_success(self):
        proc = _make_proc(["ok\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["cmd"]))
        assert "data: [DONE]\n\n" in events

    def test_error_emitted_on_nonzero_exit(self):
        proc = _make_proc(["bad\n"], returncode=1)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["cmd"]))
        error_events = [e for e in events if "[ERROR]" in e]
        assert error_events, "Expected at least one [ERROR] event"
        assert any("1" in e for e in error_events), "Exit code should appear in [ERROR] message"

    def test_done_not_emitted_on_failure(self):
        """On non-zero exit [DONE] must not appear — only [ERROR]."""
        proc = _make_proc(["bad\n"], returncode=2)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = collect(stream_subprocess(["cmd"]))
        assert "data: [DONE]\n\n" not in events

    def test_hint_emitted_for_known_failure(self):
        """If detect_known_failure returns a hint, a [HINT] event must follow."""
        proc = _make_proc(["cutlass error here\n"], returncode=1)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with patch("app.main.detect_known_failure", return_value="Try --cutlass=n"):
                events = collect(stream_subprocess(["cmd"]))
        hint_events = [e for e in events if "[HINT]" in e]
        assert hint_events, "Expected a [HINT] event"
        assert "Try --cutlass=n" in hint_events[0]

    def test_hint_emitted_at_most_once(self):
        """Even with multiple matching lines, only one [HINT] is emitted."""
        lines = ["cutlass error\n", "cutlass error again\n"]
        proc = _make_proc(lines, returncode=1)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with patch("app.main.detect_known_failure", return_value="Try --cutlass=n"):
                events = collect(stream_subprocess(["cmd"]))
        hint_count = sum(1 for e in events if "[HINT]" in e)
        assert hint_count == 1

    def test_no_hint_when_no_known_failure(self):
        """With no recognised failure pattern, no [HINT] event must appear."""
        proc = _make_proc(["normal output\n"], returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with patch("app.main.detect_known_failure", return_value=None):
                events = collect(stream_subprocess(["cmd"]))
        assert not any("[HINT]" in e for e in events)
