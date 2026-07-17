"""
The suite opens no network connection -- and now says so.

`serving/` performs no outbound I/O. That is not a nice-to-have; it is the
premise the rest of the serving story rests on. `MockBureau` (`serving/bureau.py`)
is deterministic *because* it never calls a vendor -- its own docstring says "A
CreditBureau that never calls a real vendor" -- and determinism is what lets
test_same_applicant_id_scores_reproducibly_through_http and
test_the_same_knob_setting_returns_the_same_numbers (`tests/test_serving.py`)
assert byte-identical responses at all. The frontend advertises the property to
the user's face at `frontend/src/App.tsx:127`: "The bureau is a deterministic
mock (MockBureau) -- it never calls a real vendor."

Until this file, nothing checked it.

Why it was unguarded, and why that reason was wrong
---------------------------------------------------
The reason was: the suite makes no outbound calls, so nothing is wrong. The
check that "established" it was a grep over `tests/` for socket / requests /
urlopen / timeout, which came back empty.

That grep was worthless, and this repo already knew why. `docs/data-decisions.md`
("Why 'serving never reaches matplotlib' was false, and why grep could not catch
it") ends: "Grep proves what this repo's own code says. Only the interpreter
knows what it actually imports." The same sentence holds one layer out -- only
the interpreter knows what it actually *calls*. No test names a socket. That
proves nothing about whether one is opened, because the opener is never
first-party: it arrives transitively, from site-packages, exactly the way
`matplotlib` arrived through lightgbm's compat module.

And it was already false. Recording every connect/bind/getaddrinfo across a full
302-test run found the suite reaching `config.mlflow-telemetry.io:443` and
`api.mlflow-telemetry.io:443` -- three connects, three real IPs. MLflow's
telemetry client phones home on a background thread (`mlflow/telemetry/client.py`,
`_fetch`).

`tests/test_train.py` is the only file that triggers it, and the reason is worth
one sentence because it is the same trap again: importing mlflow is NOT enough.
`tests/test_training_flow.py` imports it too, transitively (`training_flow.py` ->
`train_and_save()` -> `src/train.py`, which imports mlflow at module scope), and
sends nothing. The telemetry fires on USE -- `mlflow.create_experiment()` /
`train_and_save()`, which only `tests/test_train.py` calls. So even the import
graph, the tool this repo reaches for first, would have cleared this. Only the
socket knew.

So this guard was committed RED (e8bc315), on purpose, the same way
tests/test_docs_fairness.py was: a red test committed deliberately is the record
that the bug was real. The telemetry is silenced in the commit AFTER it, not the
same one -- the reason it is silenced (MLflow chose to phone home; this repo
chose not to) is a different fact from the reason the property is guarded, and
one commit holding both would leave neither on the record. The ordering is the
point: the guard existed before the fix, so the red is a record and not a
rehearsal.

The general shape, which outlives the MLflow specifics: a property nothing
asserts is not a property, it is a coincidence -- and this one had already
stopped being true without anything noticing. That is the same finding
`docs/data-decisions.md`'s damage-inventory entry records about `.gitignore:46`:
a site that is correct today and has no rope on it is not safe, it is unwatched.
Here the site was not even correct.

Where the line is drawn: connect, not bind
-------------------------------------------
`bind()` is permitted. The only bind this suite performs is urllib3's IPv6
capability probe (`urllib3/util/connection.py`, `_has_ipv6`), which binds
`('::1', 0)` at import time and never connects -- a question asked of the kernel,
not of the network.

Permitting *bind-without-connect* is deliberately narrower than permitting
*loopback*. A loopback allowance would wave through a future test that stands up
a real uvicorn and talks to it over 127.0.0.1 -- which is precisely the kind of
test whose network behavior should have to be declared. So a connect to
127.0.0.1 is blocked like any other. If that day comes, it declares itself with
the marker below.

What this does NOT guard, said out loud
----------------------------------------
`getaddrinfo` is left alone. A DNS lookup does leave the machine, so the property
enforced here is strictly narrower than "no packet leaves": it is "no connection
is opened". Two reasons, neither of them "it was hard". First, resolution is not
contact with the host -- an unconnected lookup accomplishes nothing on its own.
Second, blocking it buys no coverage: in the recorded run every telemetry
`getaddrinfo` was followed by a `connect` to what it resolved, so the connect
catch fires on the same events, at the act rather than at the lookup. If a future
component ever exfiltrates through DNS alone, this guard will not see it, and
that sentence is here so nobody has to rediscover it.

Attribution is exact in-thread and approximate off it
------------------------------------------------------
Violations are recorded AND raised. The raise is for the honest case: a test that
calls out directly fails at the call, in its own stack, with its own nodeid.

The recording is for the dishonest one. MLflow's telemetry runs on a
`threading.Thread`, and an exception raised there does not fail anything -- it
dies in the thread. A guard that only raised would silently break that thread and
go green, which is a guard that passes without biting. So every violation is also
logged, and pytest_sessionfinish fails the run on a non-empty log regardless of
which thread it came from.

The cost of that, stated rather than hidden: a background thread's call is
attributed to whichever test happened to be running when the scheduler got to it.
That attribution is a hint, not evidence -- the thread name in the report is the
part to trust. The pass/fail verdict does not depend on it.

Session-scoped, not an autouse fixture
---------------------------------------
Installed in pytest_configure, for the same reason. An autouse fixture exists
only during the test phase; the background thread does not care, and module-level
imports (where urllib3's probe runs, during collection) happen before any fixture
is alive.
"""

from __future__ import annotations

import os
import socket
import threading
import traceback
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# The one connection the guard below found. Silenced here, at module scope,
# because that runs before pytest imports any test module -- and therefore
# before tests/test_train.py imports mlflow.
#
# Whose choice this is, stated plainly: MLflow chose to phone home by default.
# This repo chooses not to. The mechanism is theirs -- both switches are
# MLflow's own (mlflow/telemetry/utils.py checks MLFLOW_DISABLE_TELEMETRY and
# DO_NOT_TRACK). Nothing is being patched, monkeyed or worked around; a
# published opt-out is being taken.
#
# setdefault, NOT assignment. An operator who has already set either variable
# -- to opt IN, or to opt out globally via DO_NOT_TRACK -- keeps their setting.
# This states a default; it does not overrule a decision someone else made. An
# unconditional `os.environ[...] = "true"` would silently discard the DO_NOT_TRACK
# convention this repo would be honouring in the same breath as breaking.
#
# Not pytest-env: a dependency for one environment variable, and this repo has
# already declined a heavier version of that same trade. Not CI-only: `uv run
# pytest` on a laptop would still send the packets, which is not a fix, it is a
# place to keep the problem.
# ---------------------------------------------------------------------------
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

ROOT = Path(__file__).resolve().parents[1]

MARKER = "allow_network"


class NetworkAccessBlocked(RuntimeError):
    """Raised in-thread at the point a test opened a connection it never declared."""


# (nodeid, kind, address, repo frame, thread name). Written from arbitrary
# threads, so appends only -- list.append is atomic under the GIL.
_VIOLATIONS: list[tuple[str, str, str, str, str]] = []

_CURRENT: pytest.Item | None = None

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: this test opens a network connection, on purpose, and says why. "
        "Nothing uses this today -- it exists so the first thing that needs the "
        "network has to declare itself in a diff, where a reviewer can see it.",
    )
    _install()


def pytest_unconfigure(config: pytest.Config) -> None:
    _uninstall()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    global _CURRENT
    _CURRENT = item
    try:
        yield
    finally:
        _CURRENT = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _uninstall()
    if not _VIOLATIONS:
        return

    # A suite-level property fails the suite. It is not attributable to one test
    # -- see the module docstring on background threads -- so it is not reported
    # as one.
    session.exitstatus = pytest.ExitCode.TESTS_FAILED

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:                                  # -p no:terminal
        return
    reporter.write_sep("=", "NETWORK ACCESS BLOCKED", red=True, bold=True)
    reporter.write_line(
        f"{len(_VIOLATIONS)} outbound connection(s) attempted. This suite claims "
        f"to open none; `MockBureau` is deterministic because it calls no vendor "
        f"(frontend/src/App.tsx:127 says so to the user)."
    )
    for nodeid, kind, addr, where, thread in _VIOLATIONS:
        reporter.write_line("")
        reporter.write_line(f"  {kind} -> {addr}")
        reporter.write_line(f"    thread : {thread}")
        reporter.write_line(f"    from   : {where}")
        reporter.write_line(f"    during : {nodeid}")
    reporter.write_line("")
    reporter.write_line(
        f"If a connection is legitimate, mark the test @pytest.mark.{MARKER} and "
        f"write the reason in the test. Do not widen this guard."
    )


# ---------------------------------------------------------------------------
# The patch itself.
#
# Patched on the CLASS, not by rebinding `socket.socket`, so it holds regardless
# of how a caller got at the type -- `import socket; socket.socket(...)` and
# `from socket import socket` both land on the same method object. urllib3 does
# the former; nothing guarantees the next library will.
# ---------------------------------------------------------------------------
def _install() -> None:
    socket.socket.connect = _guarded_connect                      # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex                # type: ignore[method-assign]
    socket.create_connection = _guarded_create_connection         # type: ignore[assignment]


def _uninstall() -> None:
    socket.socket.connect = _REAL_CONNECT                         # type: ignore[method-assign]
    socket.socket.connect_ex = _REAL_CONNECT_EX                   # type: ignore[method-assign]
    socket.create_connection = _REAL_CREATE_CONNECTION            # type: ignore[assignment]


def _guarded_connect(self, address):
    _check("socket.connect", address)
    return _REAL_CONNECT(self, address)


def _guarded_connect_ex(self, address):
    _check("socket.connect_ex", address)
    return _REAL_CONNECT_EX(self, address)


def _guarded_create_connection(address, *args, **kwargs):
    _check("socket.create_connection", address)
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)


def _check(kind: str, address) -> None:
    item = _CURRENT
    if item is not None and item.get_closest_marker(MARKER) is not None:
        return

    nodeid = item.nodeid if item is not None else (
        "<no test running: collection, an import, or a background thread>"
    )
    _VIOLATIONS.append(
        (nodeid, kind, repr(address), _first_repo_frame(), threading.current_thread().name)
    )
    raise NetworkAccessBlocked(
        f"{kind} -> {address!r}\n"
        f"This suite opens no network connections. `MockBureau` is deterministic "
        f"because it never calls a vendor, and frontend/src/App.tsx:127 tells the "
        f"user so.\n"
        f"If this call is legitimate, mark the test @pytest.mark.{MARKER} and write "
        f"the reason in the test itself -- so it lands in a diff instead of in a "
        f"habit."
    )


def _first_repo_frame() -> str:
    """The innermost frame in this repo -- i.e. what asked for the call, not the
    library that made it.

    This file is excluded, and that is not cosmetic: _guarded_connect is itself a
    first-party frame, so without the exclusion every violation would report
    tests/conftest.py as its origin and the field would be worthless. When the
    answer comes back empty, that IS the finding -- nothing in this repo is on
    the stack, so the call came from a dependency that decided to make it on its
    own. Which is exactly what MLflow's telemetry thread does.
    """
    for frame in reversed(traceback.extract_stack()):
        path = Path(frame.filename)
        if path == Path(__file__):
            continue
        if path.is_relative_to(ROOT) and ".venv" not in path.parts:
            return f"{path.relative_to(ROOT)}:{frame.lineno}"
    return "(no first-party frame -- entirely inside site-packages)"
