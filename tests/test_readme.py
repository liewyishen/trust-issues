"""
The tests badge is an invariant, not a claim.

README.md states the test count in four places: the shields.io badge (the one
that is a static string -- measured by nothing and pinned by nothing, until
this file) and three prose mentions. The number has gone stale four times, most
recently INSIDE a single working session, when serving/ took the suite from 153
to 206 and every one of the four had to be hand-edited.

This makes the number self-checking. It reads the count pytest's collector
actually produced this run -- len(session.items), NOT a literal -- and asserts
every count string in the README equals it.

Circularity, handled: this test is itself a collected item, so it counts toward
the number it checks. That is exactly why the count must be read from the live
collection and never hardcoded -- add or remove any test and both sides move
together. The README literal is set from `pytest --collect-only -q` AFTER this
file exists, never guessed before.

Partial runs, handled: `pytest -k`, `-m`, `--lf/--ff`, `--deselect`, or an
explicit file / nodeid collect a deliberate subset, so len(session.items) is
not the full suite and there is nothing to compare against. Those SKIP rather
than fail -- a narrowed run must never turn a green badge red.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

# One pattern per shape the count takes in README.md; each captures the integer.
#   badge  ->  tests-206%20passing        (%20 is the URL-encoded space)
#   table  ->  | **pytest** | 206 tests across the modeling layer |
#   prose  ->  ... (206 passing)          (setup block, project-structure block)
# The badge's "%20" means "(\d+) passing" (a literal space) does NOT also match
# the badge line, so every line contributes exactly one count.
_COUNT_PATTERNS = (
    r"tests-(\d+)%20passing",
    r"(\d+) tests\b",
    r"(\d+) passing",
)


def _readme_counts() -> list[tuple[int, str]]:
    """Every test-count integer in README.md, tagged with the line it sits on."""
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(README.read_text().splitlines(), start=1):
        for pattern in _COUNT_PATTERNS:
            for match in re.finditer(pattern, line):
                found.append((int(match.group(1)), f"README.md:{lineno}: {line.strip()}"))
    return found


def _run_is_narrowed(config) -> bool:
    """True when this run collected a deliberate subset rather than the suite."""
    opt = config.option
    if getattr(opt, "keyword", "") or getattr(opt, "markexpr", ""):
        return True                                    # -k / -m
    if getattr(opt, "last_failed", False) or getattr(opt, "failed_first", False):
        return True                                    # --lf / --ff
    if getattr(opt, "deselect", None):
        return True                                    # --deselect
    for arg in config.invocation_params.args:          # explicit file / nodeid
        if arg.startswith("-"):
            continue
        if "::" in arg or arg.split("::", 1)[0].endswith(".py"):
            return True
    return False


def test_readme_test_count_matches_the_live_collection(request):
    collected = len(request.session.items)

    if _run_is_narrowed(request.config):
        pytest.skip(
            f"narrowed run collected {collected} item(s); the README count is "
            "verifiable only against the full suite"
        )

    counts = _readme_counts()
    assert counts, "no test-count string found in README.md -- did the badge format change?"

    stale = [where for n, where in counts if n != collected]
    assert not stale, (
        f"README states a test count that no longer matches the suite "
        f"(collected {collected}). Update these and the badge together:\n  "
        + "\n  ".join(stale)
    )
