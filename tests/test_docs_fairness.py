"""
The fairness numbers are an invariant, not a claim.

`models/fairness_audit.json` is the machine-readable source of truth for the
addr_state ablation -- this repo's loudest fairness finding. Eight files quote
it in prose. Until this file, NOTHING pinned any of them to the artifact.

The drift this exists to stop, in full: `docs/design.md` reported Mississippi's
Layer-3 EO ratio as a RANGE -- `~0.734-0.745` with the state label, `~0.988-0.990`
without. That range is a pre-authoritative spread across two different runs of
the same quantity (the notebook's 0.734/0.990 and the pipeline's 0.745/0.988).
It survived 8 days and three doc passes.

It survived because commit 6e759b8 checked the wrong thing. It hand-corrected
the AUC to run cca4c361's authoritative value, observed that 0.744823 fell
INSIDE the 0.734-0.745 spread, and concluded the spread was still true. One
commit, two epochs: the AUC got a point estimate, the EO kept a spread. But a
spread that happens to contain the point estimate is not a confidence interval.
Both containments here (0.734 in [0.7155, 0.7741], 0.990 in [0.9621, 1.0124])
are byproducts of the bootstrap being wide enough. Reporting the spread as if it
were an interval launders a stale run into a statistical claim -- worse than a
wrong number, because it reads as evidence. So this file rejects the range FORM
outright (see _SPREAD), not merely stale values: a doc that quoted
`~0.734-0.745` would otherwise satisfy a bare "does 0.745 appear?" check and go
green while saying something false.

What this is NOT
----------------
This is a STRING-EQUALITY test, not a statistical claim. It asks only "does the
doc say the same number the artifact holds?" -- never "is that number good
evidence?". `tests/test_serving.py`'s
test_the_ablation_is_the_evidence_the_readme_claims_it_is already makes the
interval claim against the artifact, and deliberately REFUSES to pin point
estimates, because pinning them there would re-commit the error the CIs were
added to fix. That refusal is correct for serving, and it is precisely why this
gap existed: "is 0.744 good evidence?" and "does the doc match the artifact?"
are two different questions. That file answers the first. This one answers only
the second, and the two must not be merged.

The exemption rule -- which files are ALLOWED to go stale
---------------------------------------------------------
Stated here because, until now, it existed only inside one commit message. A
rule that a guard enforces but no document states is itself a say-equals-do
hole.

    Files that RECORD HISTORY are exempt. Files that DESCRIBE THE CURRENT
    SYSTEM are not.

Exempt, on purpose, and never to be added to _SITES:

- `docs/data-decisions.md` -- append-only. Its whole discipline is that
  superseded numbers stay standing as the record of what was believed when.
  6e759b8's own message spells this out about line 127: "Overwriting it in place
  would falsify that history and break this file's append-only rule." A guard
  that reddens that file is arguing with the file's design. Corrections there
  are APPENDED as new entries, never edited in.
- `notebooks/analysis.ipynb` -- a with-state historical artifact. Its 0.734 is
  the true output of that run, taken with INCLUDE_ADDR_STATE = True. It is
  correct in context; "fixing" it would make the notebook lie about its own run.

Everything else quoting these numbers describes the system as it is today, and
belongs in _SITES.

Partial runs, handled: nothing here depends on the collection, so unlike
test_readme.py there is no narrowing skip. The one skip is the artifact itself
being absent -- pinning to a source of truth that is not on disk is not a
failure, it is nothing to check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "fairness_audit.json"

# The artifact is committed, but it is a build product: a fresh clone that has
# not run scripts/audit_fairness.py, or a slim image that ships no models/, has
# nothing to pin against. Skip, don't fail -- same discipline as test_readme.py
# refusing to redden a narrowed run.
NEEDS_ARTIFACT = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.relative_to(ROOT)} absent -- no source of truth to pin against",
)

# A spread: two EO-shaped floats joined by a dash. This is the FORM the drift
# took, and it is never legitimate at any site below -- every site reports one
# run, and one run yields one number per quantity.
#
# Deliberately does not fire on "0.6690 -> 0.6654" or "0.745 -> 0.988": those
# are arrows (a transition between two DIFFERENT quantities), not ranges over
# one. The `>` after the dash is not whitespace, so the second float never
# matches. Nor on a leading sign ("cost: -0.0036") -- there is no float before
# the dash.
_SPREAD = re.compile(r"0\.\d+\s*[-–—]\s*0\.\d+")

_MS = "layer3.states[MS]"

# The two shapes a site's claim takes. The asymmetry between them is
# INTENTIONAL -- do not "fix" it by giving every site the intervals:
#
#   Prose docs (design.md, README.md, frontend/README.md) carry POINT + CI,
#   because their subject is the finding, and the finding IS the interval: two
#   bare point estimates cannot tell the shift apart from sampling noise, which
#   is the whole reason audit_layer3_ablation() returns its Test frames.
#
#   Docstrings and code comments (src/fairness.py, src/features.py,
#   tests/test_fairness.py) carry the POINT PAIR ONLY, because a docstring's
#   contract is WHAT THE FUNCTION PRINTS -- and run_fairness_audit() prints
#   points. Layer 3's verdict column is decided on point estimates; the CIs come
#   from feeding its frames back through audit_layer1(), which the function does
#   not do and the caller may decline to. Putting an interval in those
#   docstrings would describe output the function never produces: the same
#   say-equals-do violation as the range, just pointing the other way.
_POINTS = (
    f"{_MS}.eo_ratio_with_state",
    f"{_MS}.eo_ratio_no_state",
)
_POINTS_AND_CIS = (
    f"{_MS}.eo_ratio_with_state",
    f"{_MS}.ci_low_with_state",
    f"{_MS}.ci_high_with_state",
    f"{_MS}.eo_ratio_no_state",
    f"{_MS}.ci_low_no_state",
    f"{_MS}.ci_high_no_state",
)


class Site(NamedTuple):
    """One prose claim about the MS ablation, and what it must say.

    region : regex bracketing the claim, matched against the whole file. Must
        match EXACTLY once -- zero matches or several means the prose was
        restructured and this registry entry, not the doc, is what went stale.
        Anchored on surrounding words rather than on the numbers themselves, so
        a stale value still matches and reports a value mismatch instead of a
        useless "pattern not found".
    keys : artifact key paths whose formatted values must all appear inside the
        region.
    precision : decimals the site writes, and therefore the decimals the
        artifact value is rounded to before comparing.
    """

    name: str
    path: str
    region: str
    keys: tuple[str, ...]
    precision: int


# This table deliberately EXCLUDES this file. The module docstring above quotes
# 0.734/0.990 and the ~0.734-0.745 spread verbatim, because explaining why the
# guard exists requires naming what it caught -- the exemption rule's "records
# history" clause, applied to the file that states the rule. Registering it
# would make _SPREAD fire on that docstring and redden the guard forever. A
# convention, not an assertion; written down here so it stops being merely a
# convention, and can be applied against whoever tries.
#
# A registry, not a grep. Free-grepping floats across prose would hit uv.lock's
# upload timestamps, the ~0.4s bureau latency, Brier scores and the AUC pair --
# so every site is named, bracketed, and says which artifact keys it owes.
_SITES = (
    Site(
        name="design.md/drop-addr_state",
        path="docs/design.md",
        region=r"\*\*Drop `addr_state`\.\*\*(?s:.+?)(?=\n\n|\Z)",
        keys=_POINTS_AND_CIS,
        precision=3,
    ),
    Site(
        name="README.md/what-the-paranoia-caught",
        path="README.md",
        region=r"\*\*`addr_state` was a digital-redlining shortcut(?s:.+?)(?=\n\n|\Z)",
        keys=_POINTS_AND_CIS,
        precision=3,
    ),
    Site(
        # The capability table's one-line summary of the same finding. It states
        # the recovery as a point pair and says "with non-overlapping 95% CIs"
        # in words rather than numbers, so only the points are owed here.
        name="README.md/capability-table-row-6",
        path="README.md",
        region=r"\| \*\*6\. Explainability \+ fairness\*\*.+",
        keys=_POINTS,
        precision=3,
    ),
    Site(
        name="frontend/README.md/layer-3-counterfactual",
        path="frontend/README.md",
        region=r"\| \*\*3 · The counterfactual\*\*.+",
        keys=_POINTS_AND_CIS,
        precision=3,
    ),
    Site(
        name="src/fairness.py/module-docstring",
        path="src/fairness.py",
        region=(
            r"On the real dataset, this reproduces the notebook's headline finding:"
            r"(?s:.+?)See run_fairness_audit\(\)'s docstring"
        ),
        keys=_POINTS,
        precision=3,
    ),
    Site(
        name="src/fairness.py/run_fairness_audit-docstring",
        path="src/fairness.py",
        region=r"Layer 3 \(threshold=0\.22\) reproduces approximately:(?s:.+?)i\.e\. Mississippi's",
        keys=_POINTS,
        precision=3,
    ),
    Site(
        name="src/features.py/INCLUDE_ADDR_STATE-comment",
        path="src/features.py",
        region=(
            r"geographic-proxy risk / digital-redlining shortcut:"
            r"(?s:.+?)That audit's conclusion"
        ),
        keys=_POINTS,
        precision=3,
    ),
    Site(
        name="tests/test_fairness.py/module-docstring",
        path="tests/test_fairness.py",
        region=r"see fairness\.py's module docstring for(?s:.+?)CSV\):",
        keys=_POINTS,
        precision=3,
    ),
    Site(
        name="tests/test_fairness.py/layer-3-section-comment",
        path="tests/test_fairness.py",
        region=r"# moves that state's EO ratio toward parity(?s:.+?)\)\.\n",
        keys=_POINTS,
        precision=3,
    ),
    Site(
        # The `!models/fairness_audit.json` negation's justification: the audit's
        # output is source-controlled BECAUSE this finding would otherwise exist
        # only on whichever machine last ran it. Quotes both ratios at 2 dp.
        #
        # Missed by the original blast-radius sweep, which grepped
        # 0.734|0.745|0.988|0.990 -- i.e. searched for the STALE values, and so
        # could only find sites already wrong. This one was already right, and
        # therefore invisible. A registry's job is not to catch what has
        # drifted; it is to bind everything that could.
        name=".gitignore/fairness_audit-negation-rationale",
        path=".gitignore",
        region=(
            r"evidence behind this repo's loudest fairness claim"
            r"(?s:.+?)nobody's authority"
        ),
        keys=_POINTS,
        precision=2,
    ),
    Site(
        # The 409 docstring's worked example of a stale artifact. Quotes the
        # with-state point at 4 dp, the precision the JSON's own consumers use.
        name="frontend/src/lib/api.ts/stale-409-docstring",
        path="frontend/src/lib/api.ts",
        region=r"cheerfully reports Mississippi at .+?, about a model that no longer exists",
        keys=(f"{_MS}.eo_ratio_with_state",),
        precision=4,
    ),
)


def _resolve(audit: dict, path: str) -> float:
    """Resolve a dotted artifact key path.

    `states[MS]` selects the list element whose "state" field is "MS" -- the
    artifact stores per-state rows as a list, not a mapping, so a bare index
    would silently follow the list's order instead of the state's identity.
    """
    node: object = audit
    for part in path.split("."):
        selector = re.fullmatch(r"(\w+)\[(\w+)\]", part)
        if selector:
            field, want = selector.groups()
            rows = node[field]                                  # type: ignore[index]
            hits = [r for r in rows if r.get("state") == want]
            assert len(hits) == 1, (
                f"artifact key path '{path}': expected exactly one row with "
                f"state={want!r} under '{field}', found {len(hits)}"
            )
            node = hits[0]
        else:
            assert part in node, (                              # type: ignore[operator]
                f"artifact key path '{path}': '{part}' is not a key -- the "
                f"audit's schema changed, so this registry is what is stale"
            )
            node = node[part]                                   # type: ignore[index]
    assert isinstance(node, (int, float)), (
        f"artifact key path '{path}' resolved to {type(node).__name__}, not a number"
    )
    return float(node)


@NEEDS_ARTIFACT
@pytest.mark.parametrize("site", _SITES, ids=[s.name for s in _SITES])
def test_doc_fairness_numbers_match_the_artifact(site: Site):
    audit = json.loads(ARTIFACT.read_text())
    text = (ROOT / site.path).read_text()

    matches = re.findall(site.region, text)
    assert len(matches) == 1, (
        f"{site.path}: the claim region for '{site.name}' matched {len(matches)} "
        f"times, expected exactly 1. The prose was restructured -- fix this "
        f"registry entry's `region`, not the doc."
    )
    region = matches[0]

    spread = _SPREAD.search(region)
    assert not spread, (
        f"{site.path}: '{site.name}' reports a RANGE, {spread.group(0)!r}.\n"
        f"A spread across runs is not a confidence interval, and a spread that "
        f"happens to contain the point estimate is not evidence of anything. "
        f"Quote the single authoritative value from "
        f"{ARTIFACT.relative_to(ROOT)}; if an interval is wanted, quote the "
        f"artifact's own bootstrap CI."
    )

    stale = []
    for key in site.keys:
        want = f"{_resolve(audit, key):.{site.precision}f}"
        if want not in region:
            stale.append(f"{key} -> {want}")

    assert not stale, (
        f"{site.path}: '{site.name}' does not carry the value(s) "
        f"{ARTIFACT.relative_to(ROOT)} holds, at {site.precision} dp:\n  "
        + "\n  ".join(stale)
        + f"\n\nThe artifact is the source of truth; the prose is what is stale. "
        f"Region checked:\n{region.strip()}"
    )
