"""
Generate src/lib/enums.generated.ts from the REAL Python value sets.

Why this exists at all: the form's three closed-enum dropdowns (purpose,
home_ownership_n, emp_length) must offer exactly the values the model was fit on
and the API will accept. A hand-typed copy of those lists in TypeScript is a
second source of truth, and a second source of truth is the drift this repo's
feature contract exists to prevent -- src/features.py's own comment makes the
same argument about category lists, and serving/schema.py imports every bound and
category set from src.data_validation rather than retyping them.

So the lists are not typed here either. They are READ from the same modules
serving/schema.py imports them from, and written into a generated file that is
never hand-edited. package.json runs this before `dev` and before `build`, so the
TypeScript cannot be stale against the Python: a value added to VALID_PURPOSE
appears in the dropdown on the next `npm run dev`, with no one having to remember.

emp_order's ORDER is preserved, not sorted. It is an ordinal feature
(features.py's add_features() maps it to emp_length_ord), so "< 1 year" precedes
"10+ years" because that is what the encoding means -- alphabetical order would
put "10+ years" first and quietly misrepresent the scale to anyone reading the
dropdown.

Run:  uv run python frontend/scripts/generate-enums.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from serving.schema import EMP_LENGTH_NOT_DISCLOSED  # noqa: E402
from src.data_validation import (  # noqa: E402
    DTI_MAX_REAL,
    DTI_SENTINEL,
    LOAN_MAX,
    LOAN_MIN,
    REVENUE_MIN,
    VALID_HOME_OWNERSHIP,
    VALID_PURPOSE,
)
from src.features import emp_order  # noqa: E402

OUT = REPO_ROOT / "frontend" / "src" / "lib" / "enums.generated.ts"


def ts_string_array(values: list[str]) -> str:
    inner = ",\n  ".join(f'"{v}"' for v in values)
    return f"[\n  {inner},\n]"


def main() -> None:
    # purpose and home_ownership_n are frozensets -- unordered by nature, so they
    # are sorted for a stable file (a set's iteration order is not a contract).
    # emp_order is a LIST and its order IS the contract: it is the ordinal scale.
    purpose = sorted(VALID_PURPOSE)
    home_ownership = sorted(VALID_HOME_OWNERSHIP)
    emp = list(emp_order)

    src = f'''// GENERATED FILE -- DO NOT EDIT.
//
// Written by frontend/scripts/generate-enums.py, which reads these values from
// the same Python modules serving/schema.py imports them from
// (src/data_validation.py, src/features.py). package.json regenerates this
// before `dev` and `build`, so it cannot drift from the API's real contract.
//
// Edit the Python sets, not this file.

/** src.data_validation.VALID_PURPOSE ({len(purpose)} values). Closed enum: an unseen
 *  purpose is a 422, never scored from LightGBM's untrained NaN bin. */
export const VALID_PURPOSE: readonly string[] = {ts_string_array(purpose)} as const

/** src.data_validation.VALID_HOME_OWNERSHIP ({len(home_ownership)} values). */
export const VALID_HOME_OWNERSHIP: readonly string[] = {ts_string_array(home_ownership)} as const

/** src.features.emp_order ({len(emp)} values), IN ORDER -- this is an ordinal
 *  scale (add_features() maps it to emp_length_ord), so the order is meaning,
 *  not presentation. Do not sort. */
export const EMP_ORDER: readonly string[] = {ts_string_array(emp)} as const

/** serving/schema.py's EMP_LENGTH_NOT_DISCLOSED. JSON null on emp_length MEANS
 *  "declined to disclose" and the backend normalizes it to exactly this string.
 *  It is a first-class choice, not a blank. */
export const EMP_LENGTH_NOT_DISCLOSED = "{EMP_LENGTH_NOT_DISCLOSED}"

/** Numeric bounds ScoreRequest enforces (src/data_validation.py). Mirrored here
 *  so the form can refuse a value the API would 422 -- the API remains the
 *  authority; this is a courtesy, not a second gate. */
export const REVENUE_MIN = {REVENUE_MIN}
export const LOAN_MIN = {LOAN_MIN}
export const LOAN_MAX = {LOAN_MAX}
export const DTI_MAX_REAL = {DTI_MAX_REAL}
export const DTI_SENTINEL = {DTI_SENTINEL}
'''

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(src, encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(REPO_ROOT)}: "
        f"{len(purpose)} purpose, {len(home_ownership)} home_ownership, "
        f"{len(emp)} emp_length (+ '{EMP_LENGTH_NOT_DISCLOSED}')"
    )


if __name__ == "__main__":
    main()
