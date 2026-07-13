# frontend

A browser client for the `trust-issues` scoring API. It submits a loan application to the
real FastAPI service, the real model scores it, and the page renders the real decision.

**Nothing on the result page is computed in the browser** — every number comes off the API
response — with exactly one exception, and that exception is the point: tier 3 re-adds the
SHAP contributions itself, so a reader can *check* the backend's additivity claim instead of
trusting it.

## Run both sides

The API and the UI are separate processes. You need both.

```bash
# terminal 1 — the API, from the repo root
uv run uvicorn serving.app:app --reload          # → http://localhost:8000

# terminal 2 — the UI
cd frontend
npm install
npm run dev                                      # → http://localhost:5173
```

Then open <http://localhost:5173>.

**Port 5173 is not arbitrary.** It is one of the two origins `serving/app.py`'s
`CORS_ALLOW_ORIGINS` enumerates. The API does not use `"*"`, so a UI served from any other
origin is refused by the browser — deliberately. `vite.config.ts` sets `strictPort: true` so
Vite fails loudly rather than silently sliding to 5174 and producing a CORS error that looks
like a bug in the API.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE` | `http://localhost:8000` | Where the FastAPI service lives. |

This is **local-real**, not deployed: the API runs on your machine. Deployment is a recorded
deferral (`docs/design.md`), not an oversight. When the API is eventually deployed, pointing
this UI at it is a one-line change — set `VITE_API_BASE` in `.env` and rebuild. Nothing else
in this app knows the URL.

## The enum lists are generated, not typed

`src/lib/enums.generated.ts` is written by `scripts/generate-enums.py`, which **reads** the
real value sets out of the same Python the API validates against
(`src/data_validation.py`'s `VALID_PURPOSE` / `VALID_HOME_OWNERSHIP`, `src/features.py`'s
`emp_order`). It runs automatically before `npm run dev` and `npm run build`.

A hand-typed copy of those lists would be a second source of truth, and a second source of
truth drifts. `serving/schema.py` imports every bound and category set from
`src/data_validation.py` rather than retyping them for exactly this reason; the frontend
holds the same line. **Edit the Python sets, never the generated file.**

`emp_order`'s order is preserved, never sorted: it is an ordinal scale
(`add_features()` maps it to `emp_length_ord`), so `"< 1 year"` precedes `"10+ years"`
because that is what the encoding *means*.

Consequence worth knowing: `npm run dev` shells out to `uv run python`, so the repo's Python
environment must be installed. Not a burden in practice — this UI is useless without the API,
which needs that environment anyway.

## What the UI enforces, and what it does not

The form renders `purpose`, `home_ownership_n` and `emp_length` as **dropdowns**, so an
invalid value cannot be typed. That is not decoration. `serving/schema.py` measures what an
arbitrary unmapped `emp_length` string does: it encodes *identically* to `null` and to
`"bogus"` once `add_features()` has run — all three collapse to
`(emp_length_ord=NaN, emp_length_missing=0)`, a combination occurring in **0 of Train's
453,804 rows**. The dropdown makes that state unreachable from the UI, rather than merely
rejected after the fact.

`emp_length`'s **"Not disclosed"** option is a first-class choice, not a blank. It submits
JSON `null`, which the backend normalizes to `"NI"` — where the model has real support.
Declining to state your employment length is an answer.

**The form has no FICO input, and never will.** The service fetches `fico_n` from the credit
bureau at scoring time (`CreditBureau.fetch`, `serving/bureau.py`); `ScoreRequest` does not
carry the field, and `extra="forbid"` turns a client-submitted one into a 422. A client that
could set its own FICO could describe an applicant whose score never came from a bureau pull
at all. The number is *displayed* on the result page — under "fetched by the system", with
the pull's provenance beside it.

The client's inline validation is a **courtesy, not a gate**. The API is the authority: it
re-checks everything, and it is the thing that returns 422. The UI merely tries not to waste
a round trip.

Numeric fields are coerced from string to number before the request is sent, and an
unconvertible one never leaves the browser. `ScoreRequest`'s floats are `Field(strict=True)`:
the API accepts `700` and `700.0` and **rejects the string `"700"`** with a 422 — not because
it cannot parse it, but because a client that sent it has a bug.

## Real error states

422 and 503 are documented states of this API (`serving/errors.py`), not generic failures,
and each renders as itself:

- **422** — the request was refused by the validation contract. The per-field `loc` + `msg`
  detail is rendered field by field, so a bad value points at the field that produced it.
- **503** — the service never finished loading its artifacts. It refuses to serve rather than
  score with something it could not verify at startup.
- **500** — the additivity guard failed: the explanation did not reconstruct the score, so
  the service returned no decision at all.

## Not built here (yet)

The **calibrator step-function explainer** — an interactive rendering of the shipped 52-level
isotonic calibrator, drawn from `GET /calibrator`'s real thresholds and its real decision
threshold — is the next step. `getCalibrator()` in `src/lib/api.ts` is already typed and
wired to that endpoint; nothing consumes it yet.
