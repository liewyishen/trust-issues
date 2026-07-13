# frontend

A browser client for the `trust-issues` scoring API. It submits a loan application to the
real FastAPI service, the real model scores it, and the page renders the real decision.

**No number describing the applicant is computed in the browser** — every one comes off the API
response. There are exactly two things the page works out for itself, and both exist to be
*checked* rather than trusted: tier 4 re-adds the SHAP contributions to see whether they
reconstruct the score, and tier 3 re-reads the served calibrator curve to see whether it
reproduces the probability the API returned. Neither can change a decision; both can catch one.

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

## Compare two applicants

The **Compare two** tab scores two applicants against the real `/score` and shows what the
difference between them did. It exists because the interesting question about this model is
almost never "what is this applicant's probability" — it is "what did *changing that* do".

The move it makes cheap: **copy A → B, change exactly one field, score both.** That is a
ceteris-paribus comparison, and the view says so — it counts the differing fields and tells you
when there is more than one, because with two levers pulled at once no result is attributable
to either.

`applicant_id` is the exception it warns about loudly. It is not a model feature — it is the
bureau's key, and `MockBureau` hashes it to seed the FICO draw. Change it and you have changed
the credit score too, so the comparison is no longer ceteris paribus. The view detects that and
says it, with both FICOs shown, rather than letting you credit the outcome to whatever else you
edited.

**The headline is positional.** Both applicants are plotted on one calibrator chart. When they
land on the same flat block you see them standing on it together, and the summary states the
thing the chart shows:

> **The log-odds moved. The probability did not.**

Which is not a bug and not a rounding artifact. Changing `home_ownership` from `RENT` to `OTHER`
moves the raw margin by `−0.075518` log-odds and moves `home_ownership_n`'s own contribution
from `+0.062703` to `−0.026266` — the model saw the change perfectly well. Both applicants still
land on block `#47`, a flat block, where `dp_cal/dp_raw` is exactly `0`. So `p_calibrated` is
`46.60%` for both and the decision cannot tell them apart. **This is what "attribution in
probability points is undefined" looks like from the outside: there are no probability points to
assign, because the probability did not move at all.**

Everything in the diff is in **log-odds**, and there is no probability-point delta anywhere in
it. Same reason as everywhere else in this app.

Two traps the comparison is careful about, because both would otherwise mislead:

- **A dropped reason code is not a demoted one.** When `home_ownership_n`'s contribution turns
  *negative*, it does not fall in rank — it stops being a principal adverse factor at all,
  because an adverse-action notice can only name factors that pushed the applicant *toward*
  default. The view says that, rather than letting a vanishing row read as a glitch.
- **A moved contribution is not a moved input.** SHAP attributes over the whole feature vector,
  so editing `dti_n` shifts what `fico_n` gets credited with — `1.389652 → 1.277808` — while the
  FICO itself is *bit-identical* (`631.6342719875626`, same applicant id, same bureau pull). Any
  feature whose input was the same in A and B is tagged **`input unchanged`** in the delta table.
  Without that tag a reader would see `fico_n  Δ −0.11` and conclude the bureau returned a
  different score. It did not.

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

## The calibrator, drawn

`CalibratorExplainer` sits between the decision and the additive explanation, and it is there
to answer the question those two raise between them: tier 2 shows a probability, tier 4 shows
contributions in log-odds and refuses to convert them into probability points. Why?

Because of the shape. The shipped calibrator is a **step function** — flat blocks joined by
ramps — so `dp_cal/dp_raw` is **exactly zero almost everywhere**. "This feature is worth N
points of probability" is not a number nobody has bothered to compute; there is no number.
`docs/explainability.md` measures the flat fraction of the reject region at **99.31%** of the
`p_raw` axis (a statement about the *axis*, not about how many applicants sit on it — the
distinction is the doc's, and the UI keeps it).

Drag the probe and you feel it: `p_calibrated` holds flat, holds flat, holds flat, then jumps.

**Every knot, the domain and the decision threshold come from `GET /calibrator`.** Nothing is
hardcoded, and that is load-bearing rather than tidy. The repo's claim is *the calibrator you
explain is the calibrator you ship* — a snapshot baked into this client would keep drawing a
confident picture of an artifact that had since been retrained, which is precisely the failure
the claim exists to rule out. Retrain the calibrator and the plot changes, or it is lying.

Three things the rendering will not do:

- **It does not smooth.** All 104 knots are emitted in order and joined by straight segments,
  because that is what a piecewise-linear function *is*. The ramps come out under a pixel wide
  (`9.1e-09` to `1.5e-03`) because they *are* that narrow. A curve-fitted rendering would be a
  lie about the artifact, and the artifact's discreteness is the whole finding.
- **It does not hide the clipping.** `out_of_bounds="clip"` means every `p_raw` above `x_max`
  returns one identical value — **40.3% of the `[0, 1]` axis collapsing to a single output**.
  That is drawn, shaded and labelled, not trimmed away as an edge case.
- **It does not recompute the applicant.** Their point is the API's own `p_raw` and
  `p_calibrated`, plotted. Only the *exploratory probe* is evaluated client-side, and it is
  labelled as a reading of the server's curve rather than as a score.

That client-side reading is exact — it transcribes
`IsotonicRegression(out_of_bounds="clip").predict`, and was checked against the real shipped
calibrator over 40,212 probes (every knot, every block and ramp midpoint, uniform draws across
the whole axis): max error `5.55e-17`. You do not have to take that sentence's word for it. The
panel re-runs the check live at the applicant's own `p_raw`, compares it to the
`p_calibrated` the API returned, and says on screen whether it reproduced it — the same move
tier 4 makes with the additivity sum.

**The ramps cannot be dragged onto.** The slider steps at `1e-6` across ~600px, so one pixel of
travel is ~1600 steps, and every ramp is narrower than a pixel. That is not a defect of the
control — it is the artifact, felt. The `‹ ›` buttons jump to a ramp deliberately, since it
cannot be reached by accident. One of the 51 is the entire decision boundary: `p_cal` goes
`24.15% → 26.48%` across **4.0 parts per million** of `p_raw`, at a slope of `5829.5`. Below
that sliver, approve. Above it, reject. The panel flags that ramp when you land on it.
