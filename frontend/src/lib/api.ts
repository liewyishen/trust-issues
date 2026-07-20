/**
 * The typed client for the trust-issues scoring API.
 *
 * Types mirror serving/schema.py. Where they disagree, schema.py wins -- it is
 * the contract; this file is a client of it.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000"

// --------------------------------------------------------------------------
// Request. serving/schema.py's ScoreRequest: applicant_id + SIX applicant-
// reported fields. There is deliberately no fico_n -- the service fetches it
// from the credit bureau (CreditBureau.fetch, serving/bureau.py) and
// extra="forbid" means a client that submits its own gets a 422. A client that
// could set its own FICO could describe an applicant whose score never came
// from a bureau pull at all.
// --------------------------------------------------------------------------
export interface ScoreRequest {
  applicant_id: string
  revenue: number
  dti_n: number
  loan_amnt: number
  /** null MEANS "declined to disclose"; the backend normalizes it to "NI". */
  emp_length: string | null
  purpose: string
  home_ownership_n: string
}

// --------------------------------------------------------------------------
// Response. Mirrors serving/schema.py's ExplainedScoreResponse -- which is
// ScoreResponse plus exactly one field, `explanation`. The interface below
// keeps the base class's name because that is what 29 references across six
// files already call it; the wire type is the subclass. Naming the subclass
// here would be more precise and would also be the whole of the diff, which is
// why it was left -- but "mirrors ScoreResponse key-for-key" was false the
// moment /score started returning the subclass, and a mirror claim that has
// stopped being true is the exact staleness this client is written against.
//
// There is no `contribution_to_probability`, here or there, and its absence is
// not an omission: docs/explainability.md proves percentage-point attribution is
// UNDEFINED under the shipped 52-level isotonic calibrator (zero slope across
// 99.31% of the reject region). A field for it would tell the next reader that a
// value belongs there and someone merely has not computed it. Nothing belongs
// there. This client does not invent one either.
// --------------------------------------------------------------------------
export interface ReasonCode {
  rank: number
  feature: string
  /** A string for EVERY feature, including numeric ones (_rank_adverse writes
   *  str(v)). An "NI" applicant's emp_length_ord arrives as the string "nan". */
  value: string
  contribution_log_odds: number
}

/** serving/schema.py's ScoredCreditReport: the credit value the model actually
 *  scored on, plus what identifies the pull that supplied it. The bureau's
 *  dti_n is deliberately NOT here -- the decision does not use it. */
export interface ScoredCreditReport {
  fico_n: number
  bureau: string
  fico_version: string
  pulled_at: string
}

export interface ScoreResponse {
  scale: string
  /** sigmoid(margin) -- the calibrator's INPUT. */
  p_raw: number
  /** the calibrator's OUTPUT, and the quantity actually decided on. */
  p_calibrated: number
  threshold: number
  decision: "APPROVE" | "REJECT"
  base_value_log_odds: number
  /** == base_value_log_odds + sum(contributions_log_odds). The backend's
   *  additivity guard 500s rather than return a response where it does not. */
  raw_margin_log_odds: number
  contributions_log_odds: Record<string, number>
  reason_codes: ReasonCode[]
  model_trained_at: string | null
  calibrator_trained_at: string | null
  credit_report: ScoredCreditReport
  /** The one field ExplainedScoreResponse adds. Plain-language prose rendered
   *  by serving/render.py -- pure code, no model, self-checked against
   *  explanation_fragments() before /score will return it. Multi-line, with a
   *  two-space-indented factor list. This client displays it BYTE FOR BYTE and
   *  never parses, splits, reflows or reformats it: the renderer's care is all
   *  in the exact wording (it says "the length of your employment history"
   *  rather than "10 years", because emp_order maps "10+ years" to 10), and
   *  every one of those choices survives only if nothing here touches it. */
  explanation: string
}

/** serving/schema.py's CalibratorResponse -- the shipped calibrator's own shape,
 *  read off the same bundle /score decides with. Consumed by CalibratorExplainer,
 *  which draws it. Nothing in this client holds a copy of these knots: the plot
 *  is a rendering of whatever the service is serving today, so a retrain changes
 *  the picture. A baked-in snapshot would break exactly the claim the plot makes. */
export interface CalibratorResponse {
  x_thresholds: number[]
  y_thresholds: number[]
  x_min: number
  x_max: number
  n_knots: number
  n_distinct_y: number
  /** SELECTED_THRESHOLD -- the value /score decides at. NOT the literal 0.25,
   *  which is a different float (serving/config.py). Draw the line here. */
  threshold: number
}

// --------------------------------------------------------------------------
// serving/schema.py's DriftResponse. POST /drift turns MockBureau's mean_fico
// knob and returns what pipelines/drift_check.py -- the REAL monitor, the one the
// batch job runs -- made of the shifted population.
//
// Nothing in this client computes a PSI. It could not do so honestly: PSI needs
// the reference population's quantile bin edges, and those live in the monitor.
// Every number below is the monitor's own.
// --------------------------------------------------------------------------
export interface FeatureDrift {
  psi: number
  ks: number
  /** evaluate_alarms()'s verdict on THIS column's psi and ks -- not a re-test of
   *  psi against a threshold. The monitor alarms on EITHER signal crossing, and
   *  the two do not cross together: ks is the more sensitive of the pair and
   *  fires first. Swept in 1-point steps: at mean_fico 691 ks is 0.0980 and the
   *  feature is QUIET; at 690 ks is 0.1045 and it ALARMS -- while psi does not
   *  cross 0.25 until 676 (0.2520; at 677 it is still 0.2352). So across the
   *  whole band 677..690 a feature is `alarmed` with its PSI under the line, and
   *  the UI must not present `alarmed` as if it meant "PSI crossed". */
  alarmed: boolean
}

export interface DriftResponse {
  /** the knob, as requested */
  mean_fico: number
  reference_mean_fico: number
  /** what the batch actually DREW. MockBureau clips into [FICO_MIN, FICO_MAX],
   *  so this can differ from the requested mean -- showing only the request
   *  would hide the clipping at exactly the settings where it dominates. */
  observed_mean_fico_reference: number
  observed_mean_fico_current: number
  /** MONITOR SLICE LABELS (2015/2016), not real years, and not a claim about
   *  real LendingClub data. Returned so the flat `metrics` keys can be built
   *  (psi_fico_n_2016, ...) instead of hardcoding a year. */
  reference_year: number
  current_year: number
  n_reference: number
  n_current: number
  /** keyed by column: "fico_n" (what the knob moves) and "dti_n" (the control). */
  features: Record<string, FeatureDrift>
  /** DEFAULT_ALARM_THRESHOLDS verbatim -- the whole ruleset, not the two lines
   *  that flatter the demo. Draw the alarm line HERE and nowhere else. */
  thresholds: Record<string, number>
  /** evaluate_alarms()'s own strings, unedited -- including the dti tripwire,
   *  which fires at EVERY setting of the knob and is a disclosed artifact of
   *  MockBureau's synthetic dti_n, not drift. It is not filtered out; dropping
   *  the one alarm that embarrasses the demo is the dishonest edit available
   *  here. */
  alarms: string[]
  /** drift_metrics()'s flat dict as logged, the thing `features` was re-keyed
   *  FROM -- shipped so the re-keying is checkable rather than trusted. */
  metrics: Record<string, number>
}

// --------------------------------------------------------------------------
// serving/schema.py's FairnessResponse. GET /fairness serves the FROZEN
// three-layer audit (models/fairness_audit.json), produced offline by
// scripts/audit_fairness.py from the real src/fairness.py.
//
// Frozen, not live, and the reason is a hard one: the audit needs the 167 MB
// assessment CSV -- the first line of .dockerignore, because the brief forbids
// redistributing it -- and ~40s to retrain both ablation variants. Unlike
// /drift, no amount of engineering makes it a request. What ships is its
// OUTPUT: derived aggregate ratios, which are not the dataset.
//
// Nothing in this client computes a ratio, a CI, or a verdict. It draws what
// the audit found.
// --------------------------------------------------------------------------

/** Which model the audit ran against. `trained_at` is THE binding field: the
 *  service compares it to the shipped booster's and 409s on mismatch. */
export interface AuditedModel {
  trained_at: string | null
  calibrator_trained_at: string | null
  best_iteration: number
  /** The fairness conclusion, executed and checkable rather than asserted in
   *  prose: the model that made these decisions has no addr_state to lean on. */
  features: string[]
  includes_addr_state: boolean
}

/** src/fairness.py's OWN constants. Draw the 0.80 line HERE and nowhere else --
 *  the same discipline DriftResponse.thresholds enforces for the alarm line. */
export interface AuditConstants {
  eo_threshold: number
  min_n: number
  n_boot: number
  sweep_thresholds: number[]
  /** 0.22 -- deliberately NOT the operating threshold. Section 9.2's sweep found
   *  the disparity most visible there, and an ablation wants its comparison where
   *  the signal is strongest. The divergence is the point, not a mistake. */
  ablation_threshold: number
  watch_states: string[]
}

/** One state's Equal-Opportunity ratio, with the interval around it. The CI is
 *  the load-bearing part: a point estimate below 0.80 on a finite sample is
 *  noise, not evidence, which is why Layer 1 bootstraps instead of trusting a
 *  groupby mean. */
export interface StateEO {
  state: string
  n_good: number
  eo_ratio: number
  ci_low: number
  ci_high: number
  /** "confirmed (CI fully < 0.80)" | "inconclusive (CI straddles 0.80)" | "clear" */
  verdict: string
}

export interface Layer1 {
  /** SELECTED_THRESHOLD -- the point /score actually decides at. */
  threshold: number
  states: StateEO[]
  n_confirmed: number
}

/** Each row is {threshold, national_good_approval_rate, ...one key per watch
 *  state}. The watch-state keys are DERIVED from constants.watch_states, never
 *  hardcoded here -- typing them as named fields would silently drop a column
 *  the day src/fairness.py's WATCH_STATES changes. */
export interface Layer2 {
  rows: Record<string, number>[]
}

/** One state on BOTH sides of the ablation, each side with a bootstrap CI. This
 *  is the pair the audit could not report until Layer 3 started returning its
 *  Test frames: same applicants, same outcomes, one feature toggled. */
export interface AblationStateCI {
  state: string
  n_good_with_state: number
  eo_ratio_with_state: number
  ci_low_with_state: number
  ci_high_with_state: number
  verdict_with_state: string
  n_good_no_state: number
  eo_ratio_no_state: number
  ci_low_no_state: number
  ci_high_no_state: number
  verdict_no_state: string
}

/** src/fairness.py's own Layer-3 verdict row -- decided on POINT ESTIMATES, and
 *  therefore capable of honestly disagreeing with the CI verdicts above. On the
 *  real data it does: NV reads "was already clear" on eo_with_state = 0.800147,
 *  a verdict turning on the fourth decimal, while its CI straddles 0.80. Both
 *  are shipped so the UI can show that rather than pick whichever reads better. */
export interface AblationWatchState {
  state: string
  eo_with_state: number
  eo_no_state: number
  shift: number
  verdict: string
}

export interface Layer3 {
  threshold: number
  auc_with_state: number
  auc_no_state: number
  /** NEGATIVE: dropping addr_state costs AUC. That is the price, and it is paid. */
  auc_cost: number
  base_approval_with_state: number
  base_approval_no_state: number
  watch: AblationWatchState[]
  states: AblationStateCI[]
}

export interface FairnessResponse {
  schema_version: number
  generated_at: string
  model: AuditedModel
  /** Read off the LIVE bundle at request time, not copied from the artifact. It
   *  is echoed beside model.trained_at so a client can SEE the two agree, rather
   *  than infer it from the absence of a 409. */
  shipped_model_trained_at: string | null
  constants: AuditConstants
  layer1: Layer1
  layer2: Layer2
  layer3: Layer3
}

// --------------------------------------------------------------------------
// Errors. 422, 503 and 409 are real, documented states of this API
// (serving/errors.py, serving/fairness.py), not generic failures, so they get
// their own types rather than a string.
// --------------------------------------------------------------------------
export interface ValidationIssue {
  /** e.g. ["body", "emp_length"] */
  loc: (string | number)[]
  msg: string
  type: string
}

export class ValidationError extends Error {
  issues: ValidationIssue[]
  constructor(issues: ValidationIssue[]) {
    super("The request was rejected by the API's validation contract.")
    this.name = "ValidationError"
    this.issues = issues
  }
}

export class ServiceUnavailableError extends Error {
  detail: string
  constructor(detail: string) {
    super(detail)
    this.name = "ServiceUnavailableError"
    this.detail = detail
  }
}

/** A 500. NOT necessarily the additivity guard -- this class used to be called
 *  AdditivityError, and every 500 from every route was constructed as one, so a
 *  renderer refusal on /score and an unhandled bug in /fairness both reached the
 *  screen wearing "the contributions did not reconstruct the model's own score".
 *
 *  `detail` is the sentence the SERVICE wrote, or null when it wrote none.
 *  serving/app.py's HTTPException is the only explicit 500 in serving/ and it
 *  answers JSON with a detail; every other 500 is Starlette's default handler
 *  answering text/plain, which res.json() cannot parse. Pinned by
 *  tests/test_serving.py::test_a_non_additivity_500_carries_no_json_detail.
 *
 *  A detail proves the service chose to say something. It does NOT prove WHICH
 *  guard said it, and this client no longer claims to know -- it shows the
 *  sentence and stops, the same posture PlainLanguageNotice takes toward the
 *  rendered explanation. */
export class InternalServerError extends Error {
  detail: string | null
  constructor(detail: string | null) {
    super(detail ?? "The service failed and returned no result.")
    this.name = "InternalServerError"
    this.detail = detail
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "NetworkError"
  }
}

/**
 * 409 -- the audit on disk is about a DIFFERENT model than the one being served.
 *
 * The only route with this state is /fairness, and it is the price of the audit
 * being a frozen artifact rather than a live read. /calibrator cannot go stale:
 * it reads the live bundle, so whatever it returns IS what /score decides with.
 * A JSON file has no such protection -- retrain the booster and it still
 * cheerfully reports Mississippi at 0.7448, about a model that no longer exists.
 *
 * So the service binds the artifact to the model by `trained_at` (the same
 * binding load_calibrator() enforces between the calibrator and the booster) and,
 * on mismatch, sends this: both timestamps and NOT ONE RATIO.
 *
 * That the payload carries no numbers is deliberate, and this client must not
 * treat it as an inconvenience to route around. A client handed stale ratios with
 * a warning attached WILL draw the ratios. The only reliable way to stop a stale
 * number being rendered as a current one is to never have it.
 */
export class AuditStaleError extends Error {
  auditModelTrainedAt: string | null
  shippedModelTrainedAt: string | null
  constructor(detail: {
    error?: string
    audit_model_trained_at?: string | null
    shipped_model_trained_at?: string | null
  }) {
    super(detail.error ?? "The fairness audit does not describe the shipped model.")
    this.name = "AuditStaleError"
    this.auditModelTrainedAt = detail.audit_model_trained_at ?? null
    this.shippedModelTrainedAt = detail.shipped_model_trained_at ?? null
  }
}

/**
 * 404 -- the route is not mounted on this deployment.
 *
 * A real, expected state, and the only route that has it today is /drift. It is
 * gated on DRIFT_DEMO_AVAILABLE (serving/app.py): the drift monitor lives in
 * pipelines/, which the slim serving image does not copy and whose mlflow and
 * metaflow dependencies it does not install. So /drift is present in local dev
 * and simply absent from the container.
 *
 * The client must render that as what it is. A 404 says "not here" -- and the
 * one thing this app must never do in response is draw the chart anyway from
 * something it made up. An empty state that admits the endpoint is missing is
 * strictly more useful than a plausible curve that came from nowhere.
 */
export class RouteNotAvailableError extends Error {
  constructor(route: string) {
    super(`${route} is not available on this deployment.`)
    this.name = "RouteNotAvailableError"
  }
}

async function handle(res: Response, route: string): Promise<unknown> {
  if (res.ok) return res.json()

  let body: { detail?: unknown } = {}
  try {
    body = await res.json()
  } catch {
    /* a non-JSON error body; fall through to the status-based branches */
  }

  if (res.status === 404) {
    // The route is not mounted here. See RouteNotAvailableError.
    throw new RouteNotAvailableError(route)
  }
  if (res.status === 409) {
    // /fairness only: the frozen audit describes another model. See AuditStaleError.
    // The body carries both timestamps and no ratios, by design.
    const detail = (body.detail ?? {}) as Record<string, string | null>
    throw new AuditStaleError(detail)
  }
  if (res.status === 422) {
    // FastAPI's RequestValidationError handler: detail is an array of issues,
    // each with a `loc` path and a human `msg`. Surfaced field-by-field rather
    // than flattened, so a bad value points at the field that produced it.
    const detail = Array.isArray(body.detail) ? (body.detail as ValidationIssue[]) : []
    throw new ValidationError(detail)
  }
  if (res.status === 503) {
    // ARTIFACTS_UNAVAILABLE_DETAIL / BUREAU_UNAVAILABLE_DETAIL (serving/errors.py).
    throw new ServiceUnavailableError(String(body.detail ?? "Service is not ready."))
  }
  if (res.status === 500) {
    // Pass the service's own sentence through, or null if it sent none. The
    // `??` that used to sit here substituted "Internal consistency check
    // failed." for a MISSING detail, which turned every non-additivity 500 into
    // a claim about the additivity guard. Absence of a detail is information;
    // filling it in destroyed the one bit that distinguished the two shapes.
    throw new InternalServerError(typeof body.detail === "string" ? body.detail : null)
  }
  throw new NetworkError(`Unexpected ${res.status} from the API.`)
}

/**
 * POST /score.
 *
 * The payload's numbers must be NUMBERS. ScoreRequest's floats are
 * Field(strict=True): the API accepts int 700 and float 700.0 and REJECTS the
 * string "700" with a 422 -- not because it cannot parse it, but because a
 * client that sent it has a bug. HTML number inputs yield strings, so the form
 * coerces before it gets here; this function does not re-coerce, because
 * silently accepting a string would reintroduce exactly the laxness the API
 * refuses.
 */
export async function scoreApplicant(payload: ScoreRequest): Promise<ScoreResponse> {
  let res: Response
  try {
    res = await fetch(`${BASE}/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new NetworkError(
      `Could not reach the API at ${BASE}. Is uvicorn running on that port?`,
    )
  }
  return (await handle(res, "/score")) as ScoreResponse
}

/** GET /calibrator. The shipped calibrator's knots, domain and decision threshold,
 *  read off the bundle /score decides with. Fetched once per page load -- it is an
 *  artifact, not a per-applicant computation. */
export async function getCalibrator(): Promise<CalibratorResponse> {
  let res: Response
  try {
    res = await fetch(`${BASE}/calibrator`)
  } catch {
    throw new NetworkError(`Could not reach the API at ${BASE}.`)
  }
  return (await handle(res, "/calibrator")) as CalibratorResponse
}

/**
 * POST /drift. Turn the market's mean FICO to `mean_fico` and ask the real
 * monitor what it makes of the shift.
 *
 * Deterministic: the same mean_fico returns byte-identical numbers, every time,
 * in every process (fixed applicant ids, hash-seeded MockBureau). That is what
 * lets a slider produce a CURVE rather than a shimmer -- drag back to 700 and you
 * land on exactly the numbers you left.
 *
 * Expect the FIRST call to cost ~0.40s and every later one ~50ms (measured, on a
 * cold uvicorn: 0.409 / 0.397 across two restarts, then 0.053). The handler
 * imports the monitor lazily (serving/app.py), because importing it at module
 * scope would drag mlflow and metaflow into the serving image and kill the
 * container at boot. The cost is real and paid once; callers should warm it.
 *
 * NOT the ~1.3s that `import scripts.demo_drift` costs a bare interpreter. Most of
 * that 1.3s is pandas/scipy/sklearn, and uvicorn has already imported all of them
 * at startup for /score -- so what the first /drift request actually pays is only
 * mlflow and metaflow's marginal cost on top of a warm graph. The bare-interpreter
 * figure overstates the user-visible cost by about 3x, and it is the wrong number
 * to design the loading state around.
 *
 * A 404 here is not a bug -- see RouteNotAvailableError.
 */
export async function getDrift(mean_fico: number): Promise<DriftResponse> {
  let res: Response
  try {
    res = await fetch(`${BASE}/drift`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // A NUMBER, not a string: DriftRequest.mean_fico is Field(strict=True), so
      // "650" is a 422 for the same reason ScoreRequest's floats reject it.
      body: JSON.stringify({ mean_fico }),
    })
  } catch {
    throw new NetworkError(`Could not reach the API at ${BASE}.`)
  }
  return (await handle(res, "/drift")) as DriftResponse
}

/**
 * GET /fairness. The frozen three-layer audit -- but only if it is about the
 * model being served.
 *
 * Cheap (it is a ~35 KB file read, not a computation) and static for the life of
 * the deployment, so it is fetched once when the view mounts. There is no knob:
 * unlike /drift, nothing here is parameterized, because the audit is a
 * population-level fact about a model, not a what-if. A per-applicant fairness
 * view is not merely unbuilt -- it is impossible: addr_state is not a ScoreRequest
 * field, and the shipped model does not carry it as a feature.
 *
 * Two failure modes that are NOT bugs, and neither may be papered over:
 *   404 RouteNotAvailableError -- no audit artifact on this deployment.
 *   409 AuditStaleError        -- the audit is about a different model. No
 *                                 numbers are sent, so none can be drawn.
 */
export async function getFairness(): Promise<FairnessResponse> {
  let res: Response
  try {
    res = await fetch(`${BASE}/fairness`)
  } catch {
    throw new NetworkError(`Could not reach the API at ${BASE}.`)
  }
  return (await handle(res, "/fairness")) as FairnessResponse
}

export const API_BASE = BASE
