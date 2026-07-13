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
// Response. Mirrors ScoreResponse key-for-key.
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
}

/** serving/schema.py's CalibratorResponse -- the shipped calibrator's own shape,
 *  read off the same bundle /score decides with. Defined now; the step-function
 *  explainer that consumes it is the next step. */
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
// Errors. 422 and 503 are real, documented states of this API (serving/errors.py),
// not generic failures, so they get their own types rather than a string.
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

export class AdditivityError extends Error {
  detail: string
  constructor(detail: string) {
    super(detail)
    this.name = "AdditivityError"
    this.detail = detail
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "NetworkError"
  }
}

async function handle(res: Response): Promise<unknown> {
  if (res.ok) return res.json()

  let body: { detail?: unknown } = {}
  try {
    body = await res.json()
  } catch {
    /* a non-JSON error body; fall through to the status-based branches */
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
    // ADDITIVITY_FAILURE_DETAIL: the explanation did not reconstruct the score,
    // so the service refused to return a decision at all. A decision without a
    // valid explanation is worse than no decision.
    throw new AdditivityError(String(body.detail ?? "Internal consistency check failed."))
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
  return (await handle(res)) as ScoreResponse
}

/** GET /calibrator. Defined now; consumed by the step-function explainer next. */
export async function getCalibrator(): Promise<CalibratorResponse> {
  let res: Response
  try {
    res = await fetch(`${BASE}/calibrator`)
  } catch {
    throw new NetworkError(`Could not reach the API at ${BASE}.`)
  }
  return (await handle(res)) as CalibratorResponse
}

export const API_BASE = BASE
