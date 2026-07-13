/**
 * The application form's state, validation, and its mapping onto ScoreRequest.
 *
 * This lives OUTSIDE the component because the compare view renders two of that
 * component. A second copy of the field list, the coercion rules or the enum
 * wiring is a second source of truth, and a second source of truth drifts --
 * which is the same argument serving/schema.py makes when it imports every bound
 * from src/data_validation.py rather than retyping them, and the same argument
 * scripts/generate-enums.py makes by reading the category sets out of the real
 * Python. Applicant A and applicant B are two instances of ONE form, not two
 * forms that happen to agree today.
 */

import type { ScoreRequest } from "@/lib/api"

/** The sentinel for "the user chose Not disclosed". It is not a value the API
 *  sees: toRequest() maps it to JSON null, which the backend normalizes to "NI".
 *  Radix Select cannot carry a null item value, hence a local sentinel rather
 *  than an empty string (indistinguishable from "nothing selected yet"). */
export const NOT_DISCLOSED = "__not_disclosed__"

/** Every field is a STRING here, because that is what the DOM yields. The
 *  conversion to numbers happens once, in toRequest(), on the way out. */
export interface FormState {
  applicant_id: string
  revenue: string
  dti_n: string
  loan_amnt: string
  emp_length: string
  purpose: string
  home_ownership_n: string
}

export const INITIAL: FormState = {
  applicant_id: "demo-001",
  revenue: "65000",
  dti_n: "18",
  loan_amnt: "10000",
  emp_length: "5 years",
  purpose: "debt_consolidation",
  home_ownership_n: "RENT",
}

/** Human labels for the diff summary. Keyed on the FormState field so a renamed
 *  field cannot keep a label that describes the old one. */
export const FIELD_LABELS: Record<keyof FormState, string> = {
  applicant_id: "Applicant id",
  revenue: "Annual revenue",
  dti_n: "DTI",
  loan_amnt: "Loan amount",
  emp_length: "Employment length",
  purpose: "Purpose",
  home_ownership_n: "Home ownership",
}

/** applicant_id is NOT a model feature. It is the bureau's key: MockBureau hashes
 *  it with SHA-256 to seed the draw, so it is the only thing that varies the
 *  credit pull. Changing it changes fico_n -- which is exactly why a
 *  ceteris-paribus comparison must usually hold it FIXED. */
export const BUREAU_KEY_FIELD: keyof FormState = "applicant_id"

/** The six fields the model actually sees, in form order. */
export const MODEL_INPUT_FIELDS: Array<keyof FormState> = [
  "revenue",
  "dti_n",
  "loan_amnt",
  "emp_length",
  "purpose",
  "home_ownership_n",
]

/**
 * Which of the model's 8 features each form field feeds.
 *
 * Mostly one-to-one, with two that are not:
 *   emp_length    -> emp_length_ord AND emp_length_missing (add_features, src/features.py)
 *   applicant_id  -> fico_n, via the bureau: MockBureau seeds the FICO draw from a
 *                    SHA-256 of the id, so the id is an input to the model's feature
 *                    vector even though it is not itself a feature.
 *
 * This exists so the compare view can tell an HONEST story about a moved
 * contribution. A feature's SHAP attribution can change when its own input did
 * not -- TreeSHAP attributes over the whole feature vector, so moving dti_n
 * shifts what fico_n gets credited with even though fico_n itself is untouched.
 * Without this map the UI would show `fico_n  Δ −0.11` next to an unchanged
 * applicant id and let a reader conclude the bureau returned a different score.
 * It did not.
 */
export const FIELD_TO_FEATURES: Record<keyof FormState, string[]> = {
  applicant_id: ["fico_n"],
  revenue: ["revenue"],
  dti_n: ["dti_n"],
  loan_amnt: ["loan_amnt"],
  emp_length: ["emp_length_ord", "emp_length_missing"],
  purpose: ["purpose"],
  home_ownership_n: ["home_ownership_n"],
}

/** The model features downstream of a set of edited form fields. */
export function featuresTouchedBy(fields: Array<keyof FormState>): Set<string> {
  return new Set(fields.flatMap((f) => FIELD_TO_FEATURES[f]))
}

export function randomApplicantId(): string {
  const n = Math.floor(Math.random() * 1_000_000)
    .toString()
    .padStart(6, "0")
  return `applicant-${n}`
}

/** Numeric fields arrive from the DOM as strings. ScoreRequest's floats are
 *  Field(strict=True) -- the API accepts 700 and 700.0 and REJECTS "700" with a
 *  422. So the string is converted here, and an unconvertible one never leaves
 *  the browser: a NaN on the wire would be a client bug dressed up as an
 *  applicant. */
export function numeric(raw: string): number | null {
  const trimmed = raw.trim()
  if (trimmed === "") return null
  const n = Number(trimmed)
  return Number.isFinite(n) ? n : null
}

/**
 * The client's inline validation -- a COURTESY, not a gate. The API re-checks
 * everything and is the thing that returns 422. This only tries not to waste a
 * round trip. The bounds it cites are imported from the generated enums, which
 * are read out of src/data_validation.py; they are not restated here.
 */
export function validate(
  f: FormState,
  bounds: { REVENUE_MIN: number; LOAN_MIN: number; LOAN_MAX: number },
): Partial<Record<keyof FormState, string>> {
  const errors: Partial<Record<keyof FormState, string>> = {}

  if (f.applicant_id.trim() === "") {
    errors.applicant_id = "Required — the bureau is keyed on it."
  }

  const revenue = numeric(f.revenue)
  if (revenue === null) errors.revenue = "Enter a number."
  else if (revenue < bounds.REVENUE_MIN) errors.revenue = `Must be at least ${bounds.REVENUE_MIN}.`

  const dti = numeric(f.dti_n)
  if (dti === null) errors.dti_n = "Enter a number."
  else if (dti < 0) errors.dti_n = "DTI is never negative."

  const loan = numeric(f.loan_amnt)
  if (loan === null) errors.loan_amnt = "Enter a number."
  else if (loan < bounds.LOAN_MIN || loan > bounds.LOAN_MAX) {
    errors.loan_amnt = `Must be between ${bounds.LOAN_MIN} and ${bounds.LOAN_MAX}.`
  }

  return errors
}

export function isValid(errors: Partial<Record<keyof FormState, string>>): boolean {
  return Object.keys(errors).length === 0
}

/** FormState -> the wire. Call only on a state that validate() accepted; the
 *  non-null assertions are what that precondition buys. */
export function toRequest(f: FormState): ScoreRequest {
  return {
    applicant_id: f.applicant_id.trim(),
    revenue: numeric(f.revenue)!,
    dti_n: numeric(f.dti_n)!,
    loan_amnt: numeric(f.loan_amnt)!,
    // "Not disclosed" -> JSON null. The backend normalizes null to "NI", where
    // the model has 453,804 rows of support. Declining to state your employment
    // length is a real answer, not a missing one.
    emp_length: f.emp_length === NOT_DISCLOSED ? null : f.emp_length,
    purpose: f.purpose,
    home_ownership_n: f.home_ownership_n,
  }
}

/** What the user actually typed, for display in the diff. The sentinel is not a
 *  value the API ever sees, so it must not be shown as one. */
export function displayValue(f: FormState, k: keyof FormState): string {
  const v = f[k]
  if (k === "emp_length" && v === NOT_DISCLOSED) return "Not disclosed (null → NI)"
  return v
}

/**
 * The fields on which two applicants differ -- the levers that were pulled.
 *
 * Compared on the SUBMITTED value, not the raw string, so "18" and "18.0" do not
 * register as a change the model never saw. A diff that reported a difference the
 * API cannot observe would be a lie in the user's favour: it would credit an
 * edit with an effect it did not have.
 */
export function differingFields(a: FormState, b: FormState): Array<keyof FormState> {
  const [ra, rb] = [toRequest(a), toRequest(b)]
  return (Object.keys(FIELD_LABELS) as Array<keyof FormState>).filter(
    (k) => ra[k as keyof ScoreRequest] !== rb[k as keyof ScoreRequest],
  )
}
