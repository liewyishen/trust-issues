import { Loader2, Shuffle } from "lucide-react"
import * as React from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { ScoreRequest } from "@/lib/api"
import {
  EMP_ORDER,
  LOAN_MAX,
  LOAN_MIN,
  REVENUE_MIN,
  VALID_HOME_OWNERSHIP,
  VALID_PURPOSE,
} from "@/lib/enums.generated"

/**
 * The six applicant-reported fields, plus applicant_id.
 *
 * There is NO FICO input, and its absence is the design, not an omission: the
 * service fetches fico_n from the credit bureau at scoring time
 * (CreditBureau.fetch, serving/bureau.py). ScoreRequest does not carry the field
 * and extra="forbid" turns a client-submitted one into a 422 -- an applicant who
 * could state their own FICO could describe an applicant whose score never came
 * from a bureau pull.
 *
 * The three closed-enum fields are Selects, never text inputs. serving/schema.py
 * measures why: an arbitrary unmapped string for emp_length encodes IDENTICALLY
 * to null and to "bogus" once add_features() has run -- all three collapse to
 * (emp_length_ord=NaN, emp_length_missing=0), a combination that occurs in 0 of
 * Train's 453,804 rows. A dropdown is what makes that state unreachable from the
 * UI, rather than merely rejected after the fact.
 */

/** The sentinel this form uses for "the user chose Not disclosed". It is not a
 *  value the API sees: it is mapped to JSON null on submit, which the backend's
 *  validator normalizes to "NI". Radix Select cannot carry a null item value,
 *  hence a local sentinel rather than an empty string (which would be
 *  indistinguishable from "nothing selected yet"). */
const NOT_DISCLOSED = "__not_disclosed__"

interface FormState {
  applicant_id: string
  revenue: string
  dti_n: string
  loan_amnt: string
  emp_length: string
  purpose: string
  home_ownership_n: string
}

const INITIAL: FormState = {
  applicant_id: "demo-001",
  revenue: "65000",
  dti_n: "18",
  loan_amnt: "10000",
  emp_length: "5 years",
  purpose: "debt_consolidation",
  home_ownership_n: "RENT",
}

function randomApplicantId(): string {
  // MockBureau hashes applicant_id with SHA-256 to seed the draw, so the id is
  // the only thing that varies the pull. Any string works; this one is legible.
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
function numeric(raw: string): number | null {
  const trimmed = raw.trim()
  if (trimmed === "") return null
  const n = Number(trimmed)
  return Number.isFinite(n) ? n : null
}

function validate(f: FormState): Partial<Record<keyof FormState, string>> {
  const errors: Partial<Record<keyof FormState, string>> = {}

  if (f.applicant_id.trim() === "") errors.applicant_id = "Required — the bureau is keyed on it."

  const revenue = numeric(f.revenue)
  if (revenue === null) errors.revenue = "Enter a number."
  else if (revenue < REVENUE_MIN) errors.revenue = `Must be at least ${REVENUE_MIN}.`

  const dti = numeric(f.dti_n)
  if (dti === null) errors.dti_n = "Enter a number."
  else if (dti < 0) errors.dti_n = "DTI is never negative."

  const loan = numeric(f.loan_amnt)
  if (loan === null) errors.loan_amnt = "Enter a number."
  else if (loan < LOAN_MIN || loan > LOAN_MAX)
    errors.loan_amnt = `Must be between ${LOAN_MIN} and ${LOAN_MAX}.`

  return errors
}

interface Props {
  onSubmit: (payload: ScoreRequest) => void
  pending: boolean
}

export function ApplicationForm({ onSubmit, pending }: Props) {
  const [form, setForm] = React.useState<FormState>(INITIAL)
  const [touched, setTouched] = React.useState(false)
  const [showAdvanced, setShowAdvanced] = React.useState(false)

  const errors = validate(form)
  const valid = Object.keys(errors).length === 0

  const set = <K extends keyof FormState>(key: K, value: string) =>
    setForm((f) => ({ ...f, [key]: value }))

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    setTouched(true)
    if (!valid) return

    onSubmit({
      applicant_id: form.applicant_id.trim(),
      // Non-null asserted only because `valid` already proved each parses.
      revenue: numeric(form.revenue)!,
      dti_n: numeric(form.dti_n)!,
      loan_amnt: numeric(form.loan_amnt)!,
      // "Not disclosed" -> JSON null. The backend normalizes null to "NI",
      // which is where the model has 453,804 rows of support. This is a real
      // answer, not a missing one.
      emp_length: form.emp_length === NOT_DISCLOSED ? null : form.emp_length,
      purpose: form.purpose,
      home_ownership_n: form.home_ownership_n,
    })
  }

  const err = (k: keyof FormState) => (touched ? errors[k] : undefined)

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Loan application</CardTitle>
        <span className="text-[11px] text-faint">6 applicant-reported fields</span>
      </CardHeader>

      <CardContent>
        <form onSubmit={submit} className="space-y-5">
          {/* ---- applicant_id: the bureau key, not a model feature ---- */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="applicant_id">Applicant</Label>
              <button
                type="button"
                onClick={() => setShowAdvanced((s) => !s)}
                className="text-[11px] text-faint underline-offset-2 hover:text-muted hover:underline"
              >
                {showAdvanced ? "hide id" : "enter id manually"}
              </button>
            </div>

            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => set("applicant_id", randomApplicantId())}
                className="shrink-0"
              >
                <Shuffle className="h-3.5 w-3.5" />
                Generate applicant
              </Button>
              {showAdvanced ? (
                <Input
                  id="applicant_id"
                  value={form.applicant_id}
                  onChange={(e) => set("applicant_id", e.target.value)}
                  className="tnum"
                  placeholder="applicant-000000"
                />
              ) : (
                <div className="flex h-9 flex-1 items-center rounded-md border border-border bg-surface-2 px-3">
                  <span className="tnum text-sm text-muted">{form.applicant_id}</span>
                </div>
              )}
            </div>
            <FieldError msg={err("applicant_id")} />
            <p className="text-[11px] leading-relaxed text-faint">
              The bureau is deterministic — the <em>same</em> applicant id always returns
              the same credit report, so a demo can be reproduced exactly.
            </p>
          </div>

          <Divider />

          {/* ---- the three numerics ---- */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Field label="Annual revenue" htmlFor="revenue" error={err("revenue")}>
              <Input
                id="revenue"
                type="number"
                inputMode="decimal"
                value={form.revenue}
                onChange={(e) => set("revenue", e.target.value)}
              />
            </Field>

            <Field label="DTI" htmlFor="dti_n" error={err("dti_n")} hint="debt-to-income">
              <Input
                id="dti_n"
                type="number"
                inputMode="decimal"
                step="0.01"
                value={form.dti_n}
                onChange={(e) => set("dti_n", e.target.value)}
              />
            </Field>

            <Field label="Loan amount" htmlFor="loan_amnt" error={err("loan_amnt")}>
              <Input
                id="loan_amnt"
                type="number"
                inputMode="decimal"
                value={form.loan_amnt}
                onChange={(e) => set("loan_amnt", e.target.value)}
              />
            </Field>
          </div>

          {/* ---- the three closed enums ---- */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Field label="Employment length" htmlFor="emp_length">
              <Select
                value={form.emp_length}
                onValueChange={(v) => set("emp_length", v)}
              >
                <SelectTrigger id="emp_length">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* EMP_ORDER is ordinal -- rendered in the order the model's
                      encoding means, never sorted alphabetically. */}
                  {EMP_ORDER.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                  <SelectItem value={NOT_DISCLOSED}>
                    <span className="text-muted">Not disclosed</span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </Field>

            <Field label="Purpose" htmlFor="purpose">
              <Select value={form.purpose} onValueChange={(v) => set("purpose", v)}>
                <SelectTrigger id="purpose">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {VALID_PURPOSE.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            <Field label="Home ownership" htmlFor="home_ownership_n">
              <Select
                value={form.home_ownership_n}
                onValueChange={(v) => set("home_ownership_n", v)}
              >
                <SelectTrigger id="home_ownership_n">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {VALID_HOME_OWNERSHIP.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          </div>

          <Divider />

          {/* ---- the point of the whole form ---- */}
          <div className="rounded-md border border-border bg-surface-2/50 px-3 py-2.5">
            <p className="text-[12px] leading-relaxed text-muted">
              <span className="text-fg">FICO is not entered here.</span> It is fetched from
              the credit bureau at scoring time and returned with the decision — the lender
              pulls the score, the applicant does not claim it.
            </p>
          </div>

          <Button type="submit" disabled={pending} className="w-full">
            {pending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Pulling credit report and scoring…
              </>
            ) : (
              "Score applicant"
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------

function Label({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="text-xs font-medium text-muted">
      {children}
    </label>
  )
}

function Field({
  label,
  htmlFor,
  hint,
  error,
  children,
}: {
  label: string
  htmlFor: string
  hint?: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-1.5">
        <Label htmlFor={htmlFor}>{label}</Label>
        {hint && <span className="text-[10px] text-faint">{hint}</span>}
      </div>
      {children}
      <FieldError msg={error} />
    </div>
  )
}

function FieldError({ msg }: { msg?: string }) {
  if (!msg) return null
  return <p className="text-[11px] text-reject">{msg}</p>
}

function Divider() {
  return <div className="h-px bg-border" />
}
