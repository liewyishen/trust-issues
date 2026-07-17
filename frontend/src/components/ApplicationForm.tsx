import { Shuffle } from "lucide-react"
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
import {
  EMP_ORDER,
  VALID_HOME_OWNERSHIP,
  VALID_PURPOSE,
} from "@/lib/enums.generated"
import {
  NOT_DISCLOSED,
  randomApplicantId,
  type FormState,
} from "@/lib/form"
import { cn } from "@/lib/utils"

/**
 * The six applicant-reported fields, plus applicant_id.
 *
 * CONTROLLED, and deliberately so: the compare view renders TWO of these, and a
 * "copy A -> B" button has to be able to write one form's state into the other.
 * That is impossible if the state lives in here. State, validation and the
 * mapping onto ScoreRequest therefore live in lib/form.ts -- one copy, two
 * instances. Forking a second form to get a second applicant would give the field
 * list, the coercion rules and the enum wiring a second home, and the second home
 * is the one that goes stale.
 *
 * There is NO FICO input, and its absence is the design, not an omission: the
 * service fetches fico_n from the credit bureau at scoring time
 * (CreditBureau.fetch, serving/bureau.py). ScoreRequest does not carry the field
 * and extra="forbid" turns a client-submitted one into a 422 -- an applicant who
 * could state their own FICO could describe an applicant whose score never came
 * from a bureau pull.
 *
 * The three closed-enum fields are Selects, never text inputs, and their options
 * come from enums.generated.ts -- read out of src/data_validation.py and
 * src/features.py by scripts/generate-enums.py, never hand-typed here.
 * serving/schema.py measures why the dropdown matters: an arbitrary unmapped
 * string for emp_length encodes IDENTICALLY to null and to "bogus" once
 * add_features() has run -- all three collapse to (emp_length_ord=NaN,
 * emp_length_missing=0), a combination occurring in 0 of Train's 453,804 rows. A
 * dropdown makes that state unreachable from the UI rather than merely rejected
 * after the fact.
 */

interface Props {
  value: FormState
  onChange: (next: FormState) => void
  errors: Partial<Record<keyof FormState, string>>
  /** Parent decides when errors become visible (on submit, not on first keypress). */
  showErrors: boolean
  disabled?: boolean
  title: string
  /** Prefixes every DOM id. Two forms on one page must not collide on `htmlFor`,
   *  or clicking B's label focuses A's input. */
  idPrefix: string
  /** Fields to mark as differing from the other applicant. Compare view only. */
  highlight?: Array<keyof FormState>
  /** Rendered in the card header, right-aligned (an A/B badge, a copy button). */
  headerRight?: React.ReactNode
  /** Rendered at the foot of the card (the submit button, in single mode). */
  footer?: React.ReactNode
}

export function ApplicationForm({
  value,
  onChange,
  errors,
  showErrors,
  disabled,
  title,
  idPrefix,
  highlight = [],
  headerRight,
  footer,
}: Props) {
  const [showId, setShowId] = React.useState(false)

  const set = <K extends keyof FormState>(key: K, v: string) =>
    onChange({ ...value, [key]: v })

  const err = (k: keyof FormState) => (showErrors ? errors[k] : undefined)
  const lit = (k: keyof FormState) => highlight.includes(k)
  const id = (k: string) => `${idPrefix}-${k}`

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>{title}</CardTitle>
        {headerRight ?? <span className="text-[11px] text-faint">6 applicant-reported fields</span>}
      </CardHeader>

      <CardContent>
        <fieldset disabled={disabled} className="space-y-5">
          {/* ---- applicant_id: the bureau key, not a model feature ---- */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor={id("applicant_id")} lit={lit("applicant_id")}>
                Applicant
              </Label>
              <button
                type="button"
                onClick={() => setShowId((s) => !s)}
                className="text-[11px] text-faint underline-offset-2 hover:text-muted hover:underline"
              >
                {showId ? "hide id" : "enter id manually"}
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
                Generate
              </Button>
              {showId ? (
                <Input
                  id={id("applicant_id")}
                  value={value.applicant_id}
                  onChange={(e) => set("applicant_id", e.target.value)}
                  className={cn("tnum", lit("applicant_id") && "border-adverse")}
                  placeholder="applicant-000000"
                />
              ) : (
                <div
                  className={cn(
                    "flex h-9 flex-1 items-center rounded-md border bg-surface-2 px-3",
                    lit("applicant_id") ? "border-adverse" : "border-border",
                  )}
                >
                  <span className="tnum truncate text-sm text-muted">{value.applicant_id}</span>
                </div>
              )}
            </div>
            <FieldError msg={err("applicant_id")} />
            {/* Ends at the credit report on purpose. This sentence used to add "so a
                demo can be reproduced exactly" -- a claim about the whole page, and one
                no single test holds: it is INFERRED from /score's determinism
                (tests/test_serving.py's test_same_applicant_id_scores_reproducibly_
                through_http), /drift's (test_the_same_knob_setting_returns_the_same_
                numbers), and /calibrator and /fairness being static reads. True, and
                borrowed. What is left is what tests/test_bureau.py's
                test_same_applicant_produces_same_report asserts, and nothing further. */}
            <p className="text-[11px] leading-relaxed text-faint">
              The bureau is deterministic — the <em>same</em> applicant id always returns the
              same credit report.
            </p>
          </div>

          <Divider />

          {/* ---- the three numerics ---- */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Field label="Annual revenue" htmlFor={id("revenue")} error={err("revenue")} lit={lit("revenue")}>
              <Input
                id={id("revenue")}
                type="number"
                inputMode="decimal"
                value={value.revenue}
                onChange={(e) => set("revenue", e.target.value)}
                className={cn(lit("revenue") && "border-adverse")}
              />
            </Field>

            <Field label="DTI" htmlFor={id("dti_n")} error={err("dti_n")} hint="debt-to-income" lit={lit("dti_n")}>
              <Input
                id={id("dti_n")}
                type="number"
                inputMode="decimal"
                step="0.01"
                value={value.dti_n}
                onChange={(e) => set("dti_n", e.target.value)}
                className={cn(lit("dti_n") && "border-adverse")}
              />
            </Field>

            <Field label="Loan amount" htmlFor={id("loan_amnt")} error={err("loan_amnt")} lit={lit("loan_amnt")}>
              <Input
                id={id("loan_amnt")}
                type="number"
                inputMode="decimal"
                value={value.loan_amnt}
                onChange={(e) => set("loan_amnt", e.target.value)}
                className={cn(lit("loan_amnt") && "border-adverse")}
              />
            </Field>
          </div>

          {/* ---- the three closed enums ---- */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Field label="Employment length" htmlFor={id("emp_length")} lit={lit("emp_length")}>
              <Select value={value.emp_length} onValueChange={(v) => set("emp_length", v)}>
                <SelectTrigger id={id("emp_length")} className={cn(lit("emp_length") && "border-adverse")}>
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

            <Field label="Purpose" htmlFor={id("purpose")} lit={lit("purpose")}>
              <Select value={value.purpose} onValueChange={(v) => set("purpose", v)}>
                <SelectTrigger id={id("purpose")} className={cn(lit("purpose") && "border-adverse")}>
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

            <Field label="Home ownership" htmlFor={id("home_ownership_n")} lit={lit("home_ownership_n")}>
              <Select
                value={value.home_ownership_n}
                onValueChange={(v) => set("home_ownership_n", v)}
              >
                <SelectTrigger
                  id={id("home_ownership_n")}
                  className={cn(lit("home_ownership_n") && "border-adverse")}
                >
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

          {footer}
        </fieldset>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------

function Label({
  children,
  htmlFor,
  lit,
}: {
  children: React.ReactNode
  htmlFor?: string
  lit?: boolean
}) {
  return (
    <label
      htmlFor={htmlFor}
      className={cn("text-xs font-medium", lit ? "text-adverse" : "text-muted")}
    >
      {children}
    </label>
  )
}

function Field({
  label,
  htmlFor,
  hint,
  error,
  lit,
  children,
}: {
  label: string
  htmlFor: string
  hint?: string
  error?: string
  lit?: boolean
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-1.5">
        <Label htmlFor={htmlFor} lit={lit}>
          {label}
        </Label>
        {hint && <span className="text-[10px] text-faint">{hint}</span>}
        {lit && <span className="text-[10px] text-adverse">changed</span>}
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

/** The point of the whole form, said out loud. Rendered by the caller so the
 *  compare view can show it once rather than twice. */
export function NoFicoNote() {
  return (
    <div className="rounded-md border border-border bg-surface-2/50 px-3 py-2.5">
      <p className="text-[12px] leading-relaxed text-muted">
        <span className="text-fg">FICO is not entered here.</span> It is fetched from the credit
        bureau at scoring time and returned with the decision — the lender pulls the score, the
        applicant does not claim it.
      </p>
    </div>
  )
}
