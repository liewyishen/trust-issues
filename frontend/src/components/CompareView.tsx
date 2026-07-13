import { ArrowRight, CopyIcon, Loader2 } from "lucide-react"
import * as React from "react"

import { ApplicationForm, NoFicoNote } from "@/components/ApplicationForm"
import { CalibratorExplainer } from "@/components/CalibratorExplainer"
import { ComparisonSummary } from "@/components/ComparisonSummary"
import {
  CalibratorUnavailable,
  CreditReportTier,
  DecisionTier,
  TechnicalTier,
} from "@/components/DecisionResult"
import { Failure } from "@/components/Failure"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { scoreApplicant, type CalibratorResponse, type ScoreResponse } from "@/lib/api"
import { toPlotPoint } from "@/lib/calibrator"
import { LOAN_MAX, LOAN_MIN, REVENUE_MIN, VALID_HOME_OWNERSHIP } from "@/lib/enums.generated"
import {
  INITIAL,
  differingFields,
  isValid,
  toRequest,
  validate,
  type FormState,
} from "@/lib/form"

/*
 * CETERIS PARIBUS, AS A TOOL.
 *
 * Two applicants, scored against the real /score, side by side. The move it
 * exists to make cheap: copy A into B, change exactly ONE field, and read off
 * what that field did. Doing this by hand -- edit, re-score, remember the old
 * numbers, eyeball the delta -- is where the four-home_ownership collapse was
 * found in the first place. This makes it one click instead of four rounds of
 * squinting.
 *
 * Both forms are the SAME ApplicationForm component with different state. There
 * is no second copy of the field list, the coercion rules, or the enum wiring:
 * those live in lib/form.ts and lib/enums.generated.ts, and enums.generated.ts is
 * written by scripts/generate-enums.py out of the real Python the API validates
 * against. A forked form would be a second source of truth, and the second one is
 * the one that goes stale.
 *
 * Both applicants are plotted on ONE calibrator chart, because the headline
 * finding is positional: when A and B sit on the same flat block, their identical
 * probability is not a coincidence to be asserted -- it is a place you can see
 * them both standing.
 */

type Pair = { a: ScoreResponse; b: ScoreResponse; formA: FormState; formB: FormState }

type State =
  | { status: "idle" }
  | { status: "pending" }
  | { status: "scored"; pair: Pair }
  | { status: "failed"; error: unknown; side: "A" | "B" }

const BOUNDS = { REVENUE_MIN, LOAN_MIN, LOAN_MAX }

/**
 * B starts as A with ONE field already changed, so the tool opens on the exact
 * comparison it was built for rather than on two identical applicants.
 *
 * The changed value is DERIVED from the generated enum -- the first home_ownership
 * that is not A's -- rather than written in as a literal. A literal seed would be
 * a third place the category set is spelled out, and if the set were ever
 * regenerated without it, this view would open with a Select whose value is not
 * among its own options: silently blank, and blamed on the dropdown.
 */
const INITIAL_B: FormState = {
  ...INITIAL,
  home_ownership_n:
    VALID_HOME_OWNERSHIP.find((v) => v !== INITIAL.home_ownership_n) ??
    INITIAL.home_ownership_n,
}

export function CompareView({
  cal,
  calError,
}: {
  cal: CalibratorResponse | null
  calError: unknown
}) {
  const [formA, setFormA] = React.useState<FormState>(INITIAL)
  const [formB, setFormB] = React.useState<FormState>(INITIAL_B)
  const [touched, setTouched] = React.useState(false)
  const [state, setState] = React.useState<State>({ status: "idle" })

  const errorsA = validate(formA, BOUNDS)
  const errorsB = validate(formB, BOUNDS)
  const ready = isValid(errorsA) && isValid(errorsB)

  // Highlighted live, as the user types -- the changed levers are visible BEFORE
  // scoring, so it is obvious what the comparison is about to isolate.
  const changed = differingFields(formA, formB)

  const scoreBoth = async () => {
    setTouched(true)
    if (!ready) return
    setState({ status: "pending" })

    // Two real calls to the real endpoint. Nothing about B is derived from A's
    // response -- if it were, the comparison would be comparing the model against
    // this component's idea of the model.
    let a: ScoreResponse
    try {
      a = await scoreApplicant(toRequest(formA))
    } catch (error) {
      setState({ status: "failed", error, side: "A" })
      return
    }
    let b: ScoreResponse
    try {
      b = await scoreApplicant(toRequest(formB))
    } catch (error) {
      setState({ status: "failed", error, side: "B" })
      return
    }
    setState({ status: "scored", pair: { a, b, formA, formB } })
  }

  const pending = state.status === "pending"

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ApplicationForm
          idPrefix="a"
          title="Applicant A"
          value={formA}
          onChange={setFormA}
          errors={errorsA}
          showErrors={touched}
          disabled={pending}
          highlight={changed}
          headerRight={<Badge>A</Badge>}
        />
        <ApplicationForm
          idPrefix="b"
          title="Applicant B"
          value={formB}
          onChange={setFormB}
          errors={errorsB}
          showErrors={touched}
          disabled={pending}
          highlight={changed}
          headerRight={
            <Button
              type="button"
              variant="outline"
              onClick={() => setFormB(formA)}
              disabled={pending}
              title="Make B identical to A, then change exactly one field"
            >
              <CopyIcon className="h-3.5 w-3.5" />
              copy A <ArrowRight className="h-3 w-3" /> B
            </Button>
          }
        />
      </div>

      <NoFicoNote />

      <div className="space-y-2">
        <Button onClick={scoreBoth} disabled={pending} className="w-full">
          {pending ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Pulling two credit reports and scoring…
            </>
          ) : (
            "Score A and B"
          )}
        </Button>
        <p className="text-center text-[11px] leading-relaxed text-faint">
          {changed.length === 0 ? (
            <>
              A and B are currently <span className="text-muted">identical</span> — score them and
              you will get the same answer twice. Change one field in B to isolate its effect.
            </>
          ) : changed.length === 1 ? (
            <>
              One field differs. This is a{" "}
              <span className="text-muted">ceteris-paribus comparison</span>: whatever moves is
              attributable to that field.
            </>
          ) : (
            <>
              <span className="text-adverse">{changed.length} fields differ.</span> Whatever moves
              is attributable to <em>some combination</em> of them, and this view cannot say which
              — use “copy A → B” and change one at a time to isolate a lever.
            </>
          )}
        </p>
      </div>

      {state.status === "failed" && (
        <div className="space-y-2">
          <p className="text-[11px] text-faint">
            Applicant {state.side} was refused, so there is nothing to compare. Neither applicant
            is shown — a comparison against a missing side would be a comparison against nothing.
          </p>
          <Failure error={state.error} />
        </div>
      )}

      {state.status === "scored" && (
        <Comparison pair={state.pair} cal={cal} calError={calError} />
      )}
    </div>
  )
}

// ===========================================================================

function Comparison({
  pair,
  cal,
  calError,
}: {
  pair: Pair
  cal: CalibratorResponse | null
  calError: unknown
}) {
  const { a, b, formA, formB } = pair

  return (
    <div className="space-y-4">
      {/* The diff first -- it is the reason this view exists. The two full result
          panels are below it, for a reader who wants each applicant whole. */}
      <ComparisonSummary a={a} b={b} formA={formA} formB={formB} cal={cal} />

      {/* ONE chart, BOTH applicants. When they share a block you see them
          standing on it together; the summary above only asserts it. */}
      {cal ? (
        <CalibratorExplainer
          points={[toPlotPoint(a, "A", "A"), toPlotPoint(b, "B", "B")]}
          cal={cal}
        />
      ) : calError ? (
        <CalibratorUnavailable error={calError} />
      ) : null}

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
        <Side label="A" r={a} cal={cal} />
        <Side label="B" r={b} cal={cal} />
      </div>
    </div>
  )
}

/** One applicant, whole -- the same tiers the single-applicant flow renders,
 *  reused rather than reimplemented. The calibrator tier is NOT repeated here:
 *  there is one calibrator, and it is drawn once, above, with both points on it. */
function Side({
  label,
  r,
  cal,
}: {
  label: string
  r: ScoreResponse
  cal: CalibratorResponse | null
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Badge>{label}</Badge>
        <span className="tnum text-[11px] text-faint">{r.credit_report.bureau}</span>
      </div>
      <DecisionTier r={r} cal={cal} />
      <CreditReportTier report={r.credit_report} />
      <TechnicalTier r={r} />
    </div>
  )
}
