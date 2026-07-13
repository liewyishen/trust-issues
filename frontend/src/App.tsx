import { Loader2 } from "lucide-react"
import * as React from "react"

import { ApplicationForm, NoFicoNote } from "@/components/ApplicationForm"
import { CompareView } from "@/components/CompareView"
import { DecisionResult } from "@/components/DecisionResult"
import { DriftMonitor } from "@/components/DriftMonitor"
import { Failure } from "@/components/Failure"
import { Button } from "@/components/ui/button"
import {
  API_BASE,
  getCalibrator,
  scoreApplicant,
  type CalibratorResponse,
  type ScoreRequest,
  type ScoreResponse,
} from "@/lib/api"
import { LOAN_MAX, LOAN_MIN, REVENUE_MIN } from "@/lib/enums.generated"
import { INITIAL, isValid, toRequest, validate, type FormState } from "@/lib/form"
import { cn } from "@/lib/utils"

/**
 * Three questions, and they are genuinely different questions -- which is why they
 * are three modes and not three panels on one page.
 *
 *   single  / compare -- about an APPLICANT. What was decided, and why.
 *   drift             -- about a POPULATION. Has the market moved under the model,
 *                        and did the monitor notice?
 *
 * The drift tab shares no state with the other two, and needs none: it does not
 * score anybody. It talks to a different endpoint, about a different subject, on a
 * different time scale.
 */
type Mode = "single" | "compare" | "drift"

type State =
  | { status: "idle" }
  | { status: "pending" }
  | { status: "scored"; response: ScoreResponse }
  | { status: "failed"; error: unknown }

const BOUNDS = { REVENUE_MIN, LOAN_MIN, LOAN_MAX }

export default function App() {
  const [mode, setMode] = React.useState<Mode>("single")

  // The calibrator is the same object for every applicant -- it is the SHIPPED
  // artifact, not a per-request computation -- so it is fetched once, on mount,
  // and reused by both modes. It is deliberately NOT bundled into ScoreResponse:
  // /score answers "what was decided about this applicant", /calibrator answers
  // "what is the function that decided". Fetching it separately keeps that
  // boundary visible.
  //
  // Its failure is kept separate from a scoring failure. A dead /calibrator must
  // not take down a decision that /score already returned successfully.
  const [cal, setCal] = React.useState<CalibratorResponse | null>(null)
  const [calError, setCalError] = React.useState<unknown>(null)

  React.useEffect(() => {
    getCalibrator().then(setCal, setCalError)
  }, [])

  return (
    <div
      className={cn(
        "mx-auto min-h-full px-6 py-10 transition-[max-width]",
        mode === "single" ? "max-w-3xl" : mode === "drift" ? "max-w-5xl" : "max-w-6xl",
      )}
    >
      <header className="mb-6">
        <div className="flex items-baseline justify-between">
          <h1 className="text-lg font-semibold tracking-tight text-fg">trust-issues</h1>
          <span className="tnum text-[11px] text-faint">{API_BASE}</span>
        </div>
        <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-muted">
          A credit-default risk model whose decisions can be checked — and whose monitoring can
          be watched working. Everything on these pages is returned by the live API: the model,
          the calibrator, the decision threshold and the drift monitor are all the real ones, and
          none of their numbers are computed in this browser.
        </p>
      </header>

      <nav className="mb-4 flex gap-1 rounded-md border border-border bg-surface p-1">
        <Tab active={mode === "single"} onClick={() => setMode("single")}>
          Score one applicant
        </Tab>
        <Tab active={mode === "compare"} onClick={() => setMode("compare")}>
          Compare two
        </Tab>
        <Tab active={mode === "drift"} onClick={() => setMode("drift")}>
          Monitor drift
        </Tab>
      </nav>

      <main>
        {mode === "single" && <SingleView cal={cal} calError={calError} />}
        {mode === "compare" && <CompareView cal={cal} calError={calError} />}
        {/* Mounted only while its tab is open, so the ~1.3s warm-up call is paid
            by the person who asked for it rather than by everyone on page load. */}
        {mode === "drift" && <DriftMonitor />}
      </main>

      <footer className="mt-10 border-t border-border pt-4 text-[11px] leading-relaxed text-faint">
        The bureau is a deterministic mock (<code>MockBureau</code>) — it never calls a real
        vendor. Everything else on this page is the real model, the real calibrator and the real
        decision threshold.
      </footer>
    </div>
  )
}

// ===========================================================================

function SingleView({
  cal,
  calError,
}: {
  cal: CalibratorResponse | null
  calError: unknown
}) {
  const [form, setForm] = React.useState<FormState>(INITIAL)
  const [touched, setTouched] = React.useState(false)
  const [state, setState] = React.useState<State>({ status: "idle" })

  const errors = validate(form, BOUNDS)
  const ready = isValid(errors)
  const pending = state.status === "pending"

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setTouched(true)
    if (!ready) return

    const payload: ScoreRequest = toRequest(form)
    setState({ status: "pending" })
    try {
      setState({ status: "scored", response: await scoreApplicant(payload) })
    } catch (error) {
      setState({ status: "failed", error })
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <ApplicationForm
        idPrefix="one"
        title="Loan application"
        value={form}
        onChange={setForm}
        errors={errors}
        showErrors={touched}
        disabled={pending}
        footer={
          <div className="space-y-4">
            <NoFicoNote />
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
          </div>
        }
      />

      {state.status === "failed" && <Failure error={state.error} />}
      {state.status === "scored" && (
        <DecisionResult r={state.response} cal={cal} calError={calError} />
      )}
    </form>
  )
}

function Tab({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex-1 rounded px-3 py-1.5 text-[12px] font-medium transition-colors",
        active
          ? "bg-surface-2 text-fg"
          : "text-faint hover:text-muted",
      )}
    >
      {children}
    </button>
  )
}
