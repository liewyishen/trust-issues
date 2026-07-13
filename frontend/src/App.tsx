import * as React from "react"

import { ApplicationForm } from "@/components/ApplicationForm"
import { DecisionResult } from "@/components/DecisionResult"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  AdditivityError,
  API_BASE,
  NetworkError,
  ServiceUnavailableError,
  ValidationError,
  scoreApplicant,
  type ScoreRequest,
  type ScoreResponse,
} from "@/lib/api"

type State =
  | { status: "idle" }
  | { status: "pending" }
  | { status: "scored"; response: ScoreResponse }
  | { status: "failed"; error: unknown }

export default function App() {
  const [state, setState] = React.useState<State>({ status: "idle" })

  const submit = async (payload: ScoreRequest) => {
    setState({ status: "pending" })
    try {
      setState({ status: "scored", response: await scoreApplicant(payload) })
    } catch (error) {
      setState({ status: "failed", error })
    }
  }

  return (
    <div className="mx-auto min-h-full max-w-3xl px-6 py-10">
      <header className="mb-8">
        <div className="flex items-baseline justify-between">
          <h1 className="text-lg font-semibold tracking-tight text-fg">trust-issues</h1>
          <span className="tnum text-[11px] text-faint">{API_BASE}</span>
        </div>
        <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-muted">
          A credit-default risk model whose decisions can be checked. The form below scores
          against the live model — the credit score is pulled by the system, and every
          number on the result is returned by the API, not computed here.
        </p>
      </header>

      <main className="space-y-4">
        <ApplicationForm onSubmit={submit} pending={state.status === "pending"} />

        {state.status === "failed" && <Failure error={state.error} />}
        {state.status === "scored" && <DecisionResult r={state.response} />}
      </main>

      <footer className="mt-10 border-t border-border pt-4 text-[11px] leading-relaxed text-faint">
        The bureau is a deterministic mock (<code>MockBureau</code>) — it never calls a real
        vendor. Everything else on this page is the real model, the real calibrator and the
        real decision threshold.
      </footer>
    </div>
  )
}

/**
 * 422 and 503 are documented states of this API (serving/errors.py), not
 * generic failures, so each is rendered as itself rather than as "something
 * went wrong".
 */
function Failure({ error }: { error: unknown }) {
  if (error instanceof ValidationError) {
    return (
      <Alert variant="invalid">
        <AlertTitle>422 — the request was refused by the validation contract</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>
            The applicant did nothing wrong; the request carried something the API does not
            accept. It rejects rather than silently coercing — a value it had to guess at
            would be a decision made on data nobody chose.
          </p>
          <ul className="space-y-1.5">
            {error.issues.map((issue, i) => (
              <li
                key={i}
                className="rounded border border-border bg-surface px-2.5 py-1.5"
              >
                <span className="tnum text-[11px] text-reject">
                  {issue.loc.filter((p) => p !== "body").join(".") || "request"}
                </span>
                <span className="ml-2 text-[12px] text-muted">{issue.msg}</span>
              </li>
            ))}
          </ul>
        </AlertDescription>
      </Alert>
    )
  }

  if (error instanceof ServiceUnavailableError) {
    return (
      <Alert variant="unavailable">
        <AlertTitle>503 — the service is not ready</AlertTitle>
        <AlertDescription>
          {error.detail} The API refuses to serve rather than score with artifacts it could
          not verify at startup. No decision was made.
        </AlertDescription>
      </Alert>
    )
  }

  if (error instanceof AdditivityError) {
    return (
      <Alert variant="invalid">
        <AlertTitle>500 — the explanation failed its consistency check</AlertTitle>
        <AlertDescription>
          {error.detail} The contributions did not reconstruct the model's own score, so the
          service refused to return a decision at all. A decision without a valid
          explanation is worse than no decision.
        </AlertDescription>
      </Alert>
    )
  }

  if (error instanceof NetworkError) {
    return (
      <Alert variant="unavailable">
        <AlertTitle>The API is unreachable</AlertTitle>
        <AlertDescription>
          {error.message} Start it with{" "}
          <code className="text-muted">uv run uvicorn serving.app:app --reload</code>.
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Alert variant="unavailable">
      <AlertTitle>Unexpected failure</AlertTitle>
      <AlertDescription>{String(error)}</AlertDescription>
    </Alert>
  )
}
