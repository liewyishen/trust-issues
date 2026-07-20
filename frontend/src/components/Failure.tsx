import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  InternalServerError,
  NetworkError,
  RouteNotAvailableError,
  ServiceUnavailableError,
  ValidationError,
} from "@/lib/api"

/**
 * 422 and 503 are documented states of this API (serving/errors.py), not generic
 * failures, so each is rendered as itself rather than as "something went wrong".
 *
 * Shared by the single-applicant flow and the compare view. A second copy would
 * be a second place for these states to drift out of agreement with
 * serving/errors.py.
 */
export function Failure({ error }: { error: unknown }) {
  if (error instanceof ValidationError) {
    return (
      <Alert variant="invalid">
        <AlertTitle>422 — the request was refused by the validation contract</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>
            The applicant did nothing wrong; the request carried something the API does not
            accept. It rejects rather than silently coercing — a value it had to guess at would
            be a decision made on data nobody chose.
          </p>
          <ul className="space-y-1.5">
            {error.issues.map((issue, i) => (
              <li key={i} className="rounded border border-border bg-surface px-2.5 py-1.5">
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
          {error.detail} The API refuses to serve rather than score with artifacts it could not
          verify at startup. No decision was made.
        </AlertDescription>
      </Alert>
    )
  }

  if (error instanceof InternalServerError) {
    // Two shapes, and this component can tell them apart but cannot tell WHY
    // either happened. It used to name the additivity guard for both, on all
    // four routes that render this component -- so an unhandled failure in
    // /fairness reported that the contributions had not reconstructed a score.
    // The service's own sentence is shown when there is one, because it was
    // written at the raise and travels with it; when there is none, this says
    // so rather than supplying a cause it cannot prove.
    return (
      <Alert variant="invalid">
        <AlertTitle>500 — the service returned no result</AlertTitle>
        <AlertDescription>
          {error.detail ?? (
            <>
              The service failed and said nothing further. Nothing was returned, which is the
              intended posture: a result the service could not stand behind is worse than none.
            </>
          )}
        </AlertDescription>
      </Alert>
    )
  }

  if (error instanceof RouteNotAvailableError) {
    // The drift monitor renders its own, fuller version of this state (it knows
    // WHY /drift is dev-only). This is the generic fallback, so that a 404 from
    // any route reads as "not mounted here" rather than as a mystery.
    return (
      <Alert variant="unavailable">
        <AlertTitle>404 — that route is not mounted on this deployment</AlertTitle>
        <AlertDescription>
          {error.message} Not every route this client knows about is present in every deployment;
          the API said so plainly, and nothing has been drawn in its place.
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
