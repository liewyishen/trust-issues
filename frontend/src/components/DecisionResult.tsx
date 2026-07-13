import { Check, ChevronRight } from "lucide-react"
import * as React from "react"

import { CalibratorExplainer } from "@/components/CalibratorExplainer"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import type { CalibratorResponse, ScoreResponse } from "@/lib/api"
import { cn } from "@/lib/utils"

/*
 * Four tiers, in the order a reader needs them.
 *
 *   1  the pull the system made      -- what the decision was made ON
 *   2  the decision                  -- what was decided, and the named reasons
 *   3  the calibrator                -- WHY that probability and not a nearby one
 *   4  the additive explanation      -- collapsed; proof the tiers above cohere
 *
 * Tier 3 sits where it does deliberately. Tier 2 shows a probability; tier 4
 * shows contributions in log-odds and declines to convert them into probability
 * points. A reader who reads those in sequence is owed the reason in between, and
 * the reason is a picture: the calibrator is a 52-level step function, its slope
 * is zero almost everywhere, and there is therefore nothing to convert.
 *
 * Every number rendered here comes off the API response. Nothing is computed in
 * the browser except (a) the additivity SUM in tier 4 and (b) the exploratory
 * probe in tier 3 -- and both exist precisely so a reader can check the backend
 * rather than trust it. The applicant's own figures are never recomputed.
 */

export function DecisionResult({
  r,
  cal,
  calError,
}: {
  r: ScoreResponse
  cal: CalibratorResponse | null
  calError: unknown
}) {
  return (
    <div className="space-y-4">
      <CreditReportTier report={r.credit_report} />
      <DecisionTier r={r} cal={cal} />
      {cal ? (
        <CalibratorExplainer r={r} cal={cal} />
      ) : calError ? (
        <CalibratorUnavailable error={calError} />
      ) : null}
      <TechnicalTier r={r} />
    </div>
  )
}

/**
 * The decision above stands -- /score returned it. Only the PICTURE of the
 * calibrator is missing, and saying so beats quietly dropping the panel, which
 * would leave a reader believing they had seen the whole page.
 */
function CalibratorUnavailable({ error }: { error: unknown }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="text-xs font-medium text-fg">
          The calibrator could not be fetched.
        </div>
        <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
          <code className="text-faint">GET /calibrator</code> failed, so the step function
          behind the probability above is not drawn. The decision itself is unaffected —{" "}
          <code className="text-faint">/score</code> returned it, and it was composed with the
          calibrator whether or not this panel can show you its shape.{" "}
          <span className="text-faint">{String(error)}</span>
        </p>
      </CardContent>
    </Card>
  )
}

// ===========================================================================
// TIER 1 -- the credit report the SYSTEM fetched. The applicant never touched
// this number, and that is the entire point of the panel.
// ===========================================================================
function CreditReportTier({ report }: { report: ScoreResponse["credit_report"] }) {
  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Credit report · fetched by the system</CardTitle>
        <Badge>bureau pull</Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-end justify-between gap-6">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-faint">
              FICO score
            </div>
            <div className="tnum mt-1 text-5xl font-semibold leading-none text-fg">
              {report.fico_n.toFixed(2)}
            </div>
          </div>
          <p className="max-w-[16rem] text-right text-[11px] leading-relaxed text-faint">
            Pulled from the bureau at scoring time, keyed on the applicant id.
            <span className="block text-muted">Not self-reported.</span>
          </p>
        </div>

        {/* Provenance: quiet, small, an audit footnote -- it identifies WHICH
            pull produced the number above. */}
        <dl className="grid grid-cols-3 gap-3 border-t border-border pt-3 text-[11px]">
          <Provenance label="bureau" value={report.bureau} />
          <Provenance label="fico_version" value={report.fico_version} />
          <Provenance
            label="pulled_at"
            value={new Date(report.pulled_at).toISOString().replace(".000Z", "Z")}
          />
        </dl>
      </CardContent>
    </Card>
  )
}

function Provenance({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-faint">{label}</dt>
      <dd className="tnum mt-0.5 truncate text-muted" title={value}>
        {value}
      </dd>
    </div>
  )
}

// ===========================================================================
// TIER 2 -- the decision, and the principal adverse factors behind it.
// ===========================================================================
function DecisionTier({ r, cal }: { r: ScoreResponse; cal: CalibratorResponse | null }) {
  const approved = r.decision === "APPROVE"

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Decision</CardTitle>
        <Badge variant={approved ? "approve" : "reject"}>{r.decision}</Badge>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="flex items-end justify-between gap-6">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-faint">
              Calibrated default probability
            </div>
            <div
              className={cn(
                "tnum mt-1 text-5xl font-semibold leading-none",
                approved ? "text-approve" : "text-reject",
              )}
            >
              {(r.p_calibrated * 100).toFixed(2)}
              <span className="text-2xl text-muted">%</span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-wider text-faint">
              Decision threshold
            </div>
            <div className="tnum mt-1 text-lg text-muted">
              {(r.threshold * 100).toFixed(2)}%
            </div>
            <div className="mt-0.5 text-[11px] text-faint">
              {approved ? "below → approve" : "at or above → reject"}
            </div>
          </div>
        </div>

        {/* The honesty line. The probability above is not a point on a smooth
            curve; it is one of a few dozen levels. Saying so where the number is
            shown is the difference between a calibrated model and a model that
            merely outputs a decimal.

            The LEVEL COUNT is read from the live calibrator, never written into
            this sentence. A hardcoded "52" would keep rendering after a retrain
            that changed it, and a sentence that keeps its wording while the
            artifact moves underneath it is exactly the failure this repo exists
            to prevent. With /calibrator unreachable the claim is still true, so
            it is still made -- just without a number nobody can vouch for. */}
        <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2 text-[11px] leading-relaxed text-muted">
          This probability comes from an{" "}
          <span className="text-fg">
            {cal ? `${cal.n_distinct_y}-level ` : ""}isotonic calibrator
          </span>{" "}
          — it is discrete, not continuous.{" "}
          {cal
            ? `Only ${cal.n_distinct_y} distinct values are reachable across the whole score range, so nearby applicants routinely land on the identical probability.`
            : "Only a fixed set of values is reachable across the whole score range, so nearby applicants routinely land on the identical probability."}{" "}
          {cal && <span className="text-faint">The step function is drawn below.</span>}
        </p>

        <ReasonCodes codes={r.reason_codes} />
      </CardContent>
    </Card>
  )
}

function ReasonCodes({ codes }: { codes: ScoreResponse["reason_codes"] }) {
  if (codes.length === 0) {
    // A real, documented state -- not an error, and not an empty list to hide.
    return (
      <div className="rounded-md border border-border-strong bg-surface-2 px-4 py-3">
        <div className="text-xs font-medium text-fg">No principal reasons could be named.</div>
        <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
          Every feature contribution was zero or risk-<em>decreasing</em>: the model's base
          value alone cleared the decision boundary. An adverse-action notice cannot list
          principal reasons when there are none, so per{" "}
          <code className="text-faint">design.md §6</code> this applicant routes to human
          review regardless of the probability above. The API returns the fact, not the
          routing policy.
        </p>
      </div>
    )
  }

  const max = Math.max(...codes.map((c) => Math.abs(c.contribution_log_odds)))

  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline justify-between">
        <div className="text-[11px] uppercase tracking-wider text-faint">
          Principal adverse factors
        </div>
        <div className="text-[10px] text-faint">
          contribution in <span className="text-muted">log-odds</span>
        </div>
      </div>

      <div className="divide-y divide-border rounded-md border border-border">
        {codes.map((c) => (
          <div key={c.rank} className="flex items-center gap-3 px-3 py-2.5">
            <span className="tnum w-4 shrink-0 text-sm text-faint">{c.rank}</span>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm text-fg">{c.feature}</div>
              <div className="tnum truncate text-[11px] text-faint">{c.value}</div>
            </div>
            {/* The bar is a magnitude, drawn relative to the largest contribution
                in THIS response -- it is not a probability and is not scaled to
                one. */}
            <div className="hidden h-1 w-24 shrink-0 overflow-hidden rounded-full bg-surface-2 sm:block">
              <div
                className="h-full bg-adverse"
                style={{ width: `${(Math.abs(c.contribution_log_odds) / max) * 100}%` }}
              />
            </div>
            <span className="tnum w-20 shrink-0 text-right text-sm text-adverse">
              +{c.contribution_log_odds.toFixed(4)}
            </span>
          </div>
        ))}
      </div>

      <p className="text-[11px] leading-relaxed text-faint">
        These are the <span className="text-muted">risk-increasing</span> factors — the
        principal reasons an adverse-action notice would cite. They are listed even on an
        approval, because they answer “what pushed this applicant toward default,” not
        “why was this approved.” The scale is the model's raw log-odds margin.{" "}
        <span className="text-muted">
          They are deliberately never converted to probability points
        </span>{" "}
        — under a step calibrator that quantity is undefined, not merely uncomputed.
      </p>
    </div>
  )
}

// ===========================================================================
// TIER 3 -- collapsed by default. The proof that the explanation and the
// decision are the same object seen twice.
// ===========================================================================
function TechnicalTier({ r }: { r: ScoreResponse }) {
  const [open, setOpen] = React.useState(false)

  const contributions = Object.entries(r.contributions_log_odds).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
  )
  const sum = contributions.reduce((acc, [, v]) => acc + v, 0)
  const reconstructed = r.base_value_log_odds + sum
  // The backend's _assert_additivity uses ADDITIVITY_ATOL = 1e-9 and 500s rather
  // than return a response that fails it. So this check can only ever pass -- it
  // is here to be CHECKED, not to catch something. A reader who does not trust
  // the guard can do the arithmetic themselves, from numbers on this page.
  const balances = Math.abs(reconstructed - r.raw_margin_log_odds) < 1e-9

  return (
    <Card>
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-between px-5 py-3 text-left transition-colors hover:bg-surface-2/50">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-faint">
              Verifiable additive explanation
            </div>
            <div className="mt-0.5 text-[11px] text-muted">
              the quantity explained equals the quantity decided on
            </div>
          </div>
          <div className="flex items-center gap-2">
            {balances && (
              <span className="flex items-center gap-1 text-[11px] text-approve">
                <Check className="h-3 w-3" />
                balances
              </span>
            )}
            <ChevronRight
              className={cn(
                "h-4 w-4 text-faint transition-transform",
                open && "rotate-90",
              )}
            />
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="space-y-5 border-t border-border">
            {/* --- the calibration step, made visible --- */}
            <div>
              <SectionLabel>The calibration step</SectionLabel>
              <div className="mt-2 flex items-center gap-4">
                <Stat label="p_raw" sub="calibrator input" value={r.p_raw} />
                <ChevronRight className="h-4 w-4 shrink-0 text-faint" />
                <Stat
                  label="p_calibrated"
                  sub="calibrator output · decided on"
                  value={r.p_calibrated}
                  emphasis
                />
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-faint">
                <code>p_raw</code> is <code>sigmoid(margin)</code> — the booster's own
                output. It is <em>not</em> the decided quantity: the isotonic calibrator
                maps it to <code>p_calibrated</code>, and that is what the threshold is
                applied to.
              </p>
            </div>

            {/* --- the eight contributions --- */}
            <div>
              <SectionLabel>
                All contributions · log-odds · sorted by magnitude
              </SectionLabel>
              <div className="mt-2 divide-y divide-border rounded-md border border-border">
                {contributions.map(([feature, v]) => (
                  <div
                    key={feature}
                    className="flex items-center justify-between px-3 py-1.5"
                  >
                    <span className="text-[12px] text-muted">{feature}</span>
                    <span
                      className={cn(
                        "tnum text-[12px]",
                        v > 0 ? "text-adverse" : v < 0 ? "text-approve" : "text-faint",
                      )}
                    >
                      {v >= 0 ? "+" : ""}
                      {v.toFixed(6)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-faint">
                Positive pushes toward default; negative pushes away. Only the positive
                ones can be principal adverse factors.
              </p>
            </div>

            {/* --- the identity, written out --- */}
            <div>
              <SectionLabel>The additivity identity</SectionLabel>
              <div className="mt-2 space-y-1 rounded-md border border-border bg-surface-2/50 p-3 text-[12px]">
                <Row label="base_value_log_odds" value={r.base_value_log_odds} />
                <Row label="+ Σ contributions_log_odds" value={sum} />
                <div className="!my-2 h-px bg-border" />
                <Row label="= reconstructed margin" value={reconstructed} strong />
                <Row
                  label="raw_margin_log_odds (returned)"
                  value={r.raw_margin_log_odds}
                  strong
                />
              </div>

              <div
                className={cn(
                  "mt-2 flex items-center gap-2 text-[11px]",
                  balances ? "text-approve" : "text-reject",
                )}
              >
                {balances ? (
                  <>
                    <Check className="h-3.5 w-3.5" />
                    <span>
                      Identical to within 1e-9. The explanation reconstructs the score it
                      explains.
                    </span>
                  </>
                ) : (
                  <span>
                    These do not match — which should be unreachable: the API's additivity
                    guard returns a 500 instead of a decision when it fails.
                  </span>
                )}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-faint">
                This is the check the service runs before it will return anything at all
                (<code>_assert_additivity</code>, tolerance 1e-9). If the contributions did
                not sum back to the margin, the API would have returned a 500 rather than a
                decision — an explanation that does not reconstruct its own score is worse
                than no decision. The numbers above let you verify that yourself instead of
                taking its word.
              </p>
            </div>

            {/* --- artifact identity --- */}
            <div className="grid grid-cols-2 gap-3 border-t border-border pt-3 text-[11px]">
              <Provenance label="model_trained_at" value={r.model_trained_at ?? "—"} />
              <Provenance
                label="calibrator_trained_at"
                value={r.calibrator_trained_at ?? "—"}
              />
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] uppercase tracking-wider text-faint">{children}</div>
  )
}

function Stat({
  label,
  sub,
  value,
  emphasis,
}: {
  label: string
  sub: string
  value: number
  emphasis?: boolean
}) {
  return (
    <div className="flex-1 rounded-md border border-border bg-surface-2/50 px-3 py-2">
      <div className="text-[10px] text-faint">{label}</div>
      <div
        className={cn(
          "tnum mt-0.5 text-lg",
          emphasis ? "font-semibold text-fg" : "text-muted",
        )}
      >
        {value.toFixed(6)}
      </div>
      <div className="mt-0.5 text-[10px] text-faint">{sub}</div>
    </div>
  )
}

function Row({
  label,
  value,
  strong,
}: {
  label: string
  value: number
  strong?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <span className={cn(strong ? "text-fg" : "text-muted")}>{label}</span>
      <span className={cn("tnum", strong ? "text-fg" : "text-muted")}>
        {value >= 0 ? "+" : ""}
        {value.toFixed(9)}
      </span>
    </div>
  )
}
