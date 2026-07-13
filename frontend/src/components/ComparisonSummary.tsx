import { ArrowRight, Minus, Plus, TriangleAlert } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { CalibratorResponse, ScoreResponse } from "@/lib/api"
import { evaluate } from "@/lib/calibrator"
import {
  BUREAU_KEY_FIELD,
  FIELD_LABELS,
  displayValue,
  differingFields,
  featuresTouchedBy,
  type FormState,
} from "@/lib/form"
import { cn } from "@/lib/utils"

/*
 * WHAT CHANGING ONE FIELD ACTUALLY DID.
 *
 * The whole panel is in LOG-ODDS. There is no probability-point delta here and
 * there will not be one: under the shipped step calibrator the derivative of
 * p_cal with respect to the model's own output is exactly 0 across the flat
 * blocks, so "this change was worth N points of probability" is undefined, not
 * merely uncomputed (docs/explainability.md). Rendering such a number would be
 * inventing one.
 *
 * That refusal is the point rather than a limitation, and this panel is where it
 * becomes legible: when A and B land on the same calibrator block, the log-odds
 * MOVED and the probability did NOT. The contribution table shows exactly which
 * feature moved and by how much; the step row shows that none of it reached the
 * decision. Both facts are true at once, and a UI that only showed the second
 * would look like the model ignored the change.
 */

const pct = (v: number) => `${(v * 100).toFixed(2)}%`
const signed = (v: number, dp = 6) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`

export function ComparisonSummary({
  a,
  b,
  formA,
  formB,
  cal,
}: {
  a: ScoreResponse
  b: ScoreResponse
  formA: FormState
  formB: FormState
  cal: CalibratorResponse | null
}) {
  const changed = differingFields(formA, formB)

  const blockA = cal ? evaluate(cal, a.p_raw).block : null
  const blockB = cal ? evaluate(cal, b.p_raw).block : null
  const sameStep = cal !== null && blockA !== null && blockA === blockB

  const dMargin = b.raw_margin_log_odds - a.raw_margin_log_odds
  const dPCal = b.p_calibrated - a.p_calibrated

  // The lever that is not a lever: applicant_id is the bureau's key, not a model
  // feature. Change it and fico_n changes with it -- so the comparison stops
  // being ceteris paribus, and pinning the outcome on the other edited field
  // would be wrong. Say so rather than let a reader draw it.
  const bureauAlsoMoved = changed.includes(BUREAU_KEY_FIELD)

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>What the change did</CardTitle>
        <span className="text-[11px] text-faint">everything in log-odds</span>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* ---- 1. the levers ---- */}
        <section>
          <SectionLabel>Inputs that differ</SectionLabel>
          {changed.length === 0 ? (
            <p className="mt-2 rounded-md border border-border bg-surface-2/50 px-3 py-2 text-[11px] leading-relaxed text-muted">
              <span className="text-fg">A and B are identical.</span> Same inputs, same bureau
              pull, same score — the panels below agree because they are the same applicant
              submitted twice. Change a field in B to see what it moves.
            </p>
          ) : (
            <div className="mt-2 divide-y divide-border rounded-md border border-border">
              {changed.map((k) => (
                <div key={k} className="flex items-center gap-3 px-3 py-2 text-[12px]">
                  <span className="w-36 shrink-0 text-adverse">{FIELD_LABELS[k]}</span>
                  <span className="tnum min-w-0 flex-1 truncate text-muted">
                    {displayValue(formA, k)}
                  </span>
                  <ArrowRight className="h-3 w-3 shrink-0 text-faint" />
                  <span className="tnum min-w-0 flex-1 truncate text-fg">
                    {displayValue(formB, k)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {bureauAlsoMoved && (
            <div className="mt-2 flex items-start gap-2 rounded-md border border-adverse/40 bg-surface-2/50 px-3 py-2 text-[11px] leading-relaxed text-muted">
              <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-adverse" />
              <span>
                <span className="text-fg">This is not a ceteris-paribus comparison.</span>{" "}
                <code className="text-faint">applicant_id</code> is the bureau's key, not a model
                feature — <code className="text-faint">MockBureau</code> hashes it to seed the
                pull, so A and B were scored on <em>different FICO scores</em> (
                <span className="tnum">{a.credit_report.fico_n.toFixed(2)}</span> vs{" "}
                <span className="tnum">{b.credit_report.fico_n.toFixed(2)}</span>). Any other
                field you changed is confounded with that. To isolate one lever, hold the
                applicant id fixed.
              </span>
            </div>
          )}
        </section>

        {/* ---- 2. the outcome ---- */}
        <section>
          <SectionLabel>Outcome</SectionLabel>
          <div className="mt-2 overflow-x-auto rounded-md border border-border">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-border text-[10px] uppercase tracking-wider text-faint">
                  <th className="px-3 py-1.5 text-left font-medium"></th>
                  <th className="px-3 py-1.5 text-right font-medium">A</th>
                  <th className="px-3 py-1.5 text-right font-medium">B</th>
                  <th className="px-3 py-1.5 text-right font-medium">Δ (B − A)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                <tr>
                  <td className="px-3 py-2 text-muted">Decision</td>
                  <Verdict d={a.decision} />
                  <Verdict d={b.decision} />
                  <td className="px-3 py-2 text-right text-faint">
                    {a.decision === b.decision ? "unchanged" : "FLIPPED"}
                  </td>
                </tr>
                <tr>
                  <td className="px-3 py-2 text-muted">raw margin (log-odds)</td>
                  <td className="tnum px-3 py-2 text-right text-muted">
                    {a.raw_margin_log_odds.toFixed(6)}
                  </td>
                  <td className="tnum px-3 py-2 text-right text-muted">
                    {b.raw_margin_log_odds.toFixed(6)}
                  </td>
                  <td
                    className={cn(
                      "tnum px-3 py-2 text-right font-semibold",
                      dMargin === 0 ? "text-faint" : "text-fg",
                    )}
                  >
                    {signed(dMargin)}
                  </td>
                </tr>
                <tr>
                  <td className="px-3 py-2 text-muted">p_raw</td>
                  <td className="tnum px-3 py-2 text-right text-muted">{a.p_raw.toFixed(6)}</td>
                  <td className="tnum px-3 py-2 text-right text-muted">{b.p_raw.toFixed(6)}</td>
                  <td className="tnum px-3 py-2 text-right text-muted">
                    {signed(b.p_raw - a.p_raw)}
                  </td>
                </tr>
                <tr>
                  <td className="px-3 py-2 text-fg">p_calibrated</td>
                  <td className="tnum px-3 py-2 text-right text-fg">{pct(a.p_calibrated)}</td>
                  <td className="tnum px-3 py-2 text-right text-fg">{pct(b.p_calibrated)}</td>
                  <td
                    className={cn(
                      "tnum px-3 py-2 text-right font-semibold",
                      dPCal === 0 ? "text-faint" : "text-fg",
                    )}
                  >
                    {dPCal === 0 ? "0 — identical" : signed(dPCal * 100, 2) + " pp"}
                  </td>
                </tr>
                <tr>
                  <td className="px-3 py-2 text-muted">calibrator step</td>
                  <td className="tnum px-3 py-2 text-right text-muted">
                    {blockA !== null ? `#${blockA}` : "—"}
                  </td>
                  <td className="tnum px-3 py-2 text-right text-muted">
                    {blockB !== null ? `#${blockB}` : "—"}
                  </td>
                  <td
                    className={cn(
                      "px-3 py-2 text-right font-semibold",
                      sameStep ? "text-adverse" : "text-faint",
                    )}
                  >
                    {cal === null ? "—" : sameStep ? "SAME STEP" : "different"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* THE LESSON, when it applies. */}
          {sameStep && dMargin !== 0 && (
            <div className="mt-2 rounded-md border border-adverse/40 bg-surface-2/50 px-3 py-2.5 text-[11px] leading-relaxed text-muted">
              <span className="text-fg">The log-odds moved. The probability did not.</span> A and
              B differ in their inputs and in their raw margin (by{" "}
              <span className="tnum text-fg">{signed(dMargin)}</span> log-odds), but both land on
              calibrator block <span className="tnum text-fg">#{blockA}</span> — a flat block,
              where <code>dp_cal/dp_raw</code> is exactly 0. So{" "}
              <span className="text-fg">p_calibrated is identical ({pct(a.p_calibrated)})</span>{" "}
              and the decision cannot tell them apart.
              <span className="mt-1.5 block text-faint">
                The change was real and the model saw it — the contribution table below says which
                feature moved and by how much. It simply did not survive the calibrator. This is
                what “per-feature attribution in probability points is undefined” looks like from
                the outside: there is no number of probability points to assign, because the
                probability did not move at all.
              </span>
            </div>
          )}

          {!sameStep && cal !== null && dMargin !== 0 && (
            <p className="mt-2 text-[11px] leading-relaxed text-faint">
              A and B land on <span className="text-muted">different blocks</span> (#{blockA} and
              #{blockB}), so the change crossed at least one ramp and the calibrated probability
              moved with it. The Δ is shown in percentage points{" "}
              <span className="text-muted">of the outcome</span> — which is a fact about these two
              applicants, and still not an attribution to any one feature.
            </p>
          )}
        </section>

        {/* ---- 3. which feature actually moved ---- */}
        <ContributionDelta a={a} b={b} touched={featuresTouchedBy(changed)} />

        {/* ---- 4. the reason codes ---- */}
        <ReasonCodeDiff a={a} b={b} />
      </CardContent>
    </Card>
  )
}

// ===========================================================================

/**
 * Per-feature Δ contribution, sorted by |Δ|. This is the answer to "which feature
 * did my edit actually move" -- and it is answerable precisely because the scale
 * is log-odds, where the contributions are additive and a difference is
 * meaningful. The same table in probability points could not be built.
 */
function ContributionDelta({
  a,
  b,
  touched,
}: {
  a: ScoreResponse
  b: ScoreResponse
  /** The model features downstream of the edited form fields. Everything else got
   *  the SAME input in A and in B. */
  touched: Set<string>
}) {
  const features = [
    ...new Set([
      ...Object.keys(a.contributions_log_odds),
      ...Object.keys(b.contributions_log_odds),
    ]),
  ]
  const rows = features
    .map((f) => {
      const va = a.contributions_log_odds[f] ?? 0
      const vb = b.contributions_log_odds[f] ?? 0
      // A feature whose INPUT was identical in A and B, but whose ATTRIBUTION
      // still moved. Not a bug and not a changed input: TreeSHAP attributes over
      // the whole feature vector, so editing one field redistributes credit
      // across the others. Marked, because the alternative is a reader seeing
      // `fico_n  Δ −0.11` beside an unchanged applicant id and concluding the
      // bureau returned a different score. It did not.
      const reattributed = !touched.has(f) && vb !== va
      return { f, va, vb, d: vb - va, reattributed }
    })
    .sort((x, y) => Math.abs(y.d) - Math.abs(x.d))

  const max = Math.max(...rows.map((r) => Math.abs(r.d)), Number.EPSILON)
  const moved = rows.filter((r) => r.d !== 0).length
  const anyReattributed = rows.some((r) => r.reattributed)

  return (
    <section>
      <div className="flex items-baseline justify-between">
        <SectionLabel>Contributions · Δ in log-odds · sorted by |Δ|</SectionLabel>
        <span className="text-[10px] text-faint">
          {moved} of {rows.length} features moved
        </span>
      </div>

      <div className="mt-2 overflow-x-auto rounded-md border border-border">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-border text-[10px] uppercase tracking-wider text-faint">
              <th className="px-3 py-1.5 text-left font-medium">feature</th>
              <th className="px-3 py-1.5 text-right font-medium">A</th>
              <th className="px-3 py-1.5 text-right font-medium">B</th>
              <th className="px-3 py-1.5 text-right font-medium">Δ</th>
              <th className="w-24 px-3 py-1.5"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map(({ f, va, vb, d, reattributed }) => (
              <tr key={f} className={cn(d === 0 && "opacity-45")}>
                <td className="px-3 py-1.5">
                  <span className="text-muted">{f}</span>
                  {reattributed && (
                    <span
                      className="ml-2 text-[10px] text-faint"
                      title="Same input in A and B — only the attribution moved."
                    >
                      input unchanged
                    </span>
                  )}
                </td>
                <td className="tnum px-3 py-1.5 text-right text-faint">{va.toFixed(6)}</td>
                <td className="tnum px-3 py-1.5 text-right text-faint">{vb.toFixed(6)}</td>
                <td
                  className={cn(
                    "tnum px-3 py-1.5 text-right",
                    d > 0 ? "text-adverse" : d < 0 ? "text-approve" : "text-faint",
                  )}
                >
                  {d === 0 ? "0" : signed(d)}
                </td>
                <td className="px-3 py-1.5">
                  {/* Magnitude bar, relative to the largest |Δ| in THIS comparison.
                      It is not a probability and is not scaled to one. */}
                  <div className="h-1 w-full overflow-hidden rounded-full bg-surface-2">
                    <div
                      className={cn("h-full", d > 0 ? "bg-adverse" : "bg-approve")}
                      style={{ width: `${(Math.abs(d) / max) * 100}%` }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-faint">
        Δ is <span className="text-muted">B − A</span>, in the model's raw log-odds margin.
        Positive (amber) means the change pushed B <em>toward</em> default; negative (green),
        away. These sum to the Δ raw margin above, because SHAP contributions in log-odds are
        additive — which is exactly the property that would be lost if they were converted to
        probability points, and exactly why they are not.
      </p>

      {anyReattributed && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-faint">
          <span className="text-muted">“input unchanged”</span> means A and B fed that feature the
          same value and only its <em>attribution</em> moved. That is expected, not a glitch: SHAP
          attributes over the whole feature vector, so editing one field redistributes credit
          among the others.{" "}
          <span className="text-muted">
            A moved <code>fico_n</code> contribution beside an unedited applicant id does not mean
            the bureau returned a different score — the FICO is right there in both panels below,
            and it is identical.
          </span>
        </p>
      )}
    </section>
  )
}

// ===========================================================================

/**
 * Which principal adverse factors appeared, dropped, or moved rank.
 *
 * A factor DROPS off the list when its contribution stops being risk-increasing:
 * an adverse-action notice can only name factors that pushed the applicant toward
 * default, so a feature whose contribution turns negative is not demoted -- it
 * ceases to be a reason at all. That is worth saying out loud, because "it
 * disappeared" reads like a bug and it is not one.
 */
function ReasonCodeDiff({ a, b }: { a: ScoreResponse; b: ScoreResponse }) {
  const ra = new Map(a.reason_codes.map((c) => [c.feature, c]))
  const rb = new Map(b.reason_codes.map((c) => [c.feature, c]))

  const dropped = a.reason_codes.filter((c) => !rb.has(c.feature))
  const appeared = b.reason_codes.filter((c) => !ra.has(c.feature))
  const moved = b.reason_codes.filter((c) => {
    const before = ra.get(c.feature)
    return before && before.rank !== c.rank
  })
  const quiet = dropped.length === 0 && appeared.length === 0 && moved.length === 0

  return (
    <section>
      <SectionLabel>Principal adverse factors · A → B</SectionLabel>

      <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <RankList title="A" codes={a.reason_codes} otherHas={(f) => rb.has(f)} />
        <RankList title="B" codes={b.reason_codes} otherHas={(f) => ra.has(f)} />
      </div>

      <div className="mt-2 space-y-1.5">
        {quiet && (
          <p className="text-[11px] text-faint">
            The same factors, in the same order. The change did not alter which reasons an
            adverse-action notice would cite.
          </p>
        )}

        {dropped.map((c) => {
          const now = b.contributions_log_odds[c.feature] ?? 0
          return (
            <p key={c.feature} className="flex items-start gap-1.5 text-[11px] text-muted">
              <Minus className="mt-0.5 h-3 w-3 shrink-0 text-approve" />
              <span>
                <span className="text-fg">{c.feature}</span> was rank {c.rank} for A and is{" "}
                <span className="text-approve">not a reason for B at all</span>
                {now < 0 ? (
                  <>
                    {" "}
                    — its contribution turned <span className="text-approve">risk-decreasing</span>{" "}
                    (<span className="tnum">{signed(now)}</span>). A factor that pushes an
                    applicant <em>away</em> from default cannot be a principal reason for an
                    adverse action, so it does not drop in rank; it stops being a reason.
                  </>
                ) : (
                  <> — it was displaced by larger factors.</>
                )}
              </span>
            </p>
          )
        })}

        {appeared.map((c) => (
          <p key={c.feature} className="flex items-start gap-1.5 text-[11px] text-muted">
            <Plus className="mt-0.5 h-3 w-3 shrink-0 text-adverse" />
            <span>
              <span className="text-fg">{c.feature}</span> is a new principal reason for B, at
              rank {c.rank} (<span className="tnum text-adverse">
                {signed(c.contribution_log_odds)}
              </span>{" "}
              log-odds).
            </span>
          </p>
        ))}

        {moved.map((c) => (
          <p key={c.feature} className="flex items-start gap-1.5 text-[11px] text-muted">
            <ArrowRight className="mt-0.5 h-3 w-3 shrink-0 text-faint" />
            <span>
              <span className="text-fg">{c.feature}</span> moved from rank{" "}
              {ra.get(c.feature)!.rank} to rank {c.rank}.
            </span>
          </p>
        ))}
      </div>
    </section>
  )
}

function RankList({
  title,
  codes,
  otherHas,
}: {
  title: string
  codes: ScoreResponse["reason_codes"]
  otherHas: (feature: string) => boolean
}) {
  return (
    <div className="rounded-md border border-border">
      <div className="border-b border-border px-3 py-1.5 text-[10px] uppercase tracking-wider text-faint">
        {title}
      </div>
      {codes.length === 0 ? (
        <p className="px-3 py-2 text-[11px] text-faint">
          No principal reasons could be named — every contribution was zero or risk-decreasing.
        </p>
      ) : (
        <div className="divide-y divide-border">
          {codes.map((c) => (
            <div key={c.rank} className="flex items-center gap-2 px-3 py-1.5">
              <span className="tnum w-3 shrink-0 text-[11px] text-faint">{c.rank}</span>
              <span
                className={cn(
                  "min-w-0 flex-1 truncate text-[12px]",
                  otherHas(c.feature) ? "text-muted" : "text-adverse",
                )}
              >
                {c.feature}
              </span>
              <span className="tnum shrink-0 text-[11px] text-adverse">
                {signed(c.contribution_log_odds, 4)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Verdict({ d }: { d: "APPROVE" | "REJECT" }) {
  return (
    <td
      className={cn(
        "px-3 py-2 text-right font-semibold",
        d === "APPROVE" ? "text-approve" : "text-reject",
      )}
    >
      {d}
    </td>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] uppercase tracking-wider text-faint">{children}</div>
}
