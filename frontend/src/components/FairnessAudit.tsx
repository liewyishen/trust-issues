import * as React from "react"
import { Loader2 } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Failure } from "@/components/Failure"
import {
  AuditStaleError,
  RouteNotAvailableError,
  getFairness,
  type AblationStateCI,
  type FairnessResponse,
  type StateEO,
} from "@/lib/api"

/**
 * The fairness audit, told in the order the evidence was actually earned.
 *
 * Three movements, and the first two are the honest ones:
 *
 *   1. THE GREEN WALL. Layer 1, the shipped model, at the threshold /score
 *      actually decides at: fifty states, fifty confidence intervals, zero
 *      confirmed. A wall of green -- and on its own it proves almost nothing,
 *      because that threshold approves ~82% of good applicants and a permissive
 *      cutoff washes state-level differences out. The repo's own position is
 *      that a passing audit at one threshold is not a conclusion. Leading with
 *      the green wall and stopping there would be the dishonest version of this
 *      screen.
 *
 *   2. WHY WE DON'T BELIEVE IT YET. Layer 2 sweeps the threshold from 0.12 to
 *      0.30. The all-clear survives all of it. THAT is what turns "we saw no
 *      disparity" into evidence: not the finding, the failure to break it.
 *
 *   3. THE COUNTERFACTUAL. Layer 3 retrains the model WITH addr_state and
 *      WITHOUT it, and puts a bootstrap CI on both sides. Mississippi's entire
 *      interval sits under the 0.80 line with the state label in, and clears it
 *      without -- the two intervals disjoint. That is the finding, and it is the
 *      reason the shipped model has no addr_state to lean on.
 *
 * Nothing here computes a ratio, an interval, or a verdict. Every number is read
 * off GET /fairness, which reads models/fairness_audit.json, which is the output
 * of the real src/fairness.py. The 0.80 line is drawn at constants.eo_threshold
 * -- the value the audit actually used -- never at a literal typed in here.
 */

// The three verdicts src/fairness.py can return, and the ONLY thing colour means
// on this screen. Not decoration: a state is red because its whole interval sits
// below the line, amber because the interval straddles it, green because it
// doesn't reach it.
const VERDICT_TONE = {
  confirmed: {
    bar: "bg-reject",
    text: "text-reject",
    ring: "border-reject/40 bg-reject-dim/40",
    label: "confirmed",
  },
  inconclusive: {
    bar: "bg-adverse",
    text: "text-adverse",
    ring: "border-adverse/40 bg-adverse/10",
    label: "inconclusive",
  },
  clear: {
    bar: "bg-approve",
    text: "text-approve",
    ring: "border-approve/30 bg-approve-dim/30",
    label: "clear",
  },
} as const

type Tone = keyof typeof VERDICT_TONE

/** src/fairness.py's verdict strings, mapped to a tone. Matched on the PREFIX the
 *  audit itself writes ("confirmed (CI fully < 0.80)", "inconclusive (CI straddles
 *  0.80)", "clear") rather than by searching for a number inside the prose. */
function toneOf(verdict: string): Tone {
  if (verdict.startsWith("confirmed")) return "confirmed"
  if (verdict.startsWith("inconclusive")) return "inconclusive"
  return "clear"
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const eo = (x: number) => x.toFixed(3)

export function FairnessAudit() {
  const [audit, setAudit] = React.useState<FairnessResponse | null>(null)
  const [error, setError] = React.useState<unknown>(null)

  React.useEffect(() => {
    let live = true
    getFairness()
      .then((a) => live && setAudit(a))
      .catch((e) => live && setError(e))
    return () => {
      live = false
    }
  }, [])

  // The two states that are NOT failures of this app, and must not be drawn
  // around. See serving/fairness.py.
  if (error instanceof RouteNotAvailableError) return <NoAudit />
  if (error instanceof AuditStaleError) return <StaleAudit error={error} />
  if (error) return <Failure error={error} />
  if (!audit) return <FirstLoad />

  return (
    <div className="space-y-4">
      <Provenance audit={audit} />
      <GreenWall audit={audit} />
      <Sweep audit={audit} />
      <Counterfactual audit={audit} />
      <Honesty audit={audit} />
    </div>
  )
}

/* ------------------------------------------------------------------ *
 * Provenance -- the binding, shown rather than asserted.
 * ------------------------------------------------------------------ */
function Provenance({ audit }: { audit: FairnessResponse }) {
  const bound = audit.model.trained_at === audit.shipped_model_trained_at

  return (
    <Card>
      <CardHeader className="flex items-start justify-between">
        <div>
          <CardTitle>The fairness audit</CardTitle>
          <p className="mt-0.5 text-[11px] text-faint">
            Three layers, run offline against the real dataset. Read, not recomputed.
          </p>
        </div>
        {/* The service already refuses to send this response at all when the two
            timestamps differ (409). This badge is what makes that check VISIBLE
            instead of merely trustworthy. */}
        <span
          className={`shrink-0 rounded border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider ${
            bound
              ? "border-approve/30 bg-approve-dim/30 text-approve"
              : "border-reject/40 bg-reject-dim/40 text-reject"
          }`}
        >
          {bound ? "bound to the shipped model" : "not the shipped model"}
        </span>
      </CardHeader>

      <CardContent className="space-y-4">
        <p className="text-[12px] leading-relaxed text-muted">
          This audit cannot run here. It needs the 167 MB assessment dataset — which never
          enters the serving image, because the brief forbids redistributing it — and ~40
          seconds to retrain both halves of its ablation. So it runs{" "}
          <span className="text-fg">offline</span>, and what ships is its output: ~35 KB of
          derived ratios. Aggregate ratios are not the dataset.
        </p>
        <p className="text-[12px] leading-relaxed text-muted">
          A frozen file is the one thing in this service that can go{" "}
          <span className="text-fg">stale</span>: retrain the model, and the JSON would still
          cheerfully report the numbers below about a booster that no longer exists. So it is
          bound to the model by <code className="text-faint">trained_at</code> — the same
          binding that stops a calibrator being used with the wrong booster. If they ever
          disagree, this screen shows no ratios at all.
        </p>

        <dl className="grid grid-cols-2 gap-3 border-t border-border pt-3 text-[11px] sm:grid-cols-4">
          <Figure label="audit ran against" value={short(audit.model.trained_at)} />
          <Figure label="model being served" value={short(audit.shipped_model_trained_at)} />
          <Figure label="audit generated" value={short(audit.generated_at)} />
          <Figure
            label="addr_state in model"
            value={audit.model.includes_addr_state ? "yes" : "no"}
            tone={audit.model.includes_addr_state ? "text-reject" : "text-approve"}
          />
        </dl>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ *
 * Movement 1 -- the green wall, and why it isn't evidence.
 * ------------------------------------------------------------------ */
function GreenWall({ audit }: { audit: FairnessResponse }) {
  const { layer1, layer2, constants } = audit
  const states = [...layer1.states].sort((a, b) => a.eo_ratio - b.eo_ratio)

  // The approval rate AT the operating point -- the number that makes the green
  // wall weak, so it is read from the audit rather than asserted.
  //
  // An EXACT match, not the nearest row. scripts/audit_fairness.py sweeps the
  // operating threshold explicitly for this reason: the notebook's
  // SWEEP_THRESHOLDS does not contain 0.25000000000000006, and reading the
  // nearest row (0.26) while calling it "the operating point" would report a
  // real, different approval rate as if it were exact. If the row is ever
  // genuinely absent, say nothing rather than round.
  const operatingRow = layer2.rows.find((r) => r.threshold === layer1.threshold)

  return (
    <Card>
      <CardHeader className="flex items-start justify-between">
        <div>
          <CardTitle>1 · Fifty states, zero flags</CardTitle>
          <p className="mt-0.5 text-[11px] text-faint">
            Layer 1 — the shipped model, at the threshold <code>/score</code> decides at (
            {layer1.threshold.toFixed(2)})
          </p>
        </div>
        <span className="shrink-0 rounded border border-approve/30 bg-approve-dim/30 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-approve">
          {layer1.n_confirmed} confirmed
        </span>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-1">
          {states.map((s) => (
            <StateChip key={s.state} row={s} />
          ))}
        </div>

        {/* The whole point of this card. */}
        <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2.5 text-[12px] leading-relaxed text-muted">
          <span className="text-fg">And on its own, this proves almost nothing.</span>{" "}
          {operatingRow && (
            <>
              At this threshold the model approves{" "}
              <span className="tnum text-fg">
                {pct(operatingRow.national_good_approval_rate)}
              </span>{" "}
              of good applicants.{" "}
            </>
          )}
          A cutoff that permissive approves nearly everyone, everywhere — which washes
          state-level differences out and can return an all-clear whether or not a disparity
          exists. A fairness conclusion that depends on one threshold is not a conclusion. That
          is what Layer 2 is for.
        </p>

        <p className="text-[11px] leading-relaxed text-faint">
          Each chip is one state's Equal-Opportunity ratio: among applicants who actually repaid,
          the share this state got approved, over the national share. A state is flagged only if
          its <span className="text-muted">entire {constants.n_boot.toLocaleString()}-resample
          bootstrap interval</span> sits below {constants.eo_threshold.toFixed(2)} — a point
          estimate on a finite sample is noise, not evidence. States with fewer than{" "}
          {constants.min_n} good applicants are not reported at all.
        </p>
      </CardContent>
    </Card>
  )
}

function StateChip({ row }: { row: StateEO }) {
  const tone = VERDICT_TONE[toneOf(row.verdict)]
  return (
    <span
      title={`${row.state} — EO ${eo(row.eo_ratio)}, 95% CI [${eo(row.ci_low)}, ${eo(
        row.ci_high,
      )}], n=${row.n_good.toLocaleString()} good applicants · ${row.verdict}`}
      className={`tnum rounded border px-1.5 py-0.5 text-[10px] font-medium ${tone.ring} ${tone.text}`}
    >
      {row.state} {eo(row.eo_ratio)}
    </span>
  )
}

/* ------------------------------------------------------------------ *
 * Movement 2 -- the attempt to break it.
 * ------------------------------------------------------------------ */
function Sweep({ audit }: { audit: FairnessResponse }) {
  const { layer2, constants, layer1 } = audit
  const watch = constants.watch_states

  // Does any watch state fall under the line at ANY threshold in the sweep? Asked
  // of the data, not assumed -- if a future retrain broke this, the sentence
  // below would have to change, and it does.
  const broke = layer2.rows.some((r) =>
    watch.some((s) => typeof r[s] === "number" && r[s] < constants.eo_threshold),
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>2 · Try to break it</CardTitle>
        <p className="mt-0.5 text-[11px] text-faint">
          Layer 2 — tighten the threshold and see whether the all-clear survives
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <table className="tnum w-full min-w-[520px] text-[11px]">
            <thead>
              <tr className="border-b border-border text-left text-faint">
                <th className="py-1.5 pr-3 font-medium">threshold</th>
                <th className="py-1.5 pr-3 font-medium">approves</th>
                {watch.map((s) => (
                  <th key={s} className="py-1.5 pr-3 text-right font-medium">
                    {s}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {layer2.rows.map((r) => {
                // Exact, not "close enough" -- the sweep contains the operating
                // point because scripts/audit_fairness.py put it there.
                const operating = r.threshold === layer1.threshold
                return (
                  <tr
                    key={r.threshold}
                    className={`border-b border-border/50 ${
                      operating ? "bg-surface-2/60" : ""
                    }`}
                  >
                    <td className="py-1.5 pr-3 text-fg">
                      {r.threshold.toFixed(2)}
                      {operating && (
                        <span className="ml-1.5 text-[9px] uppercase tracking-wider text-faint">
                          operating
                        </span>
                      )}
                    </td>
                    <td className="py-1.5 pr-3 text-muted">
                      {pct(r.national_good_approval_rate)}
                    </td>
                    {watch.map((s) => {
                      const v = r[s]
                      const under = typeof v === "number" && v < constants.eo_threshold
                      return (
                        <td
                          key={s}
                          className={`py-1.5 pr-3 text-right ${
                            under ? "text-reject" : "text-muted"
                          }`}
                        >
                          {typeof v === "number" ? eo(v) : "—"}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2.5 text-[12px] leading-relaxed text-muted">
          {broke ? (
            <>
              <span className="text-reject">The all-clear did not survive.</span> At least one
              watched state falls below {constants.eo_threshold.toFixed(2)} once the threshold
              tightens — so Layer 1's result was an artifact of a permissive cutoff, not a
              finding.
            </>
          ) : (
            <>
              <span className="text-fg">It holds.</span> Tightening the cutoff from{" "}
              {layer2.rows[layer2.rows.length - 1].threshold.toFixed(2)} down to{" "}
              {layer2.rows[0].threshold.toFixed(2)} — dropping approvals from{" "}
              {pct(layer2.rows[layer2.rows.length - 1].national_good_approval_rate)} to{" "}
              {pct(layer2.rows[0].national_good_approval_rate)} — never pushes a watched state
              under the line. The all-clear is stable, not an artifact of where the cutoff
              happened to sit. <span className="text-fg">That</span> is what makes it evidence:
              not the finding, the failed attempt to break it.
            </>
          )}
        </p>

        <p className="text-[11px] leading-relaxed text-faint">
          This sweep runs on the model that is <span className="text-muted">actually shipped</span>{" "}
          — the one with <code>addr_state</code> already removed. It is clean because the fix is
          already in. The audit that found the problem is next.
        </p>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ *
 * Movement 3 -- the counterfactual. The chart the whole thing is for.
 * ------------------------------------------------------------------ */
function Counterfactual({ audit }: { audit: FairnessResponse }) {
  const { layer3, constants } = audit
  const [showAll, setShowAll] = React.useState(false)

  const byState = new Map(layer3.states.map((r) => [r.state, r]))
  const watch = constants.watch_states
    .map((s) => byState.get(s))
    .filter((r): r is AblationStateCI => r !== undefined)

  const rows = showAll
    ? [...layer3.states].sort((a, b) => a.eo_ratio_with_state - b.eo_ratio_with_state)
    : watch

  // The axis is DERIVED from the intervals actually being drawn, never a guess.
  // A hardcoded domain would clip a CI the day a retrain widened one, and a
  // clipped interval is a lie about uncertainty.
  const lows = layer3.states.flatMap((r) => [r.ci_low_with_state, r.ci_low_no_state])
  const highs = layer3.states.flatMap((r) => [r.ci_high_with_state, r.ci_high_no_state])
  const min = Math.floor(Math.min(...lows) * 20) / 20
  const max = Math.ceil(Math.max(...highs) * 20) / 20
  const x = (v: number) => ((v - min) / (max - min)) * 100

  const confirmedWith = layer3.states.filter((r) =>
    r.verdict_with_state.startsWith("confirmed"),
  )
  const confirmedWithout = layer3.states.filter((r) =>
    r.verdict_no_state.startsWith("confirmed"),
  )

  return (
    <Card>
      <CardHeader className="flex items-start justify-between">
        <div>
          <CardTitle>3 · Retrain it with the state label, and without</CardTitle>
          <p className="mt-0.5 text-[11px] text-faint">
            Layer 3 — the ablation, at threshold {layer3.threshold.toFixed(2)}, with a bootstrap
            interval on both sides
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="shrink-0 rounded border border-border bg-surface-2 px-2 py-1 text-[10px] uppercase tracking-wider text-muted transition-colors hover:border-border-strong hover:text-fg"
        >
          {showAll ? `${watch.length} watched` : `all ${layer3.states.length}`}
        </button>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* The legend describes what is actually drawn. The two variants are told
            apart by OPACITY, not by hue -- hue is reserved for the verdict, which
            is the only thing colour is allowed to mean on this screen. A legend
            claiming "grey = with state, green = without" would be readable and
            wrong: it would recolour a red interval as a variant marker. */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[10px] text-faint">
          <span className="flex items-center gap-1.5">
            <span className="h-[3px] w-6 rounded-full bg-muted opacity-45" />
            faded, above — <span className="text-muted">with</span> <code>addr_state</code>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-[3px] w-6 rounded-full bg-muted" />
            solid, below — <span className="text-muted">without</span> it (shipped)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-3 w-px bg-reject" />
            the {constants.eo_threshold.toFixed(2)} line — from the audit, not typed here
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-reject" />
            <span className="h-2 w-2 rounded-full bg-adverse" />
            <span className="h-2 w-2 rounded-full bg-approve" />
            colour is the verdict: confirmed · inconclusive · clear
          </span>
        </div>
        <p className="-mt-2 text-[10px] leading-relaxed text-faint">
          Each bar is a 95% bootstrap interval, with the point estimate marked on it. The bar{" "}
          <span className="text-muted">is</span> the finding — a point alone could not tell this
          shift apart from noise.
        </p>

        <div className="space-y-1">
          {rows.map((r) => (
            <AblationRow key={r.state} row={r} x={x} eoLine={constants.eo_threshold} />
          ))}
          <div className="tnum h-4 pl-[6.5rem] pr-16 text-[9px] text-faint">
            <div className="flex justify-between">
              <span>{min.toFixed(2)}</span>
              <span>EO ratio</span>
              <span>{max.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* The finding, stated as an interval claim. */}
        <div className="rounded-md border border-border bg-surface-2/50 px-3 py-2.5 text-[12px] leading-relaxed text-muted">
          <p>
            <span className="text-fg">
              {confirmedWith.length === 0
                ? "No state is confirmed with the state label in."
                : `${confirmedWith.map((r) => r.state).join(", ")} ${
                    confirmedWith.length === 1 ? "is" : "are"
                  } confirmed with the state label in — the entire interval below the line.`}{" "}
            </span>
            {confirmedWithout.length === 0
              ? "Remove it, and none are."
              : `Remove it, and ${confirmedWithout.map((r) => r.state).join(", ")} still ${
                  confirmedWithout.length === 1 ? "is" : "are"
                }.`}{" "}
            The state label was not encoding those applicants' finances — the model was leaning on
            it as a shortcut.
          </p>
          <p className="mt-2">
            The price, paid: test AUC{" "}
            <span className="tnum text-fg">{layer3.auc_with_state.toFixed(4)}</span> →{" "}
            <span className="tnum text-fg">{layer3.auc_no_state.toFixed(4)}</span>, a cost of{" "}
            <span className="tnum text-adverse">{layer3.auc_cost.toFixed(4)}</span>. The shipped
            model is the one on the right.
          </p>
        </div>

        <p className="text-[11px] leading-relaxed text-faint">
          Both models here are retrained from scratch on the same splits with the same fixed
          round count — one feature toggled, everything else held still. That is what makes the
          two sides comparable; neither of them is the shipped booster, and comparing one against
          it would confound the feature with the training procedure.
        </p>
      </CardContent>
    </Card>
  )
}

function AblationRow({
  row,
  x,
  eoLine,
}: {
  row: AblationStateCI
  x: (v: number) => number
  eoLine: number
}) {
  const withTone = VERDICT_TONE[toneOf(row.verdict_with_state)]
  const noTone = VERDICT_TONE[toneOf(row.verdict_no_state)]

  return (
    <div className="group flex items-center gap-2">
      <div className="tnum flex w-24 shrink-0 items-baseline gap-1.5 text-[11px]">
        <span className={`font-medium ${withTone.text}`}>{row.state}</span>
        <span className="text-[9px] text-faint">
          {row.n_good_with_state.toLocaleString()}
        </span>
      </div>

      <div className="relative h-7 flex-1 rounded bg-surface-2/40">
        {/* The 0.80 line, at constants.eo_threshold. */}
        <div
          aria-hidden
          className="absolute inset-y-0 w-px bg-reject/70"
          style={{ left: `${x(eoLine)}%` }}
        />

        {/* WITH addr_state: the interval, then the point estimate on it. */}
        <Interval
          lo={row.ci_low_with_state}
          hi={row.ci_high_with_state}
          point={row.eo_ratio_with_state}
          x={x}
          tone={withTone.bar}
          dim
          top="top-1.5"
          title={`${row.state} WITH addr_state — EO ${eo(row.eo_ratio_with_state)}, CI [${eo(
            row.ci_low_with_state,
          )}, ${eo(row.ci_high_with_state)}] · ${row.verdict_with_state}`}
        />

        {/* WITHOUT it: the shipped configuration. */}
        <Interval
          lo={row.ci_low_no_state}
          hi={row.ci_high_no_state}
          point={row.eo_ratio_no_state}
          x={x}
          tone={noTone.bar}
          top="top-4"
          title={`${row.state} WITHOUT addr_state — EO ${eo(row.eo_ratio_no_state)}, CI [${eo(
            row.ci_low_no_state,
          )}, ${eo(row.ci_high_no_state)}] · ${row.verdict_no_state}`}
        />
      </div>

      <span
        className={`tnum w-14 shrink-0 text-right text-[10px] ${
          row.eo_ratio_no_state > row.eo_ratio_with_state ? "text-approve" : "text-faint"
        }`}
      >
        {row.eo_ratio_no_state > row.eo_ratio_with_state ? "+" : ""}
        {(row.eo_ratio_no_state - row.eo_ratio_with_state).toFixed(3)}
      </span>
    </div>
  )
}

/** One bootstrap interval: a bar from ci_low to ci_high with the point estimate
 *  marked on it. The bar IS the uncertainty -- drawing only the point estimate is
 *  the error this whole layer exists to correct. */
function Interval({
  lo,
  hi,
  point,
  x,
  tone,
  top,
  dim,
  title,
}: {
  lo: number
  hi: number
  point: number
  x: (v: number) => number
  tone: string
  top: string
  dim?: boolean
  title: string
}) {
  return (
    <div
      title={title}
      className={`absolute ${top} h-[3px] rounded-full ${tone} ${dim ? "opacity-45" : ""}`}
      style={{ left: `${x(lo)}%`, width: `${Math.max(x(hi) - x(lo), 0.4)}%` }}
    >
      <span
        aria-hidden
        className={`absolute top-1/2 h-[7px] w-[7px] -translate-x-1/2 -translate-y-1/2 rounded-full ${tone}`}
        style={{ left: `${((point - lo) / Math.max(hi - lo, 1e-9)) * 100}%` }}
      />
    </div>
  )
}

/* ------------------------------------------------------------------ *
 * The part that costs us something to say.
 * ------------------------------------------------------------------ */
function Honesty({ audit }: { audit: FairnessResponse }) {
  const { layer3, constants } = audit
  const byState = new Map(layer3.states.map((r) => [r.state, r]))

  // Where the audit's own POINT-ESTIMATE verdict disagrees with the interval.
  // Computed, not hardcoded: if a retrain resolved these, the callout disappears
  // on its own rather than lingering as a stale confession.
  const disagreements = layer3.watch.filter((w) => {
    const ci = byState.get(w.state)
    if (!ci) return false
    const pointSaysClear = w.verdict.startsWith("was already clear")
    const ciSaysMaybe = ci.verdict_with_state.startsWith("inconclusive")
    return pointSaysClear && ciSaysMaybe
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>What this doesn't settle</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-[12px] leading-relaxed text-muted">
        {disagreements.length > 0 && (
          <p className="rounded-md border border-adverse/30 bg-adverse/5 px-3 py-2.5">
            <span className="text-adverse">
              The audit's own verdict column disagrees with its own intervals.
            </span>{" "}
            {disagreements.map((w) => w.state).join(" and ")}{" "}
            {disagreements.length === 1 ? "reads" : "read"} “was already clear” — a verdict
            decided on a point estimate (
            {disagreements.map((w) => `${w.state} ${eo(w.eo_with_state)}`).join(", ")}, against a{" "}
            {constants.eo_threshold.toFixed(2)} line) — while the bootstrap interval straddles
            that line and is honestly <span className="text-adverse">inconclusive</span>. Layer
            3's verdict logic is the notebook's, and it is left exactly as it is; the intervals
            are shipped beside it so you can see where it is thinner than it sounds. Trusting a
            point estimate is the specific error this audit exists to refuse, and it is not
            immune to it.
          </p>
        )}

        <p>
          <span className="text-fg">This is a geographic-proxy risk analysis, not a legal
          finding.</span>{" "}
          <code>addr_state</code> is not a protected class under ECOA. What is measured here is
          whether the model leans on a state label as a shortcut for risk — “digital redlining”
          in industry usage — not disparate impact under any statute. That is a legal conclusion,
          and this audit is not equipped to make it.
        </p>

        <p>
          <span className="text-fg">There is no per-applicant version of this screen</span>, and
          there cannot be: <code>addr_state</code> is not a field on <code>/score</code>, and the
          shipped model does not carry it as a feature. Fairness here is a property of a model
          over a population, not of one decision — an audit, not a score.
        </p>

        <p>
          Only states with at least {constants.min_n} good applicants are reported. The rest are
          not “clear” — they are unmeasured, and the audit says so by leaving them out rather
          than by reporting a confidence interval nobody should believe.
        </p>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ *
 * The states where there is nothing honest to draw.
 * ------------------------------------------------------------------ */
function NoAudit() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>The fairness audit isn't available on this deployment</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-[12px] leading-relaxed text-muted">
        <p>
          <code className="text-faint">GET /fairness</code> returned 404. The service has no audit
          artifact to serve, so there is nothing to show — and nothing has been drawn in its
          place.
        </p>
        <p>
          The audit is produced offline by{" "}
          <code className="text-faint">scripts/audit_fairness.py</code>, which needs the
          assessment dataset — a file that deliberately never enters the serving image. Its
          output normally ships with the model as{" "}
          <code className="text-faint">models/fairness_audit.json</code>.
        </p>
      </CardContent>
    </Card>
  )
}

function StaleAudit({ error }: { error: AuditStaleError }) {
  return (
    <Card>
      <CardHeader className="flex items-start justify-between">
        <CardTitle>This audit is about a different model</CardTitle>
        <span className="shrink-0 rounded border border-reject/40 bg-reject-dim/40 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-reject">
          409 · refused
        </span>
      </CardHeader>
      <CardContent className="space-y-3 text-[12px] leading-relaxed text-muted">
        <p>
          The fairness audit on disk was run against one booster; the service is serving another.
          Its ratios describe a model that is not making these decisions, so{" "}
          <span className="text-fg">the service did not send them</span> — and this screen has
          none to draw.
        </p>

        <dl className="grid grid-cols-1 gap-3 border-y border-border py-3 text-[11px] sm:grid-cols-2">
          <Figure label="the audit ran against" value={short(error.auditModelTrainedAt)} />
          <Figure label="the model being served" value={short(error.shippedModelTrainedAt)} />
        </dl>

        <p>
          Showing the numbers with a warning attached was the other option, and it is not a middle
          ground: a screen that is handed ratios will draw them, and a reader will remember the
          chart, not the caveat. Withholding them is the only thing that reliably stops a stale
          number being read as a current one.
        </p>
        <p className="text-faint">
          Re-run <code>uv run python scripts/audit_fairness.py</code> against the shipped model.
        </p>
      </CardContent>
    </Card>
  )
}

function FirstLoad() {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-10 text-[12px] text-faint">
        <Loader2 className="h-4 w-4 animate-spin" />
        Reading the audit off the service…
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
function Figure({
  label,
  value,
  tone = "text-fg",
}: {
  label: string
  value: string
  tone?: string
}) {
  return (
    <div>
      <dt className="text-faint">{label}</dt>
      <dd className={`tnum mt-0.5 ${tone}`}>{value}</dd>
    </div>
  )
}

/** ISO timestamp -> the part a human reads. Never reformatted into a locale --
 *  the artifact's own string is the identifier being compared. */
function short(iso: string | null): string {
  if (!iso) return "—"
  return iso.slice(0, 19).replace("T", " ")
}
