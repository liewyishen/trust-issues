import { Check, ChevronLeft, ChevronRight, Crosshair } from "lucide-react"
import * as React from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { CalibratorResponse } from "@/lib/api"
import {
  decide,
  evaluate,
  polyline,
  rampMidpoints,
  reachableOutputs,
  type PlotPoint,
  type Reading,
} from "@/lib/calibrator"
import { cn } from "@/lib/utils"

/*
 * THE CALIBRATOR, DRAWN.
 *
 * This is TIER 3. In the single-applicant flow it sits between the decision
 * (tier 2) and the additive explanation (tier 4), because it is what connects
 * them: tier 2 shows a probability, tier 4 shows contributions in log-odds, and a
 * reader who reads those in sequence is entitled to ask why the contributions are
 * not in probability points. The answer is this shape. On a flat block the
 * derivative of p_cal with respect to the model's own output is exactly zero, so
 * "this feature is worth N points of probability" has no value to compute -- not
 * a value nobody has computed yet, no value at all.
 *
 * It plots N applicants, not one. The compare view passes two, and when they land
 * on the same flat block the plot shows the collapse rather than asserting it:
 * two different raw scores, one identical probability, sitting on one step.
 *
 * Every knot, the domain and the decision threshold come from GET /calibrator,
 * which serving/app.py reads off `bundle.calibrator` -- the object /score decides
 * with. There is no snapshot in this file. Retrain the calibrator and this picture
 * changes, which is the only way it can keep being true.
 */

const VB = { w: 680, h: 430 }
const M = { l: 52, r: 16, t: 14, b: 38 }
const IN = { w: VB.w - M.l - M.r, h: VB.h - M.t - M.b }

const sx = (x: number) => M.l + x * IN.w
const sy = (y: number) => M.t + (1 - y) * IN.h
const unsx = (px: number) => (px - M.l) / IN.w

const pct = (v: number) => `${(v * 100).toFixed(2)}%`
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

export function CalibratorExplainer({
  points,
  cal,
}: {
  points: PlotPoint[]
  cal: CalibratorResponse
}) {
  const anchor = points[0]

  // The probe starts ON the first applicant, so the first thing a reader sees is
  // a real point; dragging is what separates the two.
  const [probeRaw, setProbeRaw] = React.useState(anchor.p_raw)
  React.useEffect(() => setProbeRaw(anchor.p_raw), [anchor.p_raw])

  const path = React.useMemo(
    () => polyline(cal).map(([x, y]) => `${sx(x).toFixed(3)},${sy(y).toFixed(3)}`).join(" "),
    [cal],
  )
  const ramps = React.useMemo(() => rampMidpoints(cal), [cal])
  const levels = React.useMemo(() => reachableOutputs(cal), [cal])

  const probe = evaluate(cal, probeRaw)
  const probeDecision = decide(probe.p_cal, cal.threshold)

  // Each plotted applicant's block, read off the LIVE curve. Two applicants share
  // a step exactly when these agree -- which is the compare view's headline.
  const blocks = points.map((p) => evaluate(cal, p.p_raw))
  const shared =
    points.length > 1 &&
    blocks[0].block !== null &&
    blocks.every((b) => b.block === blocks[0].block)
      ? blocks[0]
      : null

  // The applicants' own figures are never recomputed -- this reading exists only
  // to CHECK that the curve being drawn is the curve that produced the decisions
  // above: same idiom as tier 4's additivity sum. It can only pass; it is here to
  // be checked. If it ever failed, the plot would be describing a different
  // artifact than the one that decided, and it says so rather than drawing on.
  const reproduces = points.every(
    (p, i) => Math.abs(blocks[i].p_cal - p.p_calibrated) < 1e-12,
  )

  const svgRef = React.useRef<SVGSVGElement>(null)
  const seek = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return
    setProbeRaw(clamp01(unsx(((clientX - rect.left) / rect.width) * VB.w)))
  }

  const jump = (dir: 1 | -1) => {
    const next =
      dir === 1
        ? ramps.find((m) => m > probeRaw + 1e-9)
        : [...ramps].reverse().find((m) => m < probeRaw - 1e-9)
    if (next !== undefined) setProbeRaw(next)
  }

  const yMax = levels[levels.length - 1]
  const colourOf = (d: "APPROVE" | "REJECT") =>
    d === "APPROVE" ? "var(--color-approve)" : "var(--color-reject)"
  const probeColour = colourOf(probeDecision)

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>
          {points.length > 1
            ? "Where A and B land on the calibrator"
            : "Why that probability, and not a nearby one"}
        </CardTitle>
        <span className="tnum text-[11px] text-faint">
          {cal.n_distinct_y} levels · {cal.n_knots} knots
        </span>
      </CardHeader>

      <CardContent className="space-y-4">
        {points.length > 1 ? (
          <p className="text-[12px] leading-relaxed text-muted">
            The calibrator is a step function — {cal.n_distinct_y} flat blocks joined by{" "}
            {cal.n_distinct_y - 1} near-vertical ramps, drawn below exactly as it is. Two
            applicants with <em>different</em> raw scores land on the <em>same</em> calibrated
            probability whenever they fall on the same block.{" "}
            {shared ? (
              <span className="text-fg">
                A and B do: both sit on block #{shared.block}, so their probabilities are
                identical no matter how far apart their raw scores are.
              </span>
            ) : (
              <span className="text-fg">
                A and B do not: they sit on different blocks, so their probabilities differ.
              </span>
            )}
          </p>
        ) : (
          <p className="text-[12px] leading-relaxed text-muted">
            The <span className="text-fg">{pct(anchor.p_calibrated)}</span> above is not a point
            on a smooth curve. It is one of{" "}
            <span className="text-fg">{cal.n_distinct_y} reachable values</span> — the output of
            the shipped isotonic calibrator, drawn below exactly as it is:{" "}
            {cal.n_distinct_y} flat blocks joined by {cal.n_distinct_y - 1} near-vertical ramps.
            Your applicant sits on one of those blocks.
          </p>
        )}

        {/* ---------------------------------------------------------------- */}
        <figure className="rounded-md border border-border bg-surface-2/30 p-2">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${VB.w} ${VB.h}`}
            className="w-full cursor-crosshair touch-none select-none"
            role="img"
            aria-label={`The shipped isotonic calibrator: ${cal.n_distinct_y} discrete levels mapping p_raw to p_calibrated.`}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              seek(e.clientX)
            }}
            onPointerMove={(e) => {
              if (e.buttons === 1) seek(e.clientX)
            }}
          >
            {/* --- the two clipped tails. out_of_bounds="clip": everything out
                    here returns ONE value. The right tail is 40% of the raw axis
                    and it is the finding, not an edge case. --- */}
            <rect
              x={sx(0)} y={M.t} width={sx(cal.x_min) - sx(0)} height={IN.h}
              fill="var(--color-surface-2)" opacity={0.55}
            />
            <rect
              x={sx(cal.x_max)} y={M.t} width={sx(1) - sx(cal.x_max)} height={IN.h}
              fill="var(--color-surface-2)" opacity={0.55}
            />
            <text
              x={(sx(cal.x_max) + sx(1)) / 2} y={M.t + 14}
              textAnchor="middle" fontSize={9.5} fill="var(--color-faint)"
            >
              clipped — {((1 - cal.x_max) * 100).toFixed(0)}% of the axis, one output
            </text>

            {/* --- above y_max: unreachable. The calibrator cannot emit it. --- */}
            <rect
              x={M.l} y={M.t} width={IN.w} height={sy(yMax) - M.t}
              fill="var(--color-bg)" opacity={0.5}
            />
            <line
              x1={M.l} x2={M.l + IN.w} y1={sy(yMax)} y2={sy(yMax)}
              stroke="var(--color-border)" strokeWidth={1} strokeDasharray="2 3"
            />
            <text x={M.l + 6} y={sy(yMax) - 5} fontSize={9.5} fill="var(--color-faint)">
              unreachable — the calibrator never outputs above {pct(yMax)}
            </text>

            {/* --- axes --- */}
            {[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => (
              <g key={`x${t}`}>
                <line
                  x1={sx(t)} x2={sx(t)} y1={M.t} y2={M.t + IN.h}
                  stroke="var(--color-border)" strokeWidth={0.5} opacity={0.5}
                />
                <text
                  x={sx(t)} y={M.t + IN.h + 14} textAnchor="middle"
                  fontSize={10} fill="var(--color-faint)" className="tnum"
                >
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            {[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => (
              <g key={`y${t}`}>
                <line
                  x1={M.l} x2={M.l + IN.w} y1={sy(t)} y2={sy(t)}
                  stroke="var(--color-border)" strokeWidth={0.5} opacity={0.5}
                />
                <text
                  x={M.l - 8} y={sy(t) + 3.5} textAnchor="end"
                  fontSize={10} fill="var(--color-faint)" className="tnum"
                >
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            <text
              x={M.l + IN.w / 2} y={VB.h - 4} textAnchor="middle"
              fontSize={10.5} fill="var(--color-muted)"
            >
              p_raw — the model's own output, sigmoid(margin)
            </text>
            <text
              transform={`translate(12, ${M.t + IN.h / 2}) rotate(-90)`}
              textAnchor="middle" fontSize={10.5} fill="var(--color-muted)"
            >
              p_calibrated — the quantity decided on
            </text>

            {/* --- the decision threshold. The value /calibrator RETURNED, not a
                    guessed 0.25: SELECTED_THRESHOLD is 0.25000000000000006. --- */}
            <line
              x1={M.l} x2={M.l + IN.w} y1={sy(cal.threshold)} y2={sy(cal.threshold)}
              stroke="var(--color-reject)" strokeWidth={1} strokeDasharray="4 3" opacity={0.7}
            />
            <text
              x={M.l + IN.w - 4} y={sy(cal.threshold) - 5} textAnchor="end"
              fontSize={9.5} fill="var(--color-reject)" opacity={0.85}
            >
              decision threshold {pct(cal.threshold)} — at or above, reject
            </text>

            {/* --- THE CALIBRATOR. Every knot, in order, joined by straight
                    segments, because that is what a piecewise-linear function IS.
                    Nothing resampled or smoothed: the ramps come out ~1px wide
                    because they ARE 9e-09 to 1.5e-03 wide. --- */}
            <polyline
              points={path} fill="none" stroke="var(--color-fg)" strokeWidth={1.5}
              strokeLinejoin="miter" vectorEffect="non-scaling-stroke"
            />

            {/* --- the block A and B SHARE, if they share one. Emphasised in the
                    curve's own colour: it is not a new fact, it is the segment
                    both applicants are standing on, made impossible to miss. --- */}
            {shared && (
              <line
                x1={sx(shared.span[0])} x2={sx(shared.span[1])}
                y1={sy(shared.p_cal)} y2={sy(shared.p_cal)}
                stroke="var(--color-fg)" strokeWidth={5} opacity={0.28} strokeLinecap="round"
              />
            )}

            {/* --- the probe --- */}
            <line
              x1={sx(probe.p_raw)} x2={sx(probe.p_raw)} y1={sy(probe.p_cal)} y2={M.t + IN.h}
              stroke={probeColour} strokeWidth={0.75} strokeDasharray="2 2" opacity={0.55}
            />
            <line
              x1={M.l} x2={sx(probe.p_raw)} y1={sy(probe.p_cal)} y2={sy(probe.p_cal)}
              stroke={probeColour} strokeWidth={0.75} strokeDasharray="2 2" opacity={0.55}
            />
            <circle cx={sx(probe.p_raw)} cy={sy(probe.p_cal)} r={4} fill={probeColour} />

            {/* --- the applicants. Read off the API response, never recomputed.
                    A and B are told apart by SHAPE and by their letter, not by an
                    arbitrary hue: colour on this page already means approve/reject
                    and must keep meaning that. When A approves and B rejects you
                    see green vs red, which is the fact that matters; when they
                    share a step they coincide, which is the other one. --- */}
            {points.map((p, i) => {
              const c = colourOf(p.decision)
              const [cx, cy] = [sx(p.p_raw), sy(p.p_calibrated)]
              const dy = i === 0 ? -11 : 20 // labels apart when the markers coincide
              return (
                <g key={p.id || "single"}>
                  {i === 0 ? (
                    <circle cx={cx} cy={cy} r={7} fill="none" stroke={c} strokeWidth={1.75} />
                  ) : (
                    <rect
                      x={cx - 6.5} y={cy - 6.5} width={13} height={13}
                      fill="none" stroke={c} strokeWidth={1.75}
                    />
                  )}
                  <circle cx={cx} cy={cy} r={1.75} fill={c} />
                  <text x={cx + 12} y={cy + dy} fontSize={10} fill={c}>
                    {p.label} · {pct(p.p_calibrated)}
                  </text>
                </g>
              )
            })}
          </svg>
        </figure>

        {/* ---------------------------------------------------------------- */}
        <div className="space-y-2.5">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => jump(-1)}
              title="Jump to the previous ramp"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border text-faint transition-colors hover:border-border-strong hover:text-fg"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>

            <input
              type="range"
              min={0}
              max={1}
              step={0.000001}
              value={probeRaw}
              onChange={(e) => setProbeRaw(Number(e.target.value))}
              aria-label="Explore p_raw"
              className="cal-slider flex-1"
            />

            <button
              type="button"
              onClick={() => jump(1)}
              title="Jump to the next ramp"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border text-faint transition-colors hover:border-border-strong hover:text-fg"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>

            {points.map((p) => (
              <button
                key={p.id || "single"}
                type="button"
                onClick={() => setProbeRaw(p.p_raw)}
                title={`Return the probe to ${p.label}`}
                className="flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5 text-[11px] text-faint transition-colors hover:border-border-strong hover:text-fg"
              >
                <Crosshair className="h-3.5 w-3.5" />
                {p.label}
              </button>
            ))}
          </div>

          <Readout probe={probe} decision={probeDecision} cal={cal} nRamps={ramps.length} />
        </div>

        {/* ---------------------------------------------------------------- */}
        <div className="space-y-2 border-t border-border pt-3">
          <p className="text-[11px] leading-relaxed text-faint">
            Two applicants with different raw scores land on the identical calibrated
            probability whenever they share a block —{" "}
            <span className="text-muted">not a rounding artifact, but the shipped function</span>
            . That is also why the contributions are given in log-odds and never converted to
            probability points: on a flat block <code className="text-muted">dp_cal/dp_raw</code>{" "}
            is exactly 0, so there is no quantity to convert.{" "}
            <code className="text-muted">docs/explainability.md</code> measures the flat fraction
            of the reject region at <span className="text-muted">99.31%</span> of the p_raw axis —
            a statement about the axis, not about how many applicants fall on it.
          </p>

          <div
            className={cn(
              "flex items-start gap-2 text-[11px]",
              reproduces ? "text-faint" : "text-reject",
            )}
          >
            {reproduces ? (
              <>
                <Check className="mt-0.5 h-3 w-3 shrink-0 text-approve" />
                <span>
                  The curve above is the curve that decided:{" "}
                  <code className="text-muted">GET /calibrator</code>'s knots, read at{" "}
                  {points.length > 1 ? "each applicant's" : "this applicant's"} own{" "}
                  <code className="text-muted">p_raw</code>, reproduce the{" "}
                  <code className="text-muted">p_calibrated</code> the API returned — to within
                  1e-12. Nothing here is a snapshot; retrain the calibrator and this plot changes
                  with it.
                </span>
              </>
            ) : (
              <span>
                The plotted curve does <em>not</em> reproduce the API's own p_calibrated. The
                drawing and the decision disagree — trust the decision, not this plot.
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// ===========================================================================

function Readout({
  probe,
  decision,
  cal,
  nRamps,
}: {
  probe: Reading
  decision: "APPROVE" | "REJECT"
  cal: CalibratorResponse
  nRamps: number
}) {
  const [x0, x1] = probe.span
  const width = x1 - x0
  // A ramp straddles the threshold when it starts below it and lands at or above:
  // crossing THIS sliver of p_raw is what turns an approval into a rejection.
  const flipsDecision =
    probe.region === "ramp" &&
    probe.yspan[0] < cal.threshold &&
    probe.yspan[1] >= cal.threshold

  return (
    <div className="rounded-md border border-border bg-surface-2/50 px-3 py-2.5">
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
        <Cell label="p_raw" value={probe.p_raw.toFixed(6)} />
        <Cell label="p_calibrated" value={pct(probe.p_cal)} strong />
        <Cell
          label="local slope"
          value={probe.slope === 0 ? "0" : probe.slope.toFixed(1)}
          strong={probe.slope !== 0}
        />
        <Cell
          label="decision"
          value={decision}
          colour={decision === "APPROVE" ? "text-approve" : "text-reject"}
          strong
        />
      </div>

      <p className="mt-2.5 border-t border-border pt-2 text-[11px] leading-relaxed text-muted">
        {probe.region === "flat" && (
          <>
            <span className="text-fg">Slope 0.</span> You are on flat block{" "}
            <span className="tnum">#{probe.block}</span>, spanning p_raw{" "}
            <span className="tnum">
              {x0.toFixed(6)} – {x1.toFixed(6)}
            </span>{" "}
            (<span className="tnum">{width.toExponential(2)}</span> wide). Move p_raw anywhere
            inside it and p_calibrated changes by{" "}
            <span className="text-fg">exactly nothing</span> — which is why “this feature is
            worth N points of probability” has no value to compute here. The derivative of the
            decided quantity with respect to the model's own output is zero.
          </>
        )}

        {probe.region === "ramp" && (
          <>
            <span className="text-fg">You are on a ramp</span> — one of {nRamps}, and each is
            narrower than a single pixel of the plot above, so you could not have dragged onto it
            by accident. That impossibility <em>is</em> the finding. The whole jump from{" "}
            <span className="tnum">{pct(probe.yspan[0])}</span> to{" "}
            <span className="tnum">{pct(probe.yspan[1])}</span> happens across{" "}
            <span className="tnum text-fg">{(width * 1e6).toFixed(1)} parts per million</span> of
            p_raw, at a slope of <span className="tnum">{probe.slope.toFixed(1)}</span>.
            {flipsDecision && (
              <>
                {" "}
                <span className="text-reject">
                  This is a ramp the decision flips on: p_raw crossing it moves the applicant
                  from approve to reject.
                </span>
              </>
            )}
          </>
        )}

        {probe.region === "clipped-high" && (
          <>
            <span className="text-fg">Out of the fitted domain, above x_max.</span>{" "}
            <code>IsotonicRegression(out_of_bounds="clip")</code> pins the output at{" "}
            <span className="tnum text-fg">{pct(probe.p_cal)}</span>. Every p_raw from{" "}
            <span className="tnum">{cal.x_max.toFixed(6)}</span> to{" "}
            <span className="tnum">1.0</span> —{" "}
            <span className="text-fg">
              {((1 - cal.x_max) * 100).toFixed(1)}% of the raw axis
            </span>{" "}
            — collapses to that one value. Slope 0 across all of it.
          </>
        )}

        {probe.region === "clipped-low" && (
          <>
            <span className="text-fg">Out of the fitted domain, below x_min.</span>{" "}
            <code>IsotonicRegression(out_of_bounds="clip")</code> pins the output at{" "}
            <span className="tnum text-fg">{pct(probe.p_cal)}</span> for every p_raw below{" "}
            <span className="tnum">{cal.x_min.toFixed(6)}</span>. Slope 0.
          </>
        )}
      </p>
    </div>
  )
}

function Cell({
  label,
  value,
  strong,
  colour,
}: {
  label: string
  value: string
  strong?: boolean
  colour?: string
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-faint">{label}</div>
      <div
        className={cn(
          "tnum mt-0.5 text-sm",
          colour ?? (strong ? "text-fg" : "text-muted"),
          strong && "font-semibold",
        )}
      >
        {value}
      </div>
    </div>
  )
}
