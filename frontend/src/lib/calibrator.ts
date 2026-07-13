/**
 * Reading the shipped isotonic calibrator.
 *
 * Every function here takes a CalibratorResponse -- the LIVE one, off GET
 * /calibrator, which serving/app.py reads from `bundle.calibrator`: the same
 * object /score composes its decisions with. Nothing in this file has a knot,
 * a threshold or a domain baked into it. If the calibrator is retrained, the
 * curve this draws changes with it, because it never had its own copy.
 *
 * WHAT IS COMPUTED HERE, AND WHAT IS NOT
 *
 * The applicant's own (p_raw, p_calibrated) is NEVER recomputed. It is read off
 * the ScoreResponse. `evaluate()` exists for the exploratory probe -- the point
 * the reader drags -- and it is not a score: it is a READING of the server's own
 * curve at a hypothetical p_raw. The distinction matters enough that the UI says
 * so out loud.
 *
 * The reading is nonetheless exact. `evaluate()` transcribes
 * IsotonicRegression(out_of_bounds="clip").predict: clip outside [x_min, x_max],
 * linear interpolation between knots within. Checked against the real shipped
 * calibrator over 40,212 probes -- every knot, every block and ramp midpoint,
 * and uniform draws across the whole [0, 1] axis -- max |error| 5.55e-17. The UI
 * re-checks it live at the applicant's own p_raw and says so on screen, so a
 * reader does not have to take that paragraph's word for it.
 */

import type { CalibratorResponse, ScoreResponse } from "@/lib/api"

/** Where a p_raw falls, in the calibrator's own terms. */
export type Region = "clipped-low" | "flat" | "ramp" | "clipped-high"

/**
 * An applicant, as the calibrator plot needs them.
 *
 * p_raw and p_calibrated are the API's OWN numbers, off the ScoreResponse --
 * copied, never recomputed. Only the exploratory probe is evaluated in the
 * browser, and that is a reading of the server's curve rather than a score.
 */
export interface PlotPoint {
  /** "" in single mode; "A" / "B" in compare. Drawn beside the marker. */
  id: string
  label: string
  p_raw: number
  p_calibrated: number
  decision: "APPROVE" | "REJECT"
}

export function toPlotPoint(r: ScoreResponse, id: string, label: string): PlotPoint {
  return {
    id,
    label,
    p_raw: r.p_raw,
    p_calibrated: r.p_calibrated,
    decision: r.decision,
  }
}

export interface Reading {
  p_raw: number
  p_cal: number
  /** dp_cal/dp_raw. Exactly 0 on a flat block and on both clip rays. */
  slope: number
  region: Region
  /** The span of p_raw over which this reading's regime holds. For a flat block:
   *  the block. For a ramp: the ramp. For a clip ray: the whole clipped tail. */
  span: [number, number]
  /** What p_cal does across `span`. Equal endpoints on a flat block and on both
   *  clip rays; on a ramp these are the two levels it jumps between -- which is
   *  what lets the UI say whether THIS ramp is the one that flips the decision. */
  yspan: [number, number]
  /** 0..51 on a flat block; null on a ramp or a clip ray. */
  block: number | null
}

/**
 * IsotonicRegression(out_of_bounds="clip").predict, transcribed.
 *
 * The knots arrive in equal-valued PAIRS: (x[2i], x[2i+1]) share one y and are a
 * flat block; x[2i+1] -> x[2i+2] is the ramp to the next block. So 104 knots are
 * 52 flat blocks joined by 51 ramps. `region` is derived from y[i] === y[i+1]
 * rather than from index parity, so it stays correct even if a retrained
 * calibrator does not honour that pairing.
 */
export function evaluate(cal: CalibratorResponse, p_raw: number): Reading {
  const { x_thresholds: X, y_thresholds: Y, x_min, x_max } = cal

  // out_of_bounds="clip". Not an edge case to tidy away -- 40.3% of the [0, 1]
  // raw axis lies above x_max and every point of it returns the SAME y_max.
  if (p_raw <= x_min) {
    return {
      p_raw, p_cal: Y[0], slope: 0, region: "clipped-low",
      span: [0, x_min], yspan: [Y[0], Y[0]], block: null,
    }
  }
  if (p_raw >= x_max) {
    const yMax = Y[Y.length - 1]
    return {
      p_raw, p_cal: yMax, slope: 0, region: "clipped-high",
      span: [x_max, 1], yspan: [yMax, yMax], block: null,
    }
  }

  let lo = 0
  let hi = X.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (X[mid] <= p_raw) lo = mid
    else hi = mid
  }

  const [x0, x1, y0, y1] = [X[lo], X[lo + 1], Y[lo], Y[lo + 1]]
  const width = x1 - x0
  // A zero-width interval cannot happen in the shipped calibrator (the narrowest
  // ramp is 9.13e-09 wide, not 0), but a retrained one is not this one's problem
  // to guarantee, so it is handled rather than assumed away.
  const slope = width === 0 ? 0 : (y1 - y0) / width
  const p_cal = width === 0 ? y1 : y0 + slope * (p_raw - x0)

  const flat = y0 === y1
  return {
    p_raw,
    p_cal,
    slope: flat ? 0 : slope,
    region: flat ? "flat" : "ramp",
    span: [x0, x1],
    yspan: [y0, y1],
    block: flat && lo % 2 === 0 ? lo / 2 : null,
  }
}

/**
 * The polyline of the calibrator, in DATA space, over the full [0, 1] p_raw axis.
 *
 * Every one of the 104 knots is emitted, in order, plus one point at each end for
 * the clip rays. Consecutive points are joined by straight segments -- which is
 * not a rendering choice, it is what the function IS: piecewise-linear through
 * its knots. The flat blocks come out flat and the ramps come out near-vertical
 * because they ARE near-vertical (the widest is 1.5e-03, the narrowest 9.1e-09).
 *
 * Nothing is resampled, smoothed, or thinned. A curve-fitted rendering of this
 * artifact would be a lie about the artifact, and the artifact's discreteness is
 * the entire finding.
 */
export function polyline(cal: CalibratorResponse): Array<[number, number]> {
  const { x_thresholds: X, y_thresholds: Y } = cal
  const last = Y.length - 1
  return [
    [0, Y[0]], // left clip ray
    ...X.map((x, i): [number, number] => [x, Y[i]]),
    [1, Y[last]], // right clip ray
  ]
}

/** The midpoint of every ramp, ascending. 51 of them for the shipped calibrator.
 *  Each is narrower than one pixel of the plot, so they exist to be JUMPED to --
 *  a reader cannot drag onto one, and that impossibility is the point. */
export function rampMidpoints(cal: CalibratorResponse): number[] {
  const { x_thresholds: X, y_thresholds: Y } = cal
  const mids: number[] = []
  for (let i = 0; i + 1 < X.length; i++) {
    if (Y[i] !== Y[i + 1]) mids.push((X[i] + X[i + 1]) / 2)
  }
  return mids
}

/** The distinct reachable outputs, ascending. The calibrator can return these 52
 *  values and nothing else -- every other probability in [0, 1] is unreachable. */
export function reachableOutputs(cal: CalibratorResponse): number[] {
  return [...new Set(cal.y_thresholds)].sort((a, b) => a - b)
}

/** The decision /score would reach for a calibrated probability. serving/app.py
 *  compares p_calibrated >= threshold -- at or above rejects. */
export function decide(p_cal: number, threshold: number): "APPROVE" | "REJECT" {
  return p_cal >= threshold ? "REJECT" : "APPROVE"
}
