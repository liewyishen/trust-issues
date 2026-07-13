import { Loader2 } from "lucide-react"
import * as React from "react"

import { Failure } from "@/components/Failure"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  getDrift,
  RouteNotAvailableError,
  type DriftResponse,
  type FeatureDrift,
} from "@/lib/api"
import { cn } from "@/lib/utils"

/*
 * THE MONITOR, WITH A KNOB ON IT.
 *
 * Drag the market's mean FICO down from 700 and watch pipelines/drift_check.py --
 * the monitor that actually runs, not a demo metric invented to look good -- pick
 * the shift up. The claim being demonstrated is NOT "the number moved when I moved
 * the input"; anything can do that. It is PRECISION: fico_n fires, and dti_n,
 * which the knob never touches, stays exactly where it was. A monitor that lit up
 * on both would not be detecting drift. It would be detecting that something,
 * somewhere, changed.
 *
 * NOTHING HERE IS COMPUTED IN THE BROWSER. Not the PSI, not the KS, not the alarm
 * verdict, not the thresholds the bars are measured against. This client could not
 * compute a PSI honestly even if it wanted to -- PSI is defined against the
 * reference population's quantile bin edges, and those live inside the monitor.
 * Every number below came off POST /drift.
 *
 * The two things the browser DOES work out are geometry and arithmetic, and both
 * exist to be checked rather than believed:
 *   - whether a bar extends past the line drawn across it (both the bar's value
 *     and the line's position come from the server; "is it past it" is a fact
 *     about the picture, not a second opinion about the alarm), and
 *   - the delta between a reading and the baseline reading, which is what turns
 *     "dti_n is a negative control" from a caption into something you can watch
 *     stay at 0.0000.
 */

// ---------------------------------------------------------------------------
// The knob's range. 700 is the REFERENCE population's mean -- the response's own
// reference_mean_fico -- so the slider opens at "no drift" and every drag away
// from it is a drift you introduced. The range is asymmetric on purpose: the
// interesting direction is DOWN (credit quality slipping), and 640 is far enough
// to take PSI past 1.3. Upward is included to 710 because a market getting BETTER
// is also drift, and a monitor that only fired one way would be a worse monitor.
// ---------------------------------------------------------------------------
const KNOB_MIN = 640
const KNOB_MAX = 710
const KNOB_DEFAULT = 700

/**
 * The PSI bar's axis. PSI is unbounded above, so an axis is a display choice and
 * has to be made honestly: this one is ANCHORED TO THE SLIDER'S OWN REACH. The
 * furthest this knob goes (640) produces PSI 1.3842 -- measured, not guessed --
 * so 1.5 leaves the bar room at the extreme instead of pinning it to the edge and
 * making 1.4 look the same as 5.0. If a value ever exceeds it the bar clamps and
 * says so, rather than silently rendering as "full".
 */
const PSI_AXIS_MAX = 1.5

/**
 * KS's axis is not a choice. The Kolmogorov-Smirnov statistic is the maximum
 * displacement between two CDFs, so it is bounded on [0, 1] by construction.
 */
const KS_AXIS_MAX = 1.0

/** Fire on settle. Long enough that a drag across the range is a handful of
 *  requests rather than seventy, short enough that letting go feels immediate. */
const SETTLE_MS = 140

/** What each monitored column is FOR in this demo. The roles are not decoration:
 *  the whole argument depends on the reader knowing which column the knob drives
 *  and which one it does not touch. A column the server started monitoring that
 *  this map has never heard of renders without a role rather than being given a
 *  guessed one. */
const ROLE: Record<string, string> = {
  fico_n: "the knob drives this",
  dti_n: "negative control — the knob does not touch it",
}

export function DriftMonitor() {
  const [knob, setKnob] = React.useState(KNOB_DEFAULT)
  const [data, setData] = React.useState<DriftResponse | null>(null)
  const [error, setError] = React.useState<unknown>(null)
  const [pending, setPending] = React.useState(true)

  /**
   * The reading at the reference mean, captured once and never overwritten. It is
   * what every later reading is measured against, and it is what lets the negative
   * control PROVE itself: dti_n's Δ against this baseline stays 0.0000 at every
   * setting of the knob, which is a stronger statement than a caption asserting
   * that it should.
   */
  const baseline = React.useRef<DriftResponse | null>(null)

  /** Responses can land out of order when a drag outruns the network. Only the
   *  newest request is allowed to write state; an older one that arrives late is
   *  dropped, because rendering it would show a reading for a knob position the
   *  user has already left. */
  const latest = React.useRef(0)
  const warmed = React.useRef(false)

  React.useEffect(() => {
    // The mount call is NOT debounced. It is the warm-up -- the handler imports
    // the monitor lazily, so the first request pays ~0.40s (see getDrift) -- and
    // it is also the baseline the chart needs anyway. Sending it immediately means
    // the tab is live in well under a second, and by the time anyone has reached
    // for the slider the import is long since paid.
    const delay = warmed.current ? SETTLE_MS : 0
    const timer = setTimeout(() => {
      warmed.current = true
      const mine = ++latest.current
      setPending(true)
      getDrift(knob).then(
        (r) => {
          if (latest.current !== mine) return
          if (baseline.current === null && r.mean_fico === r.reference_mean_fico) {
            baseline.current = r
          }
          setData(r)
          setError(null)
          setPending(false)
        },
        (e) => {
          if (latest.current !== mine) return
          setError(e)
          setPending(false)
        },
      )
    }, delay)
    return () => clearTimeout(timer)
  }, [knob])

  // 404 is the documented dev-only state, and it is terminal for this whole view:
  // there is no partial version of a drift monitor. Everything else (a dead API, a
  // 422) is rendered by the shared Failure component, as itself.
  if (error instanceof RouteNotAvailableError) return <DriftUnavailable />

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex items-center justify-between">
          <CardTitle>Population drift monitor</CardTitle>
          <Badge>pipelines/drift_check.py</Badge>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-[12px] leading-relaxed text-muted">
            A different question from the two tabs beside this one. Those score{" "}
            <em>one applicant</em>. This watches a whole <em>population</em> move, and asks the
            monitor that actually runs whether it noticed.
          </p>

          <Knob
            value={knob}
            onChange={setKnob}
            pending={pending}
            data={data}
          />
        </CardContent>
      </Card>

      {error !== null && (
        <div className="space-y-2">
          {data !== null && (
            <p className="text-[11px] text-faint">
              The panels below are the last reading that succeeded — they are stale, not live.
            </p>
          )}
          <Failure error={error} />
        </div>
      )}

      {data === null ? (
        <FirstLoad pending={pending} />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {/* Server key order, which is drift_check's MONITORED order: the column
                the knob drives first, the control beside it. Rendered side by side
                and never stacked on a wide screen, because the CONTRAST is the
                claim -- fico_n going red next to dti_n staying green is the whole
                thing, and it does not survive being scrolled past. */}
            {Object.entries(data.features).map(([column, feature]) => (
              <FeaturePanel
                key={column}
                column={column}
                feature={feature}
                thresholds={data.thresholds}
                baseline={baseline.current?.features[column] ?? null}
                stale={pending}
              />
            ))}
          </div>

          <Alarms data={data} />
          <Provenance data={data} />
        </>
      )}
    </div>
  )
}

// ===========================================================================

function Knob({
  value,
  onChange,
  pending,
  data,
}: {
  value: number
  onChange: (v: number) => void
  pending: boolean
  data: DriftResponse | null
}) {
  const atReference = data !== null && value === data.reference_mean_fico

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <label htmlFor="knob" className="text-[11px] uppercase tracking-wider text-faint">
          Market FICO mean
        </label>
        <div className="flex items-baseline gap-2">
          {pending && <Loader2 className="h-3 w-3 animate-spin text-faint" />}
          <span className="tnum text-2xl font-semibold leading-none text-fg">{value}</span>
        </div>
      </div>

      <input
        id="knob"
        type="range"
        min={KNOB_MIN}
        max={KNOB_MAX}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="drift-slider w-full"
      />

      <div className="tnum flex justify-between text-[10px] text-faint">
        <span>{KNOB_MIN}</span>
        <span className={cn(atReference && "text-muted")}>
          {KNOB_DEFAULT} · reference
        </span>
        <span>{KNOB_MAX}</span>
      </div>

      {data !== null && (
        <dl className="grid grid-cols-2 gap-3 border-t border-border pt-3 text-[11px] sm:grid-cols-4">
          {/* The knob did what it says. observed_* is what the batch actually DREW
              out of MockBureau, against what was asked for -- if the two ever
              parted company (they do, near the FICO band's edges, where the mock
              clips every draw into [300, 900]) this is where you would see it. */}
          <Figure label="reference drew" value={data.observed_mean_fico_reference.toFixed(2)} />
          <Figure label="current drew" value={data.observed_mean_fico_current.toFixed(2)} />
          <Figure
            label="shift"
            value={(
              data.observed_mean_fico_current - data.observed_mean_fico_reference
            ).toFixed(2)}
            emphasis={!atReference}
          />
          <Figure
            label="applicants"
            value={`${data.n_reference.toLocaleString()} → ${data.n_current.toLocaleString()}`}
          />
        </dl>
      )}
    </div>
  )
}

function Figure({
  label,
  value,
  emphasis = false,
}: {
  label: string
  value: string
  emphasis?: boolean
}) {
  return (
    <div>
      <dt className="text-faint">{label}</dt>
      <dd className={cn("tnum mt-0.5", emphasis ? "text-fg" : "text-muted")}>{value}</dd>
    </div>
  )
}

// ===========================================================================

function FeaturePanel({
  column,
  feature,
  thresholds,
  baseline,
  stale,
}: {
  column: string
  feature: FeatureDrift
  thresholds: Record<string, number>
  baseline: FeatureDrift | null
  stale: boolean
}) {
  const psiCrossed = feature.psi > thresholds.psi
  const ksCrossed = feature.ks > thresholds.ks

  /**
   * The one thing a "PSI + alarm" panel would get visibly wrong.
   *
   * The monitor alarms on EITHER signal, and the two do not cross together: KS is
   * the more sensitive of the pair. Measured across this very slider in 1-point
   * steps: at mean_fico 691 KS is 0.0980 and the feature is QUIET; at 690 KS is
   * 0.1045 and it ALARMS. PSI does not cross 0.25 until 676 (0.2520 -- at 677 it
   * is still 0.2352). So the band where the feature is ALARMED with its PSI bar
   * sitting comfortably under the threshold is 677..690.
   *
   * Rendering `alarmed` next to a PSI bar and calling it a day would put a red
   * ALARM badge above a bar that has visibly crossed nothing, and the honest
   * reader would conclude the UI was broken. It is not: the alarm is real, and the
   * signal that raised it is the one beside it. So the panel says which.
   */
  const alarmedOnKsAlone = feature.alarmed && !psiCrossed && ksCrossed

  return (
    <Card className={cn(stale && "opacity-60 transition-opacity")}>
      <CardHeader className="flex items-start justify-between">
        <div>
          <CardTitle className="tnum">{column}</CardTitle>
          <p className="mt-0.5 text-[11px] text-faint">{ROLE[column] ?? "monitored column"}</p>
        </div>
        {/* Colour carries meaning and nothing else: red IS the alarm state, and it
            is the same red /score uses for REJECT. The verdict is the server's --
            feature.alarmed, straight off evaluate_alarms() -- never a comparison
            this component made up. */}
        {feature.alarmed ? (
          <span className="rounded border border-reject/40 bg-reject-dim/40 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-reject">
            alarm
          </span>
        ) : (
          <span className="rounded border border-approve/30 bg-approve-dim/30 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-approve">
            quiet
          </span>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        <MetricBar
          name="PSI"
          value={feature.psi}
          threshold={thresholds.psi}
          axisMax={PSI_AXIS_MAX}
          crossed={psiCrossed}
          baseline={baseline?.psi ?? null}
        />
        <MetricBar
          name="KS"
          value={feature.ks}
          threshold={thresholds.ks}
          axisMax={KS_AXIS_MAX}
          crossed={ksCrossed}
          baseline={baseline?.ks ?? null}
        />

        {alarmedOnKsAlone && (
          <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2 text-[11px] leading-relaxed text-muted">
            <span className="text-fg">PSI has not crossed its line yet — KS has.</span> The
            monitor alarms on <em>either</em> signal, and KS is the more sensitive of the two, so
            it catches this shift first. The alarm is real; the bar that raised it is the one
            below.
          </p>
        )}

        {column === "dti_n" && baseline !== null && (
          <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2 text-[11px] leading-relaxed text-muted">
            MockBureau seeds this off a hash of the applicant id and{" "}
            <em>never off the knob</em>, and both populations draw the same ids — so every
            dti_n here is byte-identical to the baseline's, and its Δ is exactly{" "}
            <span className="tnum text-fg">0.0000</span> at every setting. That is not a caption
            asking to be believed. It is the number in the row above, and you can watch it not
            move.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

/**
 * One signal, its alarm line, and where it currently stands.
 *
 * `crossed` is geometry, not a verdict: the value and the threshold both came off
 * the server, and this only asks whether one exceeds the other so the bar can be
 * drawn in the colour that matches. The ALARM badge above is the server's verdict
 * and is never derived here.
 */
function MetricBar({
  name,
  value,
  threshold,
  axisMax,
  crossed,
  baseline,
}: {
  name: string
  value: number
  threshold: number
  axisMax: number
  crossed: boolean
  baseline: number | null
}) {
  const overflow = value > axisMax
  const fillPct = Math.min(value / axisMax, 1) * 100
  const linePct = (threshold / axisMax) * 100
  const delta = baseline === null ? null : value - baseline

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="uppercase tracking-wider text-faint">{name}</span>
        <div className="flex items-baseline gap-3">
          {delta !== null && (
            <span className="tnum text-faint">
              Δ {delta >= 0 ? "+" : "−"}
              {Math.abs(delta).toFixed(4)}
            </span>
          )}
          <span className={cn("tnum text-[13px]", crossed ? "text-reject" : "text-fg")}>
            {value.toFixed(4)}
          </span>
        </div>
      </div>

      <div className="relative h-2.5 overflow-hidden rounded-full border border-border bg-surface-2">
        <div
          className={cn(
            "h-full rounded-l-full transition-[width] duration-200",
            crossed ? "bg-reject" : "bg-approve",
          )}
          style={{ width: `${fillPct}%` }}
        />
        {/* The alarm line, drawn at the threshold the SERVER returned
            (DEFAULT_ALARM_THRESHOLDS). Not a guessed 0.25 -- for the same reason
            the calibrator chart draws its reject line at the threshold /calibrator
            returns rather than at the literal 0.25, which is a different float. */}
        <div
          className="absolute inset-y-0 w-px bg-border-strong"
          style={{ left: `${linePct}%` }}
          aria-hidden
        />
      </div>

      <div className="tnum flex justify-between text-[10px] text-faint">
        <span>0</span>
        <span>
          alarms above {threshold}
          {overflow && (
            <span className="ml-2 text-adverse">
              value {value.toFixed(4)} exceeds the {axisMax} axis — bar clamped
            </span>
          )}
        </span>
        <span>{axisMax}</span>
      </div>
    </div>
  )
}

// ===========================================================================

/**
 * evaluate_alarms()'s own strings, unedited.
 *
 * Including the one that embarrasses the demo. The dti tripwire fires at EVERY
 * setting of the knob, because MockBureau's dti_n is a uniform draw over
 * [0, 1000) and drift_check's tripwire probes for the real 2016+ DTI reporting
 * regime -- values in (100, 1000] -- so ~90% of the mock's DTI lands in that band
 * by construction. It is a mock artifact, not drift. Filtering it out of this list
 * would make the demo look cleaner and would be the single dishonest edit
 * available on this page.
 *
 * The split below is on the metric KEY each alarm string starts with, which is
 * how the alarm was named by the monitor -- never on searching the alarm's prose
 * for a column name. The tripwire's own message contains the words "dti_n" inside
 * its explanation, so a text search would file it under the negative control and
 * report dti_n as firing at every setting, inverting the one contrast this page
 * exists to show. (serving/schema.py's DriftResponse records the same trap.)
 */
function Alarms({ data }: { data: DriftResponse }) {
  const isDistribution = (a: string) => a.startsWith("psi_") || a.startsWith("ks_")
  const distribution = data.alarms.filter(isDistribution)
  const artifacts = data.alarms.filter((a) => !isDistribution(a))

  return (
    <Card>
      <CardHeader>
        <CardTitle>What the monitor raised</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {distribution.length === 0 ? (
          <p className="text-[12px] text-muted">
            No distribution alarm. At this setting the monitor sees nothing in fico_n or dti_n
            worth raising — which is the state it is <em>supposed</em> to be in when the market
            has not moved.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {distribution.map((a) => (
              <li
                key={a}
                className="tnum rounded border border-reject/30 bg-reject-dim/20 px-2.5 py-1.5 text-[11px] leading-relaxed text-reject"
              >
                {a}
              </li>
            ))}
          </ul>
        )}

        {artifacts.length > 0 && (
          <div className="space-y-1.5 border-t border-border pt-3">
            <ul className="space-y-1.5">
              {artifacts.map((a) => (
                <li
                  key={a}
                  className="tnum rounded border border-border bg-surface-2/50 px-2.5 py-1.5 text-[11px] leading-relaxed text-faint"
                >
                  {a}
                </li>
              ))}
            </ul>
            <p className="text-[11px] leading-relaxed text-muted">
              <span className="text-adverse">Disclosed artifact, not drift.</span> The monitor
              really did raise this — it is in the response, unedited — but it fires at{" "}
              <em>every</em> setting of the knob and reads identically at 700 and at 650, so it
              is not what the knob moves. <code className="text-faint">MockBureau</code>'s dti_n
              is a crude uniform draw over [0, 1000), and drift_check's tripwire probes for the
              real 2016+ DTI reporting regime — values in (100, 1000] — so ~90% of the mock's
              DTI lands in that band by construction. Dropping it from this list would make the
              demo look tidier and would be a lie.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function Provenance({ data }: { data: DriftResponse }) {
  return (
    <div className="space-y-3 rounded-md border border-border bg-surface px-4 py-3 text-[11px] leading-relaxed text-muted">
      <p>
        <span className="text-fg">This is the monitor, not a picture of one.</span> Every PSI, KS
        and alarm above is computed by <code className="text-faint">pipelines/drift_check.py</code>{" "}
        — the same functions the batch job calls, on the same{" "}
        <code className="text-faint">DEFAULT_ALARM_THRESHOLDS</code>. The browser computes no PSI;
        it could not do so honestly, since PSI is defined against the reference population's
        quantile bin edges and those live inside the monitor. The alarm lines are drawn at the
        thresholds the endpoint <em>returned</em>.
      </p>
      <p>
        <span className="text-fg">Drift monitoring in production is not a slider.</span> It is an
        offline scheduled job — <code className="text-faint">uv run python
        pipelines/drift_check.py</code>, logging to its own MLflow run — reading a real applicant
        stream. This endpoint drives that same computation over a synthetic MockBureau population
        so the thing can be <em>seen</em> moving. It is a demonstration of the monitor, not a
        deployment of it, and <code className="text-faint">POST /drift</code> is dev-only for
        exactly that reason.
      </p>
      <p className="text-faint">
        Windows are keyed <span className="tnum">{data.reference_year}</span> →{" "}
        <span className="tnum">{data.current_year}</span>. Those are monitor slice labels, not a
        claim about real LendingClub data from those years — <code>drift_metrics()</code> slices
        on an <code>issue_year</code> column, so the two synthetic batches need labels to be told
        apart.
      </p>
    </div>
  )
}

// ===========================================================================

function FirstLoad({ pending }: { pending: boolean }) {
  if (!pending) return null
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-8">
        <Loader2 className="h-4 w-4 animate-spin text-faint" />
        <div className="text-[12px] leading-relaxed text-muted">
          Drawing the reference population and running the monitor…
          <span className="block text-faint">
            The first call costs about 0.4s — the endpoint imports the drift pipeline lazily,
            because importing it at module scope would pull mlflow and metaflow into the serving
            image and kill the container at boot. Every call after this one is ~50ms.
          </span>
        </div>
      </CardContent>
    </Card>
  )
}

/**
 * 404 from /drift. Not a bug, and not a thing to paper over.
 *
 * The route is gated on DRIFT_DEMO_AVAILABLE (serving/app.py): the monitor lives
 * in pipelines/, which the slim serving image neither copies nor installs the
 * dependencies for. So a deployed container answers 404 here, honestly, and this
 * is what that looks like. The one response this page must never make is to draw
 * the chart anyway from numbers it invented -- a fabricated monitor is worse than
 * an absent one, and this whole tab exists to argue that computed-here numbers
 * cannot be trusted.
 */
function DriftUnavailable() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>The drift monitor is not available on this deployment</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-[12px] leading-relaxed text-muted">
        <p>
          <code className="text-faint">POST /drift</code> returned{" "}
          <span className="tnum text-fg">404</span>. The route is not mounted here, and that is a
          deliberate state rather than a failure.
        </p>
        <p>
          The monitor lives in <code className="text-faint">pipelines/drift_check.py</code>, which
          imports mlflow and metaflow. The serving image copies neither{" "}
          <code className="text-faint">pipelines/</code> nor{" "}
          <code className="text-faint">scripts/</code> and installs neither dependency — that
          exclusion is what keeps it at ~937MB instead of ~2.6GB. So{" "}
          <code className="text-faint">serving/app.py</code> mounts this route only where the
          machinery exists, and answers 404 where it does not.
        </p>
        <p className="rounded-md border border-border bg-surface-2/50 px-3 py-2">
          <span className="text-fg">Nothing is drawn in its place.</span> A chart here would have
          to be invented, and an invented monitor is worse than an absent one. Run the API from
          the repo (<code className="text-faint">uv run uvicorn serving.app:app --reload</code>)
          and this tab comes alive.
        </p>
      </CardContent>
    </Card>
  )
}
