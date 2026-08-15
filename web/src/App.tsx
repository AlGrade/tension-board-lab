/**
 * Ask for a grade and an angle; get a problem you can then edit.
 *
 * Whatever is on the board is graded, whether the generator put it there or you did. The
 * generator loads on first use rather than at startup; it is a separate 3.8 MB.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { About } from "./About";
import { BoardView, type CalibrationMap } from "./board/BoardView";
import { BoulderGenerator, rankCandidates } from "./generate/sample";
import { Critic, type Prediction } from "./model/critic";
import {
  HOLD_ROLES,
  type BoardArtifact,
  type CriticArtifact,
  type GeneratorArtifact,
  type Hold,
  type HoldRole,
  type Placement,
} from "./types";

const ANGLES = [35, 40, 45, 50, 55];
const CANDIDATES = 12;

/** Clicking a hold walks it through the roles and back to unselected. */
function nextRole(current: HoldRole | undefined): HoldRole | undefined {
  if (current === undefined) return "start";
  const index = HOLD_ROLES.indexOf(current);
  return index === HOLD_ROLES.length - 1 ? undefined : HOLD_ROLES[index + 1];
}

/** Resolves after the browser has painted the current state. */
function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => setTimeout(resolve, 0));
  });
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    // Everything under /data and /models is produced by the export; /board is committed.
    const hint = path.startsWith("/board/") ? "" : " Run tension-export-web first.";
    throw new Error(`${path} returned ${response.status}.${hint}`);
  }
  return (await response.json()) as T;
}

export default function App() {
  const [board, setBoard] = useState<BoardArtifact>();
  const [calibration, setCalibration] = useState<CalibrationMap>();
  const [critic, setCritic] = useState<Critic>();
  const [error, setError] = useState<string>();

  const [view, setView] = useState<"board" | "about">("board");
  const [layout, setLayout] = useState("mirror");
  const [angle, setAngle] = useState(40);
  const [targetGrade, setTargetGrade] = useState("V5");

  const [holds, setHolds] = useState<Hold[]>([]);
  const [prediction, setPrediction] = useState<Prediction>();
  const [scoring, setScoring] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState<string>();

  const generator = useRef<BoulderGenerator | undefined>(undefined);

  useEffect(() => {
    (async () => {
      try {
        const [boardData, criticData, calibrationData] = await Promise.all([
          fetchJson<BoardArtifact>("/data/board.json"),
          fetchJson<CriticArtifact>("/data/critic.json"),
          fetchJson<CalibrationMap>("/board/calibration.json"),
        ]);
        setBoard(boardData);
        setCalibration(calibrationData);
        setCritic(await Critic.load("/models/grade.int8.onnx", criticData));
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    })();
  }, []);

  // Positions differ between layouts, so a selection cannot survive the switch.
  useEffect(() => setHolds([]), [layout]);

  const toggle = useCallback((placement: Placement) => {
    setHolds((current) => {
      const index = current.findIndex(
        (hold) => hold.x === placement.x && hold.y === placement.y,
      );
      const role = nextRole(index >= 0 ? current[index].role : undefined);
      if (index < 0) {
        return [
          ...current,
          {
            role: role as HoldRole,
            x: placement.x,
            y: placement.y,
            holdType: placement.hold_type,
            orientationDegrees: placement.orientation_degrees,
          },
        ];
      }
      if (role === undefined) return current.filter((_, position) => position !== index);
      return current.map((hold, position) => (position === index ? { ...hold, role } : hold));
    });
  }, []);

  // Re-grade whatever is on the board, however it got there.
  useEffect(() => {
    if (!critic || holds.length < 2) {
      setPrediction(undefined);
      return;
    }
    let cancelled = false;
    setScoring(true);
    critic
      .predict([{ angle, layout, holds }])
      .then(([result]) => {
        if (!cancelled) setPrediction(result);
      })
      .catch(() => {
        // A failed score leaves the last grade standing rather than discarding the problem.
        if (!cancelled) setPrediction(undefined);
      })
      .finally(() => {
        if (!cancelled) setScoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, [critic, holds, angle, layout]);

  const generate = useCallback(async () => {
    if (!critic || !board) return;
    setGenerating(true);
    setError(undefined);
    setStatus("Sampling");
    try {
      // Sampling holds the main thread between its awaits, so React would otherwise get no
      // frame to paint the veil in — on every run after the first, where nothing waits on
      // the network, it would appear only once the work was already over.
      await nextPaint();
      if (!generator.current) {
        setStatus("Loading model");
        const artifact = await fetchJson<GeneratorArtifact>("/data/generator.json");
        generator.current = await BoulderGenerator.load(
          "/models/generator.int8.onnx",
          artifact,
          board,
        );
      }
      setStatus("Sampling");
      const sampled = await generator.current.sample({
        layout,
        angle,
        grade: targetGrade,
        count: CANDIDATES,
      });
      setStatus("Ranking");
      const scored = await critic.predict(sampled.map((candidate) => candidate.problem));
      // Twelve are sampled and ranked; the best one goes on the board.
      const ranked = rankCandidates(sampled, scored, {
        targetIndex: critic.gradeIndex(targetGrade),
      });
      // The layout select is disabled while generating, so `layout` cannot have moved on.
      setHolds(ranked[0].problem.holds);
      setStatus(undefined);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setGenerating(false);
    }
  }, [critic, board, layout, angle, targetGrade]);

  if (error) {
    return (
      <main className="app">
        <div className="masthead">
          <h1>Tension Board AI</h1>
        </div>
        <div className="notice">
          <p className="error">{error}</p>
          <p>
            Build the artifacts first: <code>tension-export-onnx</code> and{" "}
            <code>tension-export-web</code>.
          </p>
        </div>
      </main>
    );
  }

  if (!board || !calibration) {
    return (
      <main className="app">
        <div className="masthead">
          <h1>Tension Board AI</h1>
        </div>
        <div className="workspace">
          <div className="stage">
            <div className="skeleton" />
          </div>
        </div>
      </main>
    );
  }

  const grades = critic?.gradeLabels ?? [];
  const confidence = prediction ? Math.round(prediction.confidence * 100) : 0;

  return (
    <main className="app">
      <div className="masthead">
        <h1>Tension Board AI</h1>
        <nav className="tabs" role="tablist">
          {(["board", "about"] as const).map((name) => (
            <button
              key={name}
              type="button"
              role="tab"
              aria-selected={view === name}
              className={view === name ? "tab is-active" : "tab"}
              onClick={() => setView(name)}
            >
              {name === "board" ? "Board" : "How it works"}
            </button>
          ))}
        </nav>
      </div>

      {view === "about" ? (
        <About />
      ) : (
      <div className="workspace">
        <section>
          <div className="stage">
            <BoardView
              board={board}
              calibration={calibration}
              layout={layout}
              holds={holds}
              onToggle={toggle}
            />
            {/* Generating replaces every hold on the board, so the whole board waits
                behind a veil rather than letting you edit a problem about to vanish. */}
            {generating ? (
              <div
                className="stage-veil"
                role="status"
                aria-live="polite"
                aria-label={status ?? "Working"}
              >
                <span className="spinner" />
              </div>
            ) : null}
          </div>

          <div className="editbar">
            <ul className="roles">
              {HOLD_ROLES.map((role, index) => (
                <li key={role}>
                  <span
                    className="dot"
                    style={{ background: `#${board.role_colors[role]}` }}
                  />
                  {role}
                  {index < HOLD_ROLES.length - 1 ? <span className="arrow">→</span> : null}
                </li>
              ))}
            </ul>
            <p className="hint">
              <span>
                {holds.length === 0
                  ? "Tap a hold to start building"
                  : `${holds.length} hold${holds.length === 1 ? "" : "s"} · tap to cycle, again to remove`}
              </span>
              <button
                type="button"
                className="linkish"
                onClick={() => setHolds([])}
                disabled={holds.length === 0}
              >
                Clear
              </button>
            </p>
          </div>

        </section>

        <aside className="panel">
          <div className="card card-grade">
            <h2>Predicted grade</h2>
            <div className={`verdict${prediction ? "" : " is-empty"}`}>
              {prediction ? (
                <>
                  <span className="value" key={prediction.grade}>
                    {prediction.grade}
                  </span>
                  <span className="aside">
                    {confidence}% · V{prediction.expectedGrade.toFixed(1)} expected
                  </span>
                </>
              ) : (
                <span className="value">
                  {holds.length < 2 ? "Pick at least two holds" : <span className="busy" />}
                </span>
              )}
            </div>
            {prediction ? (
              <div className="meter" aria-hidden="true">
                <span style={{ width: `${Math.max(confidence, 3)}%` }} />
              </div>
            ) : null}
            <p className="footnote">
              Estimated by a model from the holds and the angle — not an official grade. It
              re-runs every time you change the problem.{scoring && prediction ? " Updating…" : ""}
            </p>
            <div className="actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={generate}
                disabled={!critic || generating}
              >
                {generating ? (
                  <>
                    <span className="busy" />
                    {status ?? "Working"}
                  </>
                ) : (
                  "Generate a problem"
                )}
              </button>
            </div>
          </div>

          <div className="card">
            <h2>Board</h2>
            <div className="fields is-triple">
              <label className="field">
                <span>Layout</span>
                <select
                  value={layout}
                  onChange={(event) => setLayout(event.target.value)}
                  disabled={generating}
                >
                  {Object.keys(board.layouts).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Angle</span>
                <select
                  value={angle}
                  onChange={(event) => setAngle(Number(event.target.value))}
                  disabled={generating}
                >
                  {ANGLES.map((value) => (
                    <option key={value} value={value}>
                      {value}°
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Target</span>
                <select
                  value={targetGrade}
                  onChange={(event) => setTargetGrade(event.target.value)}
                  disabled={grades.length === 0 || generating}
                >
                  {grades.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="card">
            <h2>Worth knowing</h2>
            <ul className="notes">
              <li>
                <strong>Matching is allowed</strong>.
              </li>
              <li>
                <strong>V11 and harder gets shaky.</strong> There are only a few hundred such
                problems to learn from, and a handful above V12, so both networks are guessing
                more than they let on. Treat hard grades as a starting point, not a verdict.
              </li>
            </ul>
          </div>
        </aside>
      </div>
      )}
    </main>
  );
}
