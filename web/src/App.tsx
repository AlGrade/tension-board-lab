/**
 * Ask for a grade, an angle, and a style; get a problem you can then edit.
 *
 * Whatever is on the board is graded, whether the generator put it there or you did. The
 * generator loads on first use rather than at startup; it is a separate 3.7 MB.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { BoardView, type CalibrationMap } from "./board/BoardView";
import { BoulderGenerator, rankCandidates, type RankedCandidate } from "./generate/sample";
import { Critic, type Prediction } from "./model/critic";
import {
  HOLD_ROLES,
  type BoardArtifact,
  type CriticArtifact,
  type GeneratorArtifact,
  type Hold,
  type HoldRole,
  type Placement,
  type StyleArtifact,
} from "./types";

const ANGLES = [35, 40, 45, 50, 55];
const CANDIDATES = 12;

/** Clicking a hold walks it through the roles and back to unselected. */
function nextRole(current: HoldRole | undefined): HoldRole | undefined {
  if (current === undefined) return "start";
  const index = HOLD_ROLES.indexOf(current);
  return index === HOLD_ROLES.length - 1 ? undefined : HOLD_ROLES[index + 1];
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
  const [style, setStyle] = useState<StyleArtifact>();
  const [critic, setCritic] = useState<Critic>();
  const [error, setError] = useState<string>();

  const [layout, setLayout] = useState("mirror");
  const [angle, setAngle] = useState(40);
  const [targetGrade, setTargetGrade] = useState("V5");
  const [preset, setPreset] = useState("");

  const [holds, setHolds] = useState<Hold[]>([]);
  const [prediction, setPrediction] = useState<Prediction>();
  const [scoring, setScoring] = useState(false);
  const [candidates, setCandidates] = useState<RankedCandidate[]>([]);
  const [shown, setShown] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState<string>();

  const generator = useRef<BoulderGenerator | undefined>(undefined);

  useEffect(() => {
    (async () => {
      try {
        const [boardData, styleData, criticData, calibrationData] = await Promise.all([
          fetchJson<BoardArtifact>("/data/board.json"),
          fetchJson<StyleArtifact>("/data/style.json"),
          fetchJson<CriticArtifact>("/data/critic.json"),
          fetchJson<CalibrationMap>("/board/calibration.json"),
        ]);
        setBoard(boardData);
        setStyle(styleData);
        setCalibration(calibrationData);
        setCritic(await Critic.load("/models/grade.int8.onnx", criticData));
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    })();
  }, []);

  // Positions differ between layouts, so a selection cannot survive the switch.
  useEffect(() => {
    setHolds([]);
    setCandidates([]);
  }, [layout]);

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
      .catch((cause: unknown) => setError(String(cause)))
      .finally(() => {
        if (!cancelled) setScoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, [critic, holds, angle, layout]);

  const generate = useCallback(async () => {
    if (!critic || !board || !style) return;
    setGenerating(true);
    setStatus("Sampling");
    try {
      if (!generator.current) {
        setStatus("Loading model");
        const artifact = await fetchJson<GeneratorArtifact>("/data/generator.json");
        generator.current = await BoulderGenerator.load(
          "/models/generator.int8.onnx",
          artifact,
          board,
          style,
        );
      }
      setStatus("Sampling");
      const sampled = await generator.current.sample({
        layout,
        angle,
        grade: targetGrade,
        preset: preset || undefined,
        count: CANDIDATES,
      });
      setStatus("Ranking");
      const scored = await critic.predict(sampled.map((candidate) => candidate.problem));
      const ranked = rankCandidates(sampled, scored, {
        targetIndex: critic.gradeIndex(targetGrade),
        preset: preset || undefined,
        style,
      });
      setCandidates(ranked);
      setShown(0);
      setHolds(ranked[0].problem.holds);
      setStatus(undefined);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setGenerating(false);
    }
  }, [critic, board, style, layout, angle, targetGrade, preset]);

  const showNext = useCallback(() => {
    if (candidates.length === 0) return;
    const next = (shown + 1) % candidates.length;
    setShown(next);
    setHolds(candidates[next].problem.holds);
  }, [candidates, shown]);

  if (error) {
    return (
      <main className="app">
        <div className="masthead">
          <h1>Tension Board Lab</h1>
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

  if (!board || !style || !calibration) {
    return (
      <main className="app">
        <div className="masthead">
          <h1>Tension Board Lab</h1>
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
        <h1>Tension Board Lab</h1>
        <p>Tension Board 2 · 12×12</p>
      </div>

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
          <div className="card">
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
          </div>

          <div className="card">
            <h2>Board</h2>
            <div className="fields">
              <label className="field">
                <span>Layout</span>
                <select value={layout} onChange={(event) => setLayout(event.target.value)}>
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
                >
                  {ANGLES.map((value) => (
                    <option key={value} value={value}>
                      {value}°
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="card">
            <h2>Generate</h2>
            <div className="fields">
              <label className="field">
                <span>Target</span>
                <select
                  value={targetGrade}
                  onChange={(event) => setTargetGrade(event.target.value)}
                  disabled={grades.length === 0}
                >
                  {grades.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Style</span>
                <select value={preset} onChange={(event) => setPreset(event.target.value)}>
                  <option value="">any</option>
                  {Object.keys(style.presets).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
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
              <button
                type="button"
                className="btn btn-ghost"
                onClick={showNext}
                disabled={candidates.length === 0 || generating}
              >
                {candidates.length > 0
                  ? `Next suggestion · ${shown + 1}/${candidates.length}`
                  : "Next suggestion"}
              </button>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
