/**
 * Ask for a grade, an angle, and a style; get a problem you can then edit.
 *
 * The generator is loaded on first use rather than at startup. It is a separate 3.7 MB on top
 * of the critic, and someone who only wants to grade a problem they built should not pay for
 * it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { BoardView, RoleLegend } from "./board/BoardView";
import { BoulderGenerator, rankCandidates, type RankedCandidate } from "./generate/sample";
import { Critic, type Prediction } from "./model/critic";
import { computeStyleFeatures } from "./style";
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
    throw new Error(`${path} returned ${response.status}. Run tension-export-web first.`);
  }
  return (await response.json()) as T;
}

export default function App() {
  const [board, setBoard] = useState<BoardArtifact>();
  const [style, setStyle] = useState<StyleArtifact>();
  const [critic, setCritic] = useState<Critic>();
  const [error, setError] = useState<string>();

  const [layout, setLayout] = useState("mirror");
  const [angle, setAngle] = useState(40);
  const [targetGrade, setTargetGrade] = useState("V5");
  const [preset, setPreset] = useState<string>("");

  const [holds, setHolds] = useState<Hold[]>([]);
  const [prediction, setPrediction] = useState<Prediction>();
  const [candidates, setCandidates] = useState<RankedCandidate[]>([]);
  const [shown, setShown] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState<string>();

  const generator = useRef<BoulderGenerator | undefined>(undefined);

  useEffect(() => {
    (async () => {
      try {
        const [boardData, styleData, criticData] = await Promise.all([
          fetchJson<BoardArtifact>("/data/board.json"),
          fetchJson<StyleArtifact>("/data/style.json"),
          fetchJson<CriticArtifact>("/data/critic.json"),
        ]);
        setBoard(boardData);
        setStyle(styleData);
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

  useEffect(() => {
    if (!critic || holds.length < 2) {
      setPrediction(undefined);
      return;
    }
    let cancelled = false;
    critic
      .predict([{ angle, layout, holds }])
      .then(([result]) => {
        if (!cancelled) setPrediction(result);
      })
      .catch((cause: unknown) => setError(String(cause)));
    return () => {
      cancelled = true;
    };
  }, [critic, holds, angle, layout]);

  const generate = useCallback(async () => {
    if (!critic || !board || !style) return;
    setGenerating(true);
    setStatus("Sampling…");
    try {
      if (!generator.current) {
        setStatus("Loading the generator…");
        const artifact = await fetchJson<GeneratorArtifact>("/data/generator.json");
        generator.current = await BoulderGenerator.load(
          "/models/generator.int8.onnx",
          artifact,
          board,
          style,
        );
      }
      setStatus("Sampling…");
      const sampled = await generator.current.sample({
        layout,
        angle,
        grade: targetGrade,
        preset: preset || undefined,
        count: CANDIDATES,
      });
      setStatus("Scoring…");
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

  const features = useMemo(() => {
    if (holds.length === 0) return undefined;
    try {
      return computeStyleFeatures(holds);
    } catch {
      return undefined;
    }
  }, [holds]);

  if (error) {
    return (
      <main className="app">
        <h1>Tension Board Lab</h1>
        <p className="error">{error}</p>
        <p>
          Build the artifacts first: <code>tension-export-onnx</code> and{" "}
          <code>tension-export-web</code>.
        </p>
      </main>
    );
  }

  if (!board || !style) {
    return (
      <main className="app">
        <h1>Tension Board Lab</h1>
        <p>Loading the board…</p>
      </main>
    );
  }

  const grades = critic?.gradeLabels ?? [];

  return (
    <main className="app">
      <header>
        <h1>Tension Board Lab</h1>
        <p className="subtitle">
          Ask for a grade and a style, or build a problem yourself. Either way the critic
          grades what is on the board.
        </p>
      </header>

      <div className="layout">
        <section className="board-panel">
          <BoardView board={board} layout={layout} holds={holds} onToggle={toggle} />
          <RoleLegend board={board} />
        </section>

        <aside className="controls">
          <label>
            Layout
            <select value={layout} onChange={(event) => setLayout(event.target.value)}>
              {Object.keys(board.layouts).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Angle
            <select value={angle} onChange={(event) => setAngle(Number(event.target.value))}>
              {ANGLES.map((value) => (
                <option key={value} value={value}>
                  {value}°
                </option>
              ))}
            </select>
          </label>

          <label>
            Target grade
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

          <label>
            Style
            <select value={preset} onChange={(event) => setPreset(event.target.value)}>
              <option value="">any</option>
              {Object.keys(style.presets).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <div className="actions">
            <button type="button" onClick={generate} disabled={!critic || generating}>
              {generating ? (status ?? "Working…") : "Generate"}
            </button>
            <button
              type="button"
              className="secondary"
              onClick={showNext}
              disabled={candidates.length === 0 || generating}
            >
              Next suggestion
            </button>
          </div>

          {candidates.length > 0 ? (
            <p className="muted small">
              Showing {shown + 1} of {candidates.length}, ranked by grade fit, style, and
              plausibility.
            </p>
          ) : null}

          <div className="readout">
            <h2>Grade</h2>
            {holds.length < 2 ? (
              <p className="muted">Generate a problem, or select at least two holds.</p>
            ) : prediction ? (
              <>
                <p className="grade">{prediction.grade}</p>
                <p className="muted">{(prediction.confidence * 100).toFixed(1)}% confidence</p>
                <p className="muted">Expected V{prediction.expectedGrade.toFixed(2)}</p>
              </>
            ) : (
              <p className="muted">Scoring…</p>
            )}
          </div>

          {features ? (
            <div className="readout">
              <h2>Style</h2>
              <dl>
                <dt>Hands</dt>
                <dd>{features.handCount}</dd>
                <dt>Feet</dt>
                <dd>{features.footCount}</dd>
                <dt>Mean move</dt>
                <dd>{features.meanMoveLength.toFixed(3)}</dd>
                <dt>Longest move</dt>
                <dd>{features.maxMoveLength.toFixed(3)}</dd>
              </dl>
            </div>
          ) : null}

          <p className="muted small">
            Grades are subjective; confidence is the model&rsquo;s, not a promise. Both models
            thin out above V11 and at 55°, and endurance problems are rare on this board.
          </p>
        </aside>
      </div>
    </main>
  );
}
