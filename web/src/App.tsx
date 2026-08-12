/**
 * Step 4: the board, the critic, and live re-scoring as you edit.
 *
 * Generation arrives in step 5; this exists to prove the exported model and the mirrored
 * featurization behave in a real browser.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { BoardView, RoleLegend } from "./board/BoardView";
import { Critic, type Prediction } from "./model/critic";
import { computeStyleFeatures } from "./style";
import {
  HOLD_ROLES,
  type BoardArtifact,
  type CriticArtifact,
  type Hold,
  type HoldRole,
  type Placement,
  type StyleArtifact,
} from "./types";

const ANGLES = [35, 40, 45, 50, 55];

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
  const [holds, setHolds] = useState<Hold[]>([]);
  const [prediction, setPrediction] = useState<Prediction>();
  const [scoring, setScoring] = useState(false);

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

  // Changing the layout invalidates the selection: positions differ between layouts.
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
      if (role === undefined) {
        return current.filter((_, position) => position !== index);
      }
      return current.map((hold, position) => (position === index ? { ...hold, role } : hold));
    });
  }, []);

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

  const features = useMemo(() => {
    if (holds.length === 0) return undefined;
    try {
      return computeStyleFeatures(holds);
    } catch {
      // A problem with no hand, start, or finish hold has no style yet.
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

  return (
    <main className="app">
      <header>
        <h1>Tension Board Lab</h1>
        <p className="subtitle">
          Select holds to build a problem. The grade updates as you edit.
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

          <div className="readout">
            <h2>Grade</h2>
            {holds.length < 2 ? (
              <p className="muted">Select at least two holds.</p>
            ) : !critic ? (
              <p className="muted">Loading the model…</p>
            ) : prediction ? (
              <>
                <p className="grade">{prediction.grade}</p>
                <p className="muted">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                  {scoring ? " · updating" : ""}
                </p>
                <p className="muted">
                  Expected V{prediction.expectedGrade.toFixed(2)}
                </p>
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
            Grades are subjective; confidence is the model&rsquo;s, not a promise. Accuracy
            thins out above V11 and at 55°.
          </p>
        </aside>
      </div>
    </main>
  );
}
