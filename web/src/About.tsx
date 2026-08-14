/**
 * The "How it works" tab: what the two networks are, in as few words as carry the idea.
 *
 * The diagrams are inline SVG rather than images so they follow the theme and stay sharp. The
 * bar chart is the real output distribution for the example problem, not a sketch.
 */

// Measured for examples/route.json: softmax(logits / 1.67).
const DISTRIBUTION = [
  0.01, 0.008, 0.009, 0.01, 0.016, 0.01, 0.014, 0.029, 0.344, 0.383, 0.117, 0.025, 0.009, 0.007,
  0.008,
];

function GeneratorDiagram() {
  const inputs = [
    { y: 32, text: "grade + angle you asked for" },
    { y: 62, text: "which wall (mirror / spray)" },
    { y: 92, text: "holds placed so far" },
  ];
  return (
    <svg viewBox="0 0 560 150" className="figure" role="img" aria-label="How the generator works">
      {inputs.map(({ y, text }, i) => (
        <g key={i}>
          <text x="6" y={y + 4} className="fig-note">{text}</text>
          <path d={`M196 ${y} H228`} className="fig-arrow" markerEnd="url(#tip2)" />
        </g>
      ))}

      <rect x="236" y="38" width="126" height="48" rx="10" className="fig-box" />
      <text x="299" y="59" className="fig-box-text" textAnchor="middle">Decoder-only</text>
      <text x="299" y="75" className="fig-box-text" textAnchor="middle">transformer</text>

      <path d="M370 62 h28" className="fig-arrow" markerEnd="url(#tip2)" />
      <text x="406" y="52" className="fig-label">a score for every</text>
      <text x="406" y="68" className="fig-label">possible next hold</text>
      <text x="406" y="86" className="fig-note">illegal ones struck out first</text>

      {/* Pick one, append it, and run again. */}
      <path d="M474 96 v22 H150 v-18" className="fig-arrow fig-loop" markerEnd="url(#tip2)" />
      <text x="312" y="134" className="fig-note" textAnchor="middle">
        pick one, add it to the problem, repeat
      </text>

      <defs>
        <marker id="tip2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
          orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" className="fig-tip" />
        </marker>
      </defs>
    </svg>
  );
}

function CriticDiagram() {
  // A handful of holds, drawn as a graph: every point connected to every other.
  const nodes = [
    { x: 30, y: 92 },
    { x: 62, y: 58 },
    { x: 44, y: 26 },
    { x: 86, y: 96 },
    { x: 96, y: 40 },
  ];
  const edges = nodes.flatMap((a, i) => nodes.slice(i + 1).map((b) => ({ a, b })));
  const peak = Math.max(...DISTRIBUTION);

  return (
    <svg viewBox="0 0 560 130" className="figure" role="img" aria-label="How the grade critic works">
      <text x="62" y="14" className="fig-label" textAnchor="middle">the holds you picked</text>
      {edges.map(({ a, b }, i) => (
        <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="fig-edge" />
      ))}
      {nodes.map((n, i) => (
        <circle key={i} cx={n.x} cy={n.y} r="6" className="fig-node" />
      ))}
      <text x="62" y="122" className="fig-note" textAnchor="middle">+ wall angle</text>

      <path d="M126 62 h34" className="fig-arrow" markerEnd="url(#tip)" />

      <rect x="168" y="38" width="132" height="48" rx="10" className="fig-box" />
      <text x="234" y="59" className="fig-box-text" textAnchor="middle">Graph</text>
      <text x="234" y="75" className="fig-box-text" textAnchor="middle">transformer</text>

      <path d="M308 62 h30" className="fig-arrow" markerEnd="url(#tip)" />

      {DISTRIBUTION.map((value, i) => {
        const height = (value / peak) * 56;
        return (
          <rect
            key={i}
            x={352 + i * 13}
            y={92 - height}
            width="9"
            height={Math.max(height, 1.5)}
            rx="2"
            className={i === 9 ? "fig-bar fig-bar-peak" : "fig-bar"}
          />
        );
      })}
      <text x="352" y="106" className="fig-note">V0</text>
      <text x="530" y="106" className="fig-note" textAnchor="end">V14</text>
      <text x="450" y="18" className="fig-label" textAnchor="middle">one score per grade</text>
      <text x="450" y="124" className="fig-note" textAnchor="middle">highest wins: V9 at 38%</text>

      <defs>
        <marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
          orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" className="fig-tip" />
        </marker>
      </defs>
    </svg>
  );
}

export function About() {
  return (
    <article className="about">
      <p className="lede">
        Two small neural networks, trained from scratch on TB2 boulders. One
        invents problems, the other grades them. Both run in the browser.
      </p>

      <section>
        <h2>The generator</h2>
        <p>
          Ask for a grade and an angle, and it builds a problem that did not exist before — one
          hold at a time, bottom of the wall upward, the way a phone keyboard predicts the next
          word.
        </p>
        <GeneratorDiagram />
        <p>
          It chooses a <em>position</em> rather than a hold; the wall decides what is bolted
          there, which is why it also has to be told which of the two layouts you are on. Before
          each pick, impossible choices are switched off — a hold already used, a position that
          does not exist on this wall, ending the climb before there is a start and a finish. An
          invalid problem cannot come out, rather than being filtered out afterwards.
        </p>
        <p>
          One press of Generate produces twelve of them. Choosing which one you actually see is
          the second network&rsquo;s job.
        </p>
      </section>

      <section>
        <h2>The grade critic</h2>
        <p>
          Show it the holds and how steep the wall is, and it answers with a difficulty. It
          grades all twelve candidates, and the one closest to what you asked for goes on the
          board — and it grades again, live, every time you change a hold yourself.
        </p>
        <CriticDiagram />
        <p>
          Every hold becomes a point, and the network lets each point weigh every other one —
          which pair makes a long reach, which foot holds up which hand. It never sees the
          problem&rsquo;s name or how popular it is; only where the holds are, which way they
          face, and what each is for.
        </p>
        <p>
          What comes out is not one number but <strong>fifteen</strong>: a score for every grade
          from V0 to V14. The highest wins, and its share is the confidence. A V9 at 38% usually
          has V8 close behind — that near-tie <em>is</em> the low confidence, and it is roughly
          what two climbers would say to each other.
        </p>
        <p className="muted">
          Measured against 2,174 problems it had never seen: off by 0.935 grades on average,
          within one grade 78% of the time. Climbers routinely disagree by a grade.
        </p>
      </section>

      <section>
        <h2>Worth knowing</h2>
        <ul className="notes">
          <li>
            <strong>Matching is allowed</strong> on every generated problem — both hands may
            share any hold.
          </li>
          <li>
            <strong>V11 and harder gets shaky.</strong> There are only a few hundred such
            problems to learn from, and a handful above V12, so both networks are guessing more
            than they let on. Treat hard grades as a starting point, not a verdict.
          </li>
          <li>
            The generator learned which holds go together, not how a body moves between them. It
            can leave a foot too far from the hold it should support.
          </li>
        </ul>
      </section>
    </article>
  );
}
