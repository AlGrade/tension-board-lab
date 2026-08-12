/**
 * Parity for the generator tokenizer and its constraint masks.
 *
 * The sampling loop itself cannot be compared step for step — the two sides draw from
 * different random number generators — so the pieces it is built from are compared instead.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { BoulderConstraints } from "../src/generate/constraints";
import { GeneratorVocabulary, encodePrefix, positionKey } from "../src/model/tokenizer";
import type { BoardArtifact, GeneratorArtifact, HoldRole } from "../src/types";

const DATA = new URL("../public/data/", import.meta.url);

function readJson<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(name, DATA), "utf8")) as T;
}

interface GeneratorFixtures {
  layout: string;
  sequences: {
    problem: {
      angle: number;
      source_layout: string;
      grade: string;
      holds: { role: HoldRole; x: number; y: number }[];
    };
    tokens: number[];
    unconditional: number[];
  }[];
  prefixes: { layout: string; angle: number; grade: string; tokens: number[] }[];
  mask_states: { name: string; chosen: number[]; allowed: number[] }[];
}

const board = readJson<BoardArtifact>("board.json");
const artifact = readJson<GeneratorArtifact>("generator.json");
const fixtures = readJson<GeneratorFixtures>("generator_fixtures.json");
const vocabulary = new GeneratorVocabulary(artifact);

describe("generator vocabulary", () => {
  it("agrees with Python on size and offsets", () => {
    expect(vocabulary.size).toBe(2182);
    expect(vocabulary.positions.length).toBe(537);
    expect(vocabulary.prefixLength).toBe(3);
    expect({
      pad: vocabulary.pad,
      bos: vocabulary.bos,
      eos: vocabulary.eos,
      uncond: vocabulary.uncond,
    }).toEqual({ pad: 0, bos: 1, eos: 2, uncond: 3 });
  });

  it("round-trips every hold token", () => {
    for (const [x, y] of vocabulary.positions.slice(0, 40)) {
      for (const role of ["start", "hand", "foot", "finish"] as HoldRole[]) {
        const token = vocabulary.holdToken(x, y, role);
        const back = vocabulary.holdFromToken(token);
        expect(positionKey(back.x, back.y)).toBe(positionKey(x, y));
        expect(back.role).toBe(role);
      }
    }
  });

  it("rejects a position that is not on the board", () => {
    expect(() => vocabulary.holdToken(0.123456, 0.654321, "hand")).toThrow();
  });
});

describe("prefix encoding", () => {
  fixtures.prefixes.forEach((fixture, index) => {
    const label = `${fixture.layout} ${fixture.angle}° ${fixture.grade}`;
    it(`matches Python for prefix ${index} (${label})`, () => {
      const tokens = encodePrefix(vocabulary, {
        layout: fixture.layout,
        angle: fixture.angle,
        grade: fixture.grade,
      });
      expect(tokens).toEqual(fixture.tokens);
    });
  });

  it("blanks every conditioning token when unconditional", () => {
    const tokens = encodePrefix(vocabulary, {
      layout: "mirror",
      angle: 40,
      grade: "V5",
      unconditional: true,
    });
    expect(tokens[0]).toBe(vocabulary.bos);
    expect(tokens.slice(1)).toEqual(Array(vocabulary.prefixLength).fill(vocabulary.uncond));
    // The prefix keeps its width, so holds sit at the same offsets either way.
    expect(tokens.length).toBe(1 + vocabulary.prefixLength);
  });
});

describe("token sequences", () => {
  fixtures.sequences.forEach((fixture, index) => {
    it(`reproduces the hold tokens Python emitted for sequence ${index}`, () => {
      const body = fixture.tokens.slice(1 + vocabulary.prefixLength, -1);
      // Python sorts holds bottom to top; rebuild that order here and compare tokens.
      const sorted = [...fixture.problem.holds].sort((a, b) => a.y - b.y || a.x - b.x);
      const expected = sorted.map((hold) =>
        vocabulary.holdToken(hold.x, hold.y, hold.role),
      );
      expect(body).toEqual(expected);
    });

    it(`matches Python's unconditional prefix for sequence ${index}`, () => {
      expect(fixture.unconditional.slice(0, 1 + vocabulary.prefixLength)).toEqual(
        encodePrefix(vocabulary, {
          layout: fixture.problem.source_layout,
          angle: fixture.problem.angle,
          grade: fixture.problem.grade,
          unconditional: true,
        }),
      );
      // Holds are untouched by guidance dropout.
      expect(fixture.unconditional.slice(1 + vocabulary.prefixLength)).toEqual(
        fixture.tokens.slice(1 + vocabulary.prefixLength),
      );
    });

  });

  it("keeps every sequence inside the model's length limit", () => {
    for (const fixture of fixtures.sequences) {
      expect(fixture.tokens.length).toBeLessThanOrEqual(vocabulary.maxSequenceLength);
    }
  });
});

describe("constraint masks", () => {
  const constraints = new BoulderConstraints(vocabulary, board, artifact, fixtures.layout);

  fixtures.mask_states.forEach((state) => {
    it(`matches Python exactly at state "${state.name}"`, () => {
      const mask = constraints.maskFor(state.chosen);
      const allowed: number[] = [];
      mask.forEach((value, token) => {
        if (value === 1) allowed.push(token);
      });
      expect(allowed).toEqual(state.allowed);
    });
  });

  it("blocks EOS until the problem has a start and a finish", () => {
    const empty = fixtures.mask_states.find((s) => s.name === "empty")!;
    const both = fixtures.mask_states.find((s) => s.name === "start_and_finish")!;
    expect(empty.allowed).not.toContain(vocabulary.eos);
    expect(both.allowed).toContain(vocabulary.eos);
    expect(constraints.isComplete(both.chosen)).toBe(true);
    expect(constraints.isComplete(empty.chosen)).toBe(false);
  });

  it("never allows a position from another layout", () => {
    const other = vocabulary.layouts.find((name) => name !== fixtures.layout)!;
    const here = new Set(
      board.layouts[fixtures.layout].map((p) => positionKey(p.x, p.y)),
    );
    const elsewhere = board.layouts[other].filter((p) => !here.has(positionKey(p.x, p.y)));
    expect(elsewhere.length).toBeGreaterThan(0);
    const mask = constraints.maskFor([]);
    for (const placement of elsewhere) {
      for (const role of ["start", "hand", "foot", "finish"] as HoldRole[]) {
        expect(mask[vocabulary.holdToken(placement.x, placement.y, role)]).toBe(0);
      }
    }
  });

  it("forces the missing role when the hold limit is nearly reached", () => {
    const tight = new BoulderConstraints(vocabulary, board, artifact, fixtures.layout, {
      maxHolds: 3,
    });
    const start = fixtures.mask_states.find((s) => s.name === "one_start")!.chosen;
    const mask = tight.maskFor([...start, vocabulary.holdToken(...pickHand())]);
    expect(mask[vocabulary.eos]).toBe(0);
    const allowedRoles = new Set<HoldRole>();
    mask.forEach((value, token) => {
      if (value === 1 && vocabulary.isHoldToken(token)) {
        allowedRoles.add(vocabulary.holdFromToken(token).role);
      }
    });
    expect([...allowedRoles]).toEqual(["finish"]);
  });

  function pickHand(): [number, number, HoldRole] {
    const placement = board.layouts[fixtures.layout][100];
    return [placement.x, placement.y, "hand"];
  }
});
