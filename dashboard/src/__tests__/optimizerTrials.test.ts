import { describe, expect, it } from "vitest";

import {
  paretoLabel,
  paretoTone,
  parseParamsJson,
  sortTrialsForLeaderboard,
  summarizeTrials,
  topNByScore,
} from "../optimizerTrials";
import type { OptimizerTrialRow } from "../mlApi";

function row(overrides: Partial<OptimizerTrialRow>): OptimizerTrialRow {
  return {
    study_id: "study-1",
    trial_id: 0,
    ts_ms: 1_700_000_000_000,
    params_json: "{}",
    score: 0,
    sortino: 0,
    max_dd_bp: 0,
    turnover: 0,
    pareto_rank: 0,
    ...overrides,
  };
}

describe("parseParamsJson", () => {
  it("returns null on null/empty/malformed input", () => {
    expect(parseParamsJson(null)).toBeNull();
    expect(parseParamsJson(undefined)).toBeNull();
    expect(parseParamsJson("")).toBeNull();
    expect(parseParamsJson("not json")).toBeNull();
    expect(parseParamsJson("[1,2,3]")).toBeNull();
    expect(parseParamsJson("null")).toBeNull();
  });

  it("builds a snippet from the recognized keys in order", () => {
    const json = JSON.stringify({
      stop_loss_atr: 1.4,
      long_threshold: 0.62,
      short_threshold: 0.58,
      cooldown_bars: 5,
      ignored_field: "foo",
    });
    const out = parseParamsJson(json);
    expect(out).not.toBeNull();
    expect(out!.snippet).toBe("lt=0.62 st=0.58 sl=1.40 cd=5");
    expect(out!.raw.ignored_field).toBe("foo");
  });

  it("skips non-finite numeric fields gracefully", () => {
    const json = JSON.stringify({
      long_threshold: Number.NaN,
      short_threshold: 0.55,
    });
    const out = parseParamsJson(json);
    expect(out!.snippet).toBe("st=0.55");
  });

  it("formats integer fields without decimals", () => {
    const out = parseParamsJson(JSON.stringify({ min_hold_bars: 3, max_hold_bars: 60 }));
    expect(out!.snippet).toBe("minH=3 maxH=60");
  });
});

describe("sortTrialsForLeaderboard", () => {
  it("orders Pareto-ranked rows ahead of unranked, score desc within rank", () => {
    const rows: OptimizerTrialRow[] = [
      row({ trial_id: 1, score: 0.5, pareto_rank: 0 }),
      row({ trial_id: 2, score: 0.3, pareto_rank: 1 }),
      row({ trial_id: 3, score: 0.4, pareto_rank: 1 }),
      row({ trial_id: 4, score: 0.9, pareto_rank: 2 }),
      row({ trial_id: 5, score: 1.5, pareto_rank: 0 }),
    ];
    const sorted = sortTrialsForLeaderboard(rows);
    expect(sorted.map((r) => r.trial_id)).toEqual([3, 2, 4, 5, 1]);
  });

  it("does not mutate the input array", () => {
    const rows = [row({ trial_id: 1, score: 0.1 }), row({ trial_id: 2, score: 0.9 })];
    const before = rows.map((r) => r.trial_id);
    sortTrialsForLeaderboard(rows);
    expect(rows.map((r) => r.trial_id)).toEqual(before);
  });
});

describe("topNByScore", () => {
  it("returns up to N rows in leaderboard order", () => {
    const rows = Array.from({ length: 10 }, (_, i) =>
      row({ trial_id: i, score: i * 0.1, pareto_rank: i < 3 ? 1 : 0 }),
    );
    const top5 = topNByScore(rows, 5);
    expect(top5.length).toBe(5);
    // First 3 are Pareto-1 sorted by score desc → trial_ids 2, 1, 0;
    // then unranked sorted by score desc → 9, 8.
    expect(top5.map((r) => r.trial_id)).toEqual([2, 1, 0, 9, 8]);
  });

  it("clamps a negative N to 0", () => {
    expect(topNByScore([row({})], -3)).toEqual([]);
  });
});

describe("summarizeTrials", () => {
  it("returns zero-state on empty input", () => {
    const s = summarizeTrials([]);
    expect(s.count).toBe(0);
    expect(s.bestScore).toBeNull();
    expect(s.bestSortino).toBeNull();
    expect(s.bestDrawdownBp).toBeNull();
    expect(s.paretoCount).toBe(0);
  });

  it("aggregates correctly across rows", () => {
    const rows = [
      row({ trial_id: 1, score: 0.4, sortino: 1.2, max_dd_bp: 80, pareto_rank: 1, ts_ms: 100 }),
      row({ trial_id: 2, score: 0.9, sortino: 1.8, max_dd_bp: 50, pareto_rank: 1, ts_ms: 200 }),
      row({ trial_id: 3, score: 0.1, sortino: 0.3, max_dd_bp: 200, pareto_rank: 0, ts_ms: 300, study_id: "study-2" }),
    ];
    const s = summarizeTrials(rows);
    expect(s.count).toBe(3);
    expect(s.paretoCount).toBe(2);
    expect(s.bestScore).toBeCloseTo(0.9);
    expect(s.bestSortino).toBeCloseTo(1.8);
    expect(s.bestDrawdownBp).toBeCloseTo(50);
    expect(s.latestStudy).toBe("study-2");
    expect(s.latestTsMs).toBe(300);
  });
});

describe("paretoTone / paretoLabel", () => {
  it("paretoLabel returns dash for unranked", () => {
    expect(paretoLabel(0)).toBe("—");
    expect(paretoLabel(-1)).toBe("—");
    expect(paretoLabel(1)).toBe("P1");
    expect(paretoLabel(7)).toBe("P7");
  });

  it("paretoTone returns distinct colour buckets", () => {
    const t1 = paretoTone(1);
    const t2 = paretoTone(2);
    const t3 = paretoTone(3);
    const t4 = paretoTone(7);
    const t0 = paretoTone(0);
    const set = new Set([t1, t2, t3, t4, t0]);
    expect(set.size).toBe(5);
  });
});
