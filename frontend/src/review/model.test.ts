import { describe, expect, it } from "vitest";
import type { MaskRow, Rect, TrackData } from "../api/types";
import {
  MaskBudget,
  paddedRect,
  parseMaskRow,
  rasterizeMask,
  regionCoverage,
  classify,
  frameAtTime,
  indexTracks,
  normaliseRect,
  regionsAtFrame,
  sensitivityToThreshold,
  shouldBlur,
  summarise,
  timeOfFrame,
} from "./model";

const policy = { threshold: 0.35, minFaceSamples: 3 };

function track(id: number, p: number | null, samples: number, frames = [0, 1, 2]): TrackData {
  return {
    track_id: id,
    first_frame: frames[0]!,
    last_frame: frames[frames.length - 1]!,
    p_minor: p,
    expected_age: null,
    face_samples: samples,
    peak_confidence: 0.9,
    boxes: frames.map((f) => [f, 10, 10, 50, 90, 0]),
  };
}

describe("policy (mirrors backend)", () => {
  it("treats too little face evidence as uncertain, and blurs it", () => {
    const t = track(1, 0.01, 2);
    expect(classify(t, policy)).toBe("uncertain");
    expect(shouldBlur(t, policy)).toBe(true);
  });

  it("splits minors and adults at the threshold", () => {
    expect(classify(track(1, 0.35, 5), policy)).toBe("minor");
    expect(classify(track(1, 0.34, 5), policy)).toBe("adult");
    expect(shouldBlur(track(1, 0.1, 5), policy)).toBe(false);
  });

  it("lets reviewer overrides win", () => {
    expect(shouldBlur(track(1, 0.0, 9), policy, "blur")).toBe(true);
    expect(shouldBlur(track(1, 1.0, 9), policy, "keep")).toBe(false);
  });

  it("maps sensitivity to threshold", () => {
    expect(sensitivityToThreshold(0.65)).toBe(0.35);
    expect(sensitivityToThreshold(1)).toBe(0);
  });
});

describe("frame timing", () => {
  it("round-trips frame -> seek time -> frame", () => {
    for (const fps of [24, 25, 30000 / 1001, 60]) {
      for (const f of [0, 1, 29, 1799]) {
        expect(frameAtTime(timeOfFrame(f, fps), fps)).toBe(f);
        expect(frameAtTime(f / fps, fps)).toBe(f); // presentation timestamps
      }
    }
  });
});

describe("regionsAtFrame", () => {
  const tracks = indexTracks([track(1, 0.9, 5, [0, 1]), track(2, 0.05, 5, [1, 2])]);
  const manual = [{ region_id: "a", start_frame: 2, end_frame: 4, box: [0, 0, 5, 5] as const }];

  it("reports tracks and manual regions present on a frame", () => {
    const at1 = regionsAtFrame(1, tracks, policy, {}, []);
    expect(at1.map((r) => [r.trackId, r.blurred])).toEqual([
      [1, true],
      [2, false],
    ]);
    const at2 = regionsAtFrame(2, tracks, policy, { 2: "blur" }, [
      { ...manual[0]!, box: [0, 0, 5, 5] },
    ]);
    expect(at2.map((r) => [r.kind, r.blurred])).toEqual([
      ["track", true],
      ["manual", true],
    ]);
  });

  it("summarises decisions", () => {
    expect(summarise(tracks, policy, {})).toEqual({
      minor: 1,
      adult: 1,
      uncertain: 0,
      blurred: 1,
      kept: 1,
    });
  });
});

describe("normaliseRect", () => {
  it("orders corners and clamps to the frame", () => {
    expect(normaliseRect([120, 90], [-5, 10], 100, 80)).toEqual([0, 10, 100, 80]);
  });
  it("ignores tiny accidental drags", () => {
    expect(normaliseRect([10, 10], [12, 11], 100, 100)).toBeNull();
  });
});

// --- silhouette masks (mirror backend/src/safeframe/pipeline/masks.py) -------

const FRAME = { width: 320, height: 240 };
const valid = (f = 0): MaskRow => [f, 10, 10, 4, 2, [1, 3, 2, 2]];

function maskedTrack(rows: unknown[], interpolated: number[] = []): TrackData {
  return {
    ...track(1, null, 0, [0, 1, 2, 3]),
    boxes: [0, 1, 2, 3].map((f) => [f, 10, 10, 50, 90, interpolated.includes(f) ? 1 : 0]),
    masks_v1: rows as MaskRow[],
  };
}

describe("mask validation", () => {
  it.each([
    ["short", [0, 10, 10, 4, 2]],
    ["float origin", [0, 10.5, 10, 4, 2, [1, 3, 2, 2]]],
    ["negative", [0, -1, 10, 4, 2, [1, 3, 2, 2]]],
    ["past right edge", [0, 318, 10, 4, 2, [1, 3, 2, 2]]],
    ["bad sum", [0, 10, 10, 4, 2, [1, 3, 2, 1]]],
    ["zero run after first", [0, 10, 10, 4, 2, [1, 0, 5, 2]]],
    ["background only", [0, 10, 10, 4, 2, [8]]],
    ["runs not array", [0, 10, 10, 4, 2, "1,3,2,2"]],
  ])("rejects %s", (_, row) => {
    expect(parseMaskRow(row, FRAME.width, FRAME.height)).toBeNull();
  });

  it("accepts a crop starting in foreground", () => {
    expect(parseMaskRow([0, 0, 0, 2, 1, [0, 2]], 320, 240)).not.toBeNull();
  });

  it("drops invalid, duplicate, interpolated and box-less frames", () => {
    const rows = [valid(0), [1, 10, 10, 4, 2, [9, 9]], valid(2), valid(3), valid(3), valid(7)];
    const [t] = indexTracks([maskedTrack(rows, [2])], FRAME);
    expect([0, 1, 2, 3].map((f) => !!t!.byFrame.get(f)!.mask)).toEqual([true, false, false, false]);
  });

  it("ignores masks without a known frame size (legacy callers)", () => {
    const [t] = indexTracks([maskedTrack([valid(0)])]);
    expect(t!.byFrame.get(0)!.mask).toBeUndefined();
  });

  it("budget exhaustion is sticky", () => {
    const b = new MaskBudget(6);
    expect(b.take(4)).toBe(true);
    expect(b.take(4)).toBe(false);
    expect(b.take(1)).toBe(false);
  });

  it("regionsAtFrame carries masks only on observed frames", () => {
    const tracks = indexTracks([maskedTrack([valid(0), valid(1)], [1])], FRAME);
    expect(regionsAtFrame(0, tracks, policy, {}, [])[0]!.mask).toBeDefined();
    expect(regionsAtFrame(1, tracks, policy, {}, [])[0]!.mask).toBeUndefined();
  });
});

describe("mask geometry", () => {
  it("decodes at 1:1 exactly, including a row-wrapping run", () => {
    const r = rasterizeMask(valid(), 1, 1);
    expect([r.x, r.y, r.width, r.height]).toEqual([10, 10, 4, 2]);
    expect(Array.from(r.alpha)).toEqual([0, 255, 255, 255, 0, 0, 255, 255]);
  });

  it("keeps a 1 px line when downscaling (conservative)", () => {
    // 1 px wide vertical line at column 3 of an 8x8 crop.
    const runs = [3];
    for (let y = 0; y < 8; y++) runs.push(1, y < 7 ? 7 : 4);
    const r = rasterizeMask([0, 0, 0, 8, 8, runs], 0.25, 0.25);
    expect([r.width, r.height]).toEqual([2, 2]);
    expect(Array.from(r.alpha)).toEqual([255, 0, 255, 0]);
  });

  it("uses separate x and y ratios", () => {
    const r = rasterizeMask([0, 10, 20, 2, 2, [0, 4]], 2, 0.5);
    expect([r.x, r.y, r.width, r.height]).toEqual([20, 10, 4, 1]);
    expect(r.alpha.every((a) => a === 255)).toBe(true);
  });

  it("silhouette coverage leaves the box corner visible, box fallback covers it", () => {
    // A silhouette covering only the left band of box [10,10,50,90].
    const mask: MaskRow = [0, 10, 10, 10, 80, [0, 800]];
    const rect: Rect = [10, 10, 50, 90];
    const withMask = regionCoverage({ rect, mask }, 1, 1, 320, 240)!;
    const corner = { x: 48, y: 12 };
    const r = withMask.raster!;
    const inRaster =
      corner.x >= r.x && corner.x < r.x + r.width && corner.y >= r.y && corner.y < r.y + r.height;
    expect(inRaster).toBe(false);
    const [ex, ey, ew, eh] = regionCoverage({ rect }, 1, 1, 320, 240)!.effect;
    expect(corner.x >= ex && corner.x < ex + ew && corner.y >= ey && corner.y < ey + eh).toBe(true);
    expect(withMask.effect).toEqual(paddedRect(rect, 1, 1, 320, 240));
  });
});
