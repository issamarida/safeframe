/**
 * Review-screen logic, kept free of React so it can be unit tested.
 *
 * `classify` and `shouldBlur` mirror backend/src/safeframe/pipeline/policy.py
 * exactly: the preview must show what the export will contain.
 */
import type { ManualRegion, MaskRow, Override, Rect, TrackData } from "../api/types";

export type TrackStatus = "minor" | "adult" | "uncertain";

export interface Policy {
  threshold: number;
  minFaceSamples: number;
}

export function classify(track: TrackData, policy: Policy): TrackStatus {
  if (track.face_samples < policy.minFaceSamples || track.p_minor === null) return "uncertain";
  return track.p_minor >= policy.threshold ? "minor" : "adult";
}

export function shouldBlur(track: TrackData, policy: Policy, override?: Override): boolean {
  if (override === "blur") return true;
  if (override === "keep") return false;
  return classify(track, policy) !== "adult";
}

/** Sensitivity is the user-facing inverse of the P(under 18) threshold. */
export const sensitivityToThreshold = (s: number) => round2(1 - s);
export const thresholdToSensitivity = (t: number) => round2(1 - t);
const round2 = (n: number) => Math.round(n * 100) / 100;

/**
 * Frame index shown at media time `t`. Frame i has presentation time i/fps;
 * the epsilon absorbs floating-point error in reported times.
 */
export function frameAtTime(t: number, fps: number): number {
  return Math.max(0, Math.floor(t * fps + 1e-3));
}

/** Seek target for frame i: mid-frame so the browser lands on exactly that frame. */
export function timeOfFrame(frame: number, fps: number): number {
  return (frame + 0.5) / fps;
}

// --- Silhouette masks --------------------------------------------------------
// Validation and budgets mirror backend/src/safeframe/pipeline/masks.py
// rule for rule, so review and export pick the same mask-or-box per frame.

/** Same padding the renderer adds around every box (anonymize.DEFAULT_PADDING). */
export const PADDING = 0.08;
export const MAX_RUNS_PER_MASK = 8_192;
export const MAX_RUNS_PER_TRACK = 1_500_000;
export const MAX_RUNS_TOTAL = 4_000_000;

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

/** Validate one `masks_v1` row before anything area-sized is allocated. */
export function parseMaskRow(row: unknown, frameWidth: number, frameHeight: number): MaskRow | null {
  if (!Array.isArray(row) || row.length !== 6) return null;
  const [frame, x, y, w, h, runs] = row as unknown[];
  if (!isInt(frame) || !isInt(x) || !isInt(y) || !isInt(w) || !isInt(h)) return null;
  if (frame < 0 || x < 0 || y < 0 || w < 1 || h < 1) return null;
  if (x + w > frameWidth || y + h > frameHeight) return null;
  if (!Array.isArray(runs) || runs.length < 2 || runs.length > MAX_RUNS_PER_MASK) return null;
  let total = 0;
  for (let i = 0; i < runs.length; i++) {
    const r: unknown = runs[i];
    if (!isInt(r) || r < 0 || (r === 0 && i > 0)) return null;
    total += r;
  }
  if (total !== w * h) return null;
  return row as MaskRow;
}

export class MaskBudget {
  remaining: number;
  constructor(remaining = MAX_RUNS_TOTAL) {
    this.remaining = remaining;
  }
  take(runs: number): boolean {
    if (runs > this.remaining) {
      this.remaining = 0; // exhausted for good, like the backend
      return false;
    }
    this.remaining -= runs;
    return true;
  }
}

export interface FrameEntry {
  rect: Rect;
  interpolated: boolean;
  /** Validated silhouette; only ever set on observed (non-interpolated) frames. */
  mask?: MaskRow;
}

export interface IndexedTrack extends TrackData {
  byFrame: Map<number, FrameEntry>;
}

/**
 * Index boxes (and, when the frame size is known, validated masks) by frame.
 * Invalid, duplicate, misplaced or over-budget masks are dropped, so those
 * frames fall back to the padded box exactly as the export does.
 */
export function indexTracks(
  tracks: TrackData[],
  frame?: { width: number; height: number },
): IndexedTrack[] {
  const budget = new MaskBudget();
  return tracks.map((t) => {
    const byFrame = new Map<number, FrameEntry>(
      t.boxes.map(([f, x1, y1, x2, y2, interp]) => [
        f,
        { rect: [x1, y1, x2, y2] as Rect, interpolated: interp === 1 },
      ]),
    );
    if (frame && Array.isArray(t.masks_v1)) {
      for (const [f, mask] of loadTrackMasks(t.masks_v1, byFrame, frame, budget)) {
        byFrame.get(f)!.mask = mask;
      }
    }
    return { ...t, byFrame };
  });
}

function loadTrackMasks(
  rows: unknown[],
  byFrame: Map<number, FrameEntry>,
  frame: { width: number; height: number },
  budget: MaskBudget,
): Map<number, MaskRow> {
  const out = new Map<number, MaskRow>();
  const duplicates = new Set<number>();
  let trackRuns = 0;
  let exhausted = false;
  for (const row of rows) {
    if (exhausted) continue;
    const mask = parseMaskRow(row, frame.width, frame.height);
    if (!mask) continue;
    const f = mask[0];
    const entry = byFrame.get(f);
    if (!entry || entry.interpolated) continue;
    if (out.has(f) || duplicates.has(f)) {
      duplicates.add(f);
      out.delete(f);
      continue;
    }
    const n = mask[5].length;
    if (trackRuns + n > MAX_RUNS_PER_TRACK || !budget.take(n)) {
      exhausted = true;
      continue;
    }
    trackRuns += n;
    out.set(f, mask);
  }
  return out;
}

/** Coverage raster in preview pixels: alpha 255 = protected, 0 = untouched. */
export interface MaskRaster {
  x: number;
  y: number;
  width: number;
  height: number;
  alpha: Uint8Array;
}

/**
 * Resample a source-pixel mask to the preview with ratios `sx`, `sy`.
 * Conservative: a preview pixel is covered if *any* covered source pixel
 * overlaps it, so downscaling never thins limbs, and there is no partially
 * transparent edge that would reveal an outline.
 */
export function rasterizeMask(mask: MaskRow, sx: number, sy: number): MaskRaster {
  const [, mx, my, w, h, runs] = mask;
  const x0 = Math.floor(mx * sx);
  const y0 = Math.floor(my * sy);
  const width = Math.max(1, Math.ceil((mx + w) * sx) - x0);
  const height = Math.max(1, Math.ceil((my + h) * sy) - y0);
  const alpha = new Uint8Array(width * height);
  const span = (row: number, c0: number, c1: number) => {
    const tx0 = Math.max(0, Math.floor((mx + c0) * sx) - x0);
    const tx1 = Math.min(width, Math.ceil((mx + c1) * sx) - x0);
    const ty0 = Math.max(0, Math.floor((my + row) * sy) - y0);
    const ty1 = Math.min(height, Math.ceil((my + row + 1) * sy) - y0);
    for (let ty = ty0; ty < ty1; ty++) alpha.fill(255, ty * width + tx0, ty * width + tx1);
  };
  let pos = 0;
  let fg = false;
  for (const run of runs) {
    if (fg) {
      // A foreground run may wrap across rows.
      let p = pos;
      const end = pos + run;
      while (p < end) {
        const row = Math.floor(p / w);
        const c0 = p - row * w;
        const c1 = Math.min(w, c0 + (end - p));
        span(row, c0, c1);
        p += c1 - c0;
      }
    }
    pos += run;
    fg = !fg;
  }
  return { x: x0, y: y0, width, height, alpha };
}

/** Padded box in preview pixels, clipped to the canvas: [x, y, w, h], or null. */
export function paddedRect(
  rect: Rect,
  sx: number,
  sy: number,
  canvasWidth: number,
  canvasHeight: number,
): [number, number, number, number] | null {
  const [x1, y1, x2, y2] = rect;
  const padX = (x2 - x1) * PADDING;
  const padY = (y2 - y1) * PADDING;
  const rx = Math.max(0, Math.floor((x1 - padX) * sx));
  const ry = Math.max(0, Math.floor((y1 - padY) * sy));
  const rw = Math.min(canvasWidth, Math.ceil((x2 + padX) * sx)) - rx;
  const rh = Math.min(canvasHeight, Math.ceil((y2 + padY) * sy)) - ry;
  if (rw <= 0 || rh <= 0) return null;
  return [rx, ry, rw, rh];
}

/**
 * What the preview obscures for a blurred region: blur is computed over
 * `effect` (the padded box grown to contain the silhouette) and shown only
 * where `raster` is opaque, or over all of `effect` when there is no mask.
 */
export function regionCoverage(
  region: { rect: Rect; mask?: MaskRow },
  sx: number,
  sy: number,
  canvasWidth: number,
  canvasHeight: number,
  raster?: MaskRaster,
): { effect: [number, number, number, number]; raster?: MaskRaster } | null {
  const box = paddedRect(region.rect, sx, sy, canvasWidth, canvasHeight);
  if (!region.mask) return box ? { effect: box } : null;
  const r = raster ?? rasterizeMask(region.mask, sx, sy);
  const x = Math.min(box ? box[0] : r.x, r.x);
  const y = Math.min(box ? box[1] : r.y, r.y);
  const x2 = Math.max(box ? box[0] + box[2] : r.x + r.width, r.x + r.width);
  const y2 = Math.max(box ? box[1] + box[3] : r.y + r.height, r.y + r.height);
  return { effect: [x, y, x2 - x, y2 - y], raster: r };
}

export interface FrameRegion {
  key: string;
  rect: Rect;
  blurred: boolean;
  kind: "track" | "manual";
  status?: TrackStatus;
  trackId?: number;
  interpolated?: boolean;
  /** Silhouette for this frame; absent means the padded box is the coverage. */
  mask?: MaskRow;
}

/** Every region visible on `frame`, with whether it will be blurred on export. */
export function regionsAtFrame(
  frame: number,
  tracks: IndexedTrack[],
  policy: Policy,
  overrides: Record<string, Override>,
  manual: ManualRegion[],
  /** Silhouettes from the on-disk store (already validated), if any. */
  maskAt?: (trackId: number, frame: number) => MaskRow | undefined,
): FrameRegion[] {
  const out: FrameRegion[] = [];
  for (const t of tracks) {
    if (frame < t.first_frame || frame > t.last_frame) continue;
    const hit = t.byFrame.get(frame);
    if (!hit) continue;
    out.push({
      key: `t${t.track_id}`,
      rect: hit.rect,
      blurred: shouldBlur(t, policy, overrides[t.track_id]),
      kind: "track",
      status: classify(t, policy),
      trackId: t.track_id,
      interpolated: hit.interpolated,
      ...(() => {
        if (hit.interpolated) return {};
        const mask = hit.mask ?? maskAt?.(t.track_id, frame);
        return mask ? { mask } : {};
      })(),
    });
  }
  for (const m of manual) {
    if (frame >= m.start_frame && frame <= m.end_frame) {
      out.push({ key: `m${m.region_id}`, rect: m.box, blurred: true, kind: "manual" });
    }
  }
  return out;
}

export interface Summary {
  minor: number;
  adult: number;
  uncertain: number;
  blurred: number;
  kept: number;
}

export function summarise(
  tracks: TrackData[],
  policy: Policy,
  overrides: Record<string, Override>,
): Summary {
  const s: Summary = { minor: 0, adult: 0, uncertain: 0, blurred: 0, kept: 0 };
  for (const t of tracks) {
    s[classify(t, policy)] += 1;
    if (shouldBlur(t, policy, overrides[t.track_id])) s.blurred += 1;
    else s.kept += 1;
  }
  return s;
}

/** Normalise a dragged rectangle (any corner order) and clamp it to the frame. */
export function normaliseRect(
  a: [number, number],
  b: [number, number],
  width: number,
  height: number,
): Rect | null {
  const clamp = (v: number, max: number) => Math.min(Math.max(v, 0), max);
  const x1 = clamp(Math.min(a[0], b[0]), width);
  const y1 = clamp(Math.min(a[1], b[1]), height);
  const x2 = clamp(Math.max(a[0], b[0]), width);
  const y2 = clamp(Math.max(a[1], b[1]), height);
  if (x2 - x1 < 4 || y2 - y1 < 4) return null;
  return [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}
