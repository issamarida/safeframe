/**
 * Silhouettes fetched by frame range from the job's on-disk mask store
 * (backend/src/safeframe/pipeline/maskstore.py, GET /api/jobs/{id}/masks).
 *
 * Only chunks around the playhead are loaded and a bounded number are kept.
 * Anything not loaded (yet), invalid, or on a frame without an observed box
 * resolves to "no mask", which the preview draws as the padded box: the
 * conservative choice, never less coverage than the export.
 */
import type { MaskRow } from "../api/types";
import { parseMaskRow, type IndexedTrack } from "./model";

export interface MaskRange {
  start: number;
  end: number;
  tracks: Record<string, unknown>;
}

export type FetchRange = (start: number, end: number) => Promise<MaskRange>;

export const CHUNK_FRAMES = 64;
export const MAX_CHUNKS = 16;
const RETRY_MS = 2000;

interface Chunk {
  masks: Map<string, MaskRow>;
  /** Exclusive end of the frames this chunk has received so far. */
  loadedTo: number;
}

export class MaskChunkCache {
  private readonly fetchRange: FetchRange;
  private readonly tracks: Map<number, IndexedTrack>;
  private readonly width: number;
  private readonly height: number;
  private readonly frameCount: number;
  private readonly chunks = new Map<number, Chunk>();
  private readonly inflight = new Set<number>();
  private readonly failedAt = new Map<number, number>();
  private readonly listeners = new Set<() => void>();

  constructor(
    fetchRange: FetchRange,
    tracks: IndexedTrack[],
    video: { width: number; height: number; frame_count: number },
  ) {
    this.fetchRange = fetchRange;
    this.tracks = new Map(tracks.map((t) => [t.track_id, t]));
    this.width = video.width;
    this.height = video.height;
    this.frameCount = video.frame_count;
  }

  /** Called whenever newly loaded masks may change what is drawn. */
  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  lookup(trackId: number, frame: number): MaskRow | undefined {
    const c = Math.floor(frame / CHUNK_FRAMES);
    const chunk = this.chunks.get(c);
    if (!chunk || frame >= chunk.loadedTo) return undefined;
    return chunk.masks.get(`${trackId}:${frame}`);
  }

  /** Start loading the chunk containing `frame` and the one after it. */
  ensure(frame: number): void {
    const c = Math.floor(Math.max(0, frame) / CHUNK_FRAMES);
    for (const k of [c, c + 1]) {
      if (k * CHUNK_FRAMES >= this.frameCount) continue;
      const chunk = this.chunks.get(k);
      if (chunk) {
        this.chunks.delete(k); // refresh LRU position
        this.chunks.set(k, chunk);
        continue;
      }
      if (this.inflight.has(k)) continue;
      if (Date.now() - (this.failedAt.get(k) ?? -Infinity) < RETRY_MS) continue;
      void this.load(k);
    }
  }

  private async load(k: number): Promise<void> {
    const start = k * CHUNK_FRAMES;
    const end = Math.min(start + CHUNK_FRAMES, this.frameCount);
    const chunk: Chunk = { masks: new Map(), loadedTo: start };
    const duplicates = new Set<string>();
    this.inflight.add(k);
    try {
      let from = start;
      while (from < end) {
        const res = await this.fetchRange(from, end);
        // The server may cover less than asked (bounded responses), never more.
        if (res.start !== from || !Number.isInteger(res.end) || res.end <= from || res.end > end) {
          break;
        }
        this.accept(res, from, res.end, chunk, duplicates);
        from = res.end;
        chunk.loadedTo = from;
        this.store(k, chunk);
        this.notify();
      }
    } catch {
      this.failedAt.set(k, Date.now());
    } finally {
      this.inflight.delete(k);
    }
  }

  private accept(res: MaskRange, from: number, to: number, chunk: Chunk, dups: Set<string>) {
    if (typeof res.tracks !== "object" || res.tracks === null) return;
    for (const [key, rows] of Object.entries(res.tracks)) {
      const trackId = Number(key);
      const track = this.tracks.get(trackId);
      if (!track || !Array.isArray(rows)) continue;
      for (const row of rows) {
        const mask = parseMaskRow(row, this.width, this.height);
        if (!mask) continue;
        const f = mask[0];
        const entry = track.byFrame.get(f);
        if (f < from || f >= to || !entry || entry.interpolated) continue;
        const id = `${trackId}:${f}`;
        if (chunk.masks.has(id) || dups.has(id)) {
          dups.add(id); // ambiguous: use the box
          chunk.masks.delete(id);
          continue;
        }
        chunk.masks.set(id, mask);
      }
    }
  }

  private store(k: number, chunk: Chunk) {
    this.chunks.delete(k);
    this.chunks.set(k, chunk);
    while (this.chunks.size > MAX_CHUNKS) {
      this.chunks.delete(this.chunks.keys().next().value!);
    }
  }

  private notify() {
    for (const fn of this.listeners) fn();
  }
}
