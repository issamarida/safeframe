import { describe, expect, it, vi } from "vitest";
import type { MaskRow, TrackData } from "../api/types";
import { CHUNK_FRAMES, MAX_CHUNKS, MaskChunkCache, type MaskRange } from "./masks";
import { indexTracks, regionsAtFrame } from "./model";

const VIDEO = { width: 320, height: 240, frame_count: 300 };
const row = (f: number): MaskRow => [f, 10, 10, 4, 2, [1, 3, 2, 2]];

function tracks(interpolated: number[] = []) {
  const t: TrackData = {
    track_id: 1,
    first_frame: 0,
    last_frame: 299,
    p_minor: null,
    expected_age: null,
    face_samples: 0,
    peak_confidence: 0.9,
    boxes: Array.from({ length: 300 }, (_, f) => [f, 10, 10, 50, 90, interpolated.includes(f) ? 1 : 0]),
  };
  return indexTracks([t], VIDEO);
}

const flush = () => new Promise((r) => setTimeout(r, 0));

/** A fake server that returns at most `limit` frames per call. */
function server(limit = 1000, frames: (f: number) => unknown[] = (f) => [row(f)]) {
  return vi.fn(async (start: number, end: number): Promise<MaskRange> => {
    const stop = Math.min(end, start + limit);
    const rows = [];
    for (let f = start; f < stop; f++) rows.push(...frames(f));
    return { start, end: stop, tracks: { "1": rows } };
  });
}

describe("MaskChunkCache", () => {
  it("loads the playhead chunk and the next, then serves lookups", async () => {
    const fetch = server();
    const cache = new MaskChunkCache(fetch, tracks(), VIDEO);
    expect(cache.lookup(1, 5)).toBeUndefined(); // not loaded yet: box
    cache.ensure(5);
    await flush();
    expect(fetch.mock.calls).toEqual([
      [0, CHUNK_FRAMES],
      [CHUNK_FRAMES, 2 * CHUNK_FRAMES],
    ]);
    expect(cache.lookup(1, 5)).toEqual(row(5));
    expect(cache.lookup(1, CHUNK_FRAMES + 1)).toEqual(row(CHUNK_FRAMES + 1));
    expect(cache.lookup(2, 5)).toBeUndefined();
    cache.ensure(6);
    await flush();
    expect(fetch).toHaveBeenCalledTimes(2); // cached, no refetch
  });

  it("continues truncated responses and notifies subscribers", async () => {
    const fetch = server(10);
    const cache = new MaskChunkCache(fetch, tracks(), VIDEO);
    const seen = vi.fn();
    cache.subscribe(seen);
    cache.ensure(0);
    for (let i = 0; i < 20; i++) await flush();
    expect(cache.lookup(1, 63)).toEqual(row(63));
    // Chunk 0's requests (chunk 1 loads concurrently).
    expect(fetch.mock.calls.filter(([, end]) => end === 64).slice(0, 3)).toEqual([
      [0, 64],
      [10, 64],
      [20, 64],
    ]);
    expect(seen).toHaveBeenCalled();
  });

  it("rejects invalid, misplaced, interpolated and duplicate rows", async () => {
    const fetch = server(1000, (f) =>
      f === 1
        ? [[1, 10, 10, 4, 2, [1, 3, 2, 9]]] // bad run sum
        : f === 2
          ? [row(2)] // interpolated frame
          : f === 3
            ? [row(3), row(3)] // ambiguous
            : f === 4
              ? [row(99)] // outside the returned range
              : [row(f)],
    );
    const cache = new MaskChunkCache(fetch, tracks([2]), VIDEO);
    cache.ensure(0);
    await flush();
    expect([0, 1, 2, 3, 4, 5].map((f) => !!cache.lookup(1, f))).toEqual([
      true,
      false,
      false,
      false,
      false,
      true,
    ]);
  });

  it("ignores a response that claims more than was asked", async () => {
    const fetch = vi.fn(async (start: number) => ({ start, end: 10_000, tracks: { "1": [row(start)] } }));
    const cache = new MaskChunkCache(fetch, tracks(), VIDEO);
    cache.ensure(0);
    await flush();
    expect(cache.lookup(1, 0)).toBeUndefined();
  });

  it("network failure means box fallback, retried later", async () => {
    const fetch = vi.fn(async () => {
      throw new Error("offline");
    });
    const cache = new MaskChunkCache(fetch, tracks(), VIDEO);
    cache.ensure(0);
    await flush();
    expect(cache.lookup(1, 0)).toBeUndefined();
    cache.ensure(0);
    await flush();
    expect(fetch).toHaveBeenCalledTimes(2); // both chunks once; no hot retry loop
  });

  it("keeps a bounded number of chunks", async () => {
    const long = { ...VIDEO, frame_count: CHUNK_FRAMES * (MAX_CHUNKS + 4) };
    const fetch = server();
    const t = tracks();
    const cache = new MaskChunkCache(fetch, t, long);
    for (let c = 0; c < MAX_CHUNKS + 3; c++) {
      cache.ensure(c * CHUNK_FRAMES);
      await flush();
    }
    expect(cache.lookup(1, 0)).toBeUndefined(); // evicted
  });

  it("feeds regionsAtFrame, which uses the box until a silhouette is loaded", async () => {
    const t = tracks([7]);
    const cache = new MaskChunkCache(server(), t, VIDEO);
    const policy = { threshold: 0.35, minFaceSamples: 3 };
    const at = (f: number) =>
      regionsAtFrame(f, t, policy, {}, [], (id, fr) => cache.lookup(id, fr))[0]!;
    expect(at(6).mask).toBeUndefined();
    cache.ensure(6);
    await flush();
    expect(at(6).mask).toEqual(row(6));
    expect(at(7).mask).toBeUndefined(); // interpolated frame never gets a mask
  });
});
