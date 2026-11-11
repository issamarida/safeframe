import { useCallback, useEffect, useRef, useState } from "react";
import type { ManualRegion, Override, Rect, VideoInfo } from "../api/types";
import type { MaskChunkCache } from "../review/masks";
import {
  frameAtTime,
  normaliseRect,
  paddedRect,
  rasterizeMask,
  regionCoverage,
  regionsAtFrame,
  type FrameRegion,
  type IndexedTrack,
  type MaskRaster,
  type Policy,
} from "../review/model";

export type ViewMode = "preview" | "boxes";

interface Props {
  src: string;
  video: VideoInfo;
  tracks: IndexedTrack[];
  policy: Policy;
  overrides: Record<string, Override>;
  manual: ManualRegion[];
  mode: ViewMode;
  showBoxes: boolean;
  drawing: boolean;
  selectedTrack: number | null;
  selectedRegion: string | null;
  onFrame: (frame: number) => void;
  onDrawn: (rect: Rect) => void;
  onSelectTrack: (id: number | null) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Silhouettes loaded by frame range; absent for inline or box-only analyses. */
  masks?: MaskChunkCache | null;
}

// Decoded silhouette rasters kept around for scrubbing back and forth.
const MASK_CACHE_SIZE = 64;

/** Alpha-only canvas for one mask at one preview scale, LRU-cached. */
function useMaskCache() {
  const cache = useRef(new Map<string, { canvas: HTMLCanvasElement; raster: MaskRaster }>());
  return useCallback((r: FrameRegion, frame: number, sx: number, sy: number) => {
    const key = `${r.trackId}:${frame}:${sx}:${sy}`;
    const hit = cache.current.get(key);
    if (hit) {
      cache.current.delete(key); // refresh LRU position
      cache.current.set(key, hit);
      return hit;
    }
    const raster = rasterizeMask(r.mask!, sx, sy);
    const canvas = document.createElement("canvas");
    canvas.width = raster.width;
    canvas.height = raster.height;
    const img = new ImageData(raster.width, raster.height);
    for (let i = 0; i < raster.alpha.length; i++) img.data[i * 4 + 3] = raster.alpha[i]!;
    canvas.getContext("2d")!.putImageData(img, 0, 0);
    const entry = { canvas, raster };
    cache.current.set(key, entry);
    while (cache.current.size > MASK_CACHE_SIZE) {
      cache.current.delete(cache.current.keys().next().value!);
    }
    return entry;
  }, []);
}

const COLORS = {
  minor: "#e5484d",
  uncertain: "#f5a524",
  adult: "#30a46c",
  manual: "#8e4ec6",
  kept: "#30a46c",
};

export function VideoStage(props: Props) {
  const { src, video, drawing, onFrame, onDrawn, videoRef } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef(0);
  const dragRef = useRef<{ a: [number, number]; b: [number, number] } | null>(null);
  const latest = useRef(props);
  latest.current = props;
  const [filterSupported] = useState(() => {
    const ctx = document.createElement("canvas").getContext("2d");
    return !!ctx && "filter" in ctx;
  });

  const maskCanvas = useMaskCache();
  const scratchRef = useRef<HTMLCanvasElement | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const v = videoRef.current;
    if (!canvas || !v || !v.videoWidth) return;
    const p = latest.current;
    if (canvas.width !== v.videoWidth || canvas.height !== v.videoHeight) {
      canvas.width = v.videoWidth;
      canvas.height = v.videoHeight;
    }
    const ctx = canvas.getContext("2d")!;
    // Source px -> preview px, per axis (the preview may not scale uniformly).
    const sx = canvas.width / p.video.width;
    const sy = canvas.height / p.video.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frame = frameRef.current;
    p.masks?.ensure(frame);
    const maskAt = p.masks ? (t: number, f: number) => p.masks!.lookup(t, f) : undefined;
    const regions = regionsAtFrame(frame, p.tracks, p.policy, p.overrides, p.manual, maskAt);

    // Pass 1: obscure. Each effect is drawn from the untouched video and only
    // ever adds coverage, so a kept region can never reveal blurred pixels.
    if (p.mode === "preview") {
      for (const r of regions) {
        if (!r.blurred) continue;
        const cached = r.mask ? maskCanvas(r, frame, sx, sy) : undefined;
        const cov = regionCoverage(r, sx, sy, canvas.width, canvas.height, cached?.raster);
        if (!cov) continue;
        const [ex, ey, ew, eh] = cov.effect;
        if (!cached) {
          ctx.save();
          ctx.beginPath();
          ctx.rect(ex, ey, ew, eh);
          ctx.clip();
          paintEffect(ctx, v, ex, ey, ew, eh, filterSupported);
          ctx.restore();
          continue;
        }
        // Silhouette: blur into a scratch surface, keep only the mask's alpha,
        // then composite onto the review canvas.
        const scratch = (scratchRef.current ??= document.createElement("canvas"));
        if (scratch.width < ew || scratch.height < eh) {
          scratch.width = Math.max(scratch.width, ew);
          scratch.height = Math.max(scratch.height, eh);
        }
        const sctx = scratch.getContext("2d")!;
        sctx.globalCompositeOperation = "source-over";
        sctx.clearRect(0, 0, scratch.width, scratch.height);
        sctx.save();
        sctx.translate(-ex, -ey);
        sctx.beginPath();
        sctx.rect(ex, ey, ew, eh);
        sctx.clip();
        paintEffect(sctx, v, ex, ey, ew, eh, filterSupported);
        sctx.restore();
        sctx.globalCompositeOperation = "destination-in";
        sctx.imageSmoothingEnabled = false;
        sctx.drawImage(cached.canvas, cached.raster.x - ex, cached.raster.y - ey);
        sctx.globalCompositeOperation = "source-over";
        ctx.drawImage(scratch, 0, 0, ew, eh, ex, ey, ew, eh);
      }
    }

    // Pass 2: diagnostics on top of every effect.
    for (const r of regions) {
      if (!(p.showBoxes || p.mode === "boxes")) break;
      const rect = paddedRect(r.rect, sx, sy, canvas.width, canvas.height);
      if (!rect) continue;
      const [rx, ry, rw, rh] = rect;
      const selected =
        (r.trackId !== undefined && r.trackId === p.selectedTrack) ||
        (r.kind === "manual" && r.key === `m${p.selectedRegion}`);
      const color =
        r.kind === "manual" ? COLORS.manual : r.blurred ? COLORS[r.status ?? "minor"] : COLORS.kept;
      ctx.lineWidth = selected ? 4 : 2;
      ctx.strokeStyle = color;
      ctx.setLineDash(r.interpolated ? [6, 4] : r.blurred ? [] : [2, 3]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
      // Say how a blurred person is covered on this frame: by their silhouette
      // or, with no usable mask, by the whole padded box.
      const how = r.blurred ? (r.mask ? " · silhouette" : " · box") : "";
      const label =
        r.kind === "manual" ? "manual" : `#${r.trackId} ${r.blurred ? "blur" : "keep"}${how}`;
      ctx.font = "600 12px system-ui, sans-serif";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(rx, Math.max(0, ry - 16), tw, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rx + 4, Math.max(12, ry - 4));
    }

    const drag = dragRef.current;
    if (drag) {
      const [ax, ay] = drag.a;
      const [bx, by] = drag.b;
      ctx.strokeStyle = COLORS.manual;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(
        Math.min(ax, bx) * sx,
        Math.min(ay, by) * sy,
        Math.abs(bx - ax) * sx,
        Math.abs(by - ay) * sy,
      );
      ctx.setLineDash([]);
    }
  }, [filterSupported, maskCanvas, videoRef]);

  // Redraw when silhouettes for nearby frames arrive.
  useEffect(() => props.masks?.subscribe(draw), [draw, props.masks]);

  // Redraw whenever decisions change.
  useEffect(draw, [draw, props.policy, props.overrides, props.manual, props.mode, props.showBoxes,
    props.selectedTrack, props.selectedRegion, props.tracks]);

  // Keep the overlay locked to the displayed frame.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const fps = video.fps;
    let handle = 0;
    const update = (t: number) => {
      const f = Math.min(frameAtTime(t, fps), video.frame_count - 1);
      frameRef.current = f;
      onFrame(f);
      draw();
    };
    const hasRvfc = "requestVideoFrameCallback" in v;
    const onVideoFrame = (_: number, meta: { mediaTime: number }) => {
      update(meta.mediaTime);
      handle = v.requestVideoFrameCallback(onVideoFrame);
    };
    let raf = 0;
    const onRaf = () => {
      update(v.currentTime);
      if (!v.paused) raf = requestAnimationFrame(onRaf);
    };
    const onPlay = () => {
      if (!hasRvfc) raf = requestAnimationFrame(onRaf);
    };
    const onStill = () => update(v.currentTime);
    if (hasRvfc) handle = v.requestVideoFrameCallback(onVideoFrame);
    v.addEventListener("play", onPlay);
    v.addEventListener("seeked", onStill);
    v.addEventListener("loadeddata", onStill);
    v.addEventListener("pause", onStill);
    return () => {
      if (hasRvfc) v.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("seeked", onStill);
      v.removeEventListener("loadeddata", onStill);
      v.removeEventListener("pause", onStill);
    };
  }, [draw, onFrame, video.fps, video.frame_count, videoRef]);

  const toSource = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const box = e.currentTarget.getBoundingClientRect();
    return [
      ((e.clientX - box.left) / box.width) * video.width,
      ((e.clientY - box.top) / box.height) * video.height,
    ];
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const pt = toSource(e);
    if (drawing) {
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = { a: pt, b: pt };
      return;
    }
    // Click a box to select its track.
    const p = latest.current;
    const hit = regionsAtFrame(frameRef.current, p.tracks, p.policy, p.overrides, [])
      .filter((r) => r.rect[0] <= pt[0] && pt[0] <= r.rect[2] && r.rect[1] <= pt[1] && pt[1] <= r.rect[3])
      .sort((a, b) => area(a.rect) - area(b.rect))[0];
    p.onSelectTrack(hit?.trackId ?? null);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    dragRef.current.b = toSource(e);
    draw();
  };
  const onPointerUp = () => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    const rect = normaliseRect(drag.a, drag.b, video.width, video.height);
    if (rect) onDrawn(rect);
    draw();
  };

  return (
    <div className="stage" style={{ aspectRatio: `${video.width} / ${video.height}` }}>
      <video ref={videoRef} src={src} preload="auto" playsInline muted crossOrigin="anonymous" />
      <canvas
        ref={canvasRef}
        className={drawing ? "drawing" : ""}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        aria-label={drawing ? "Drag to draw a region to blur" : "Video preview with detected people"}
      />
    </div>
  );
}

const area = (r: Rect) => (r[2] - r[0]) * (r[3] - r[1]);

/**
 * Paint the blur + greyscale effect for the preview-pixel rect (x, y, w, h)
 * from the untouched video into `ctx` (caller sets the clip). Without Canvas
 * filter support the area is filled opaque grey instead: still fully covered.
 */
function paintEffect(
  ctx: CanvasRenderingContext2D,
  v: HTMLVideoElement,
  x: number,
  y: number,
  w: number,
  h: number,
  filterSupported: boolean,
) {
  // Opaque base: near the frame edge the blur filter pulls in transparent
  // pixels from outside the video, which would let the original show through.
  ctx.fillStyle = "#808080";
  ctx.fillRect(x, y, w, h);
  if (filterSupported) {
    const radius = Math.max(4, Math.max(w, h) / 14);
    const m = radius * 3;
    ctx.filter = `blur(${radius}px) grayscale(1)`;
    ctx.drawImage(v, x - m, y - m, w + 2 * m, h + 2 * m, x - m, y - m, w + 2 * m, h + 2 * m);
    ctx.filter = "none";
  }
}
