import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  AnalysisData,
  AppConfig,
  Job,
  ManualRegion,
  Override,
  Rect,
  RenderRequest,
} from "../api/types";
import {
  classify,
  formatTime,
  indexTracks,
  sensitivityToThreshold,
  shouldBlur,
  summarise,
  thresholdToSensitivity,
  timeOfFrame,
  type Policy,
} from "../review/model";
import { Timeline } from "./Timeline";
import { TrackList } from "./TrackList";
import { VideoStage, type ViewMode } from "./VideoStage";
import { MaskChunkCache } from "../review/masks";

interface Props {
  job: Job;
  config: AppConfig;
  initial: RenderRequest | null;
  onRender: (job: Job, request: RenderRequest) => void;
  onDiscard: () => void;
}

export function ReviewScreen({ job, config, initial, onRender, onDiscard }: Props) {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(initial?.threshold ?? config.default_threshold);
  const [minFaceSamples, setMinFaceSamples] = useState(
    initial?.min_face_samples ?? config.default_min_face_samples,
  );
  const [overrides, setOverrides] = useState<Record<string, Override>>(initial?.overrides ?? {});
  const [manual, setManual] = useState<ManualRegion[]>(initial?.manual_regions ?? []);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState<ViewMode>("preview");
  const [showBoxes, setShowBoxes] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [selectedTrack, setSelectedTrack] = useState<number | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [reveal, setReveal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(job.error);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    api.analysis(job.id).then(setAnalysis, (e: Error) => setLoadError(e.message));
  }, [job.id]);

  const policy: Policy = useMemo(() => ({ threshold, minFaceSamples }), [threshold, minFaceSamples]);
  const tracks = useMemo(
    () => indexTracks(analysis?.tracks ?? [], analysis?.video),
    [analysis],
  );
  const masks = useMemo(
    () =>
      analysis?.mask_store
        ? new MaskChunkCache((s, e) => api.masks(job.id, s, e), tracks, analysis.video)
        : null,
    [analysis, tracks, job.id],
  );
  const summary = useMemo(() => summarise(tracks, policy, overrides), [tracks, policy, overrides]);
  const video = analysis?.video ?? job.video!;
  const lastFrame = Math.max(video.frame_count - 1, 0);

  // Flagged segments (anything that will be blurred), for prev/next navigation.
  const flagged = useMemo(() => {
    const segs: [number, number][] = [];
    for (const t of tracks)
      if (shouldBlur(t, policy, overrides[t.track_id])) segs.push([t.first_frame, t.last_frame]);
    for (const m of manual) segs.push([m.start_frame, m.end_frame]);
    return segs.sort((a, b) => a[0] - b[0]);
  }, [tracks, policy, overrides, manual]);

  const seek = useCallback(
    (f: number) => {
      const v = videoRef.current;
      if (!v) return;
      const target = Math.min(Math.max(f, 0), lastFrame);
      v.currentTime = timeOfFrame(target, video.fps);
      setFrame(target);
    },
    [lastFrame, video.fps],
  );

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const on = () => setPlaying(true);
    const off = () => setPlaying(false);
    v.addEventListener("play", on);
    v.addEventListener("pause", off);
    v.addEventListener("ended", off);
    return () => {
      v.removeEventListener("play", on);
      v.removeEventListener("pause", off);
      v.removeEventListener("ended", off);
    };
  }, [analysis]);

  // Keyboard: space play/pause, ←/→ one frame, shift+←/→ one second, Esc cancels drawing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      const step = e.shiftKey ? Math.round(video.fps) : 1;
      if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        videoRef.current?.pause();
        seek(frame + step);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        videoRef.current?.pause();
        seek(frame - step);
      } else if (e.key === "Escape") {
        setDrawing(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, seek, togglePlay, video.fps]);

  const setOverride = useCallback((id: number, value: Override | null) => {
    setOverrides((o) => {
      const next = { ...o };
      if (value) next[id] = value;
      else delete next[id];
      return next;
    });
  }, []);

  const addRegion = useCallback(
    (rect: Rect) => {
      const id = Math.random().toString(36).slice(2, 10);
      const region: ManualRegion = {
        region_id: id,
        start_frame: frame,
        end_frame: Math.min(lastFrame, frame + Math.round(video.fps * 2)),
        box: rect,
      };
      setManual((m) => [...m, region]);
      setSelectedRegion(id);
      setDrawing(false);
    },
    [frame, lastFrame, video.fps],
  );

  const updateRegion = (id: string, patch: Partial<ManualRegion>) =>
    setManual((ms) =>
      ms.map((m) => {
        if (m.region_id !== id) return m;
        const next = { ...m, ...patch };
        next.start_frame = Math.min(Math.max(0, next.start_frame), lastFrame);
        next.end_frame = Math.min(Math.max(next.start_frame, next.end_frame), lastFrame);
        return next;
      }),
    );

  const keptFlagged = tracks.filter(
    (t) => overrides[t.track_id] === "keep" && classify(t, policy) !== "adult",
  ).length;

  const exportVideo = async () => {
    const request: RenderRequest = {
      threshold,
      min_face_samples: minFaceSamples,
      overrides,
      manual_regions: manual,
    };
    setSubmitting(true);
    setSubmitError(null);
    try {
      videoRef.current?.pause();
      onRender(await api.render(job.id, request), request);
    } catch (e) {
      setSubmitError((e as Error).message);
      setSubmitting(false);
    }
  };

  if (loadError) return <div className="card center error">Could not load analysis: {loadError}</div>;
  if (!analysis) return <div className="card center muted">Loading analysis…</div>;

  const nextFlag = flagged.find(([a]) => a > frame);
  const prevFlag = [...flagged].reverse().find(([a]) => a < frame);

  return (
    <section className="review">
      <div className="review-main">
        <div className="toolbar">
          <div className="segmented" role="radiogroup" aria-label="View">
            <button className={mode === "preview" ? "on" : ""} onClick={() => setMode("preview")}>
              Blur preview
            </button>
            <button className={mode === "boxes" ? "on" : ""} onClick={() => setMode("boxes")}>
              Original + boxes
            </button>
          </div>
          {mode === "preview" && (
            <label className="check inline">
              <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
              Outlines
            </label>
          )}
          <span className="spacer" />
          <button
            className={`btn ${drawing ? "primary" : ""}`}
            onClick={() => {
              videoRef.current?.pause();
              setDrawing((d) => !d);
            }}
            aria-pressed={drawing}
          >
            {drawing ? "Drag on the video… (Esc)" : "+ Add region"}
          </button>
        </div>

        <VideoStage
          src={api.previewUrl(job.id)}
          video={video}
          tracks={tracks}
          masks={masks}
          policy={policy}
          overrides={overrides}
          manual={manual}
          mode={mode}
          showBoxes={showBoxes}
          drawing={drawing}
          selectedTrack={selectedTrack}
          selectedRegion={selectedRegion}
          onFrame={setFrame}
          onDrawn={addRegion}
          onSelectTrack={setSelectedTrack}
          videoRef={videoRef}
        />

        <div className="transport">
          <button className="btn icon" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
            {playing ? "❚❚" : "▶"}
          </button>
          <button className="btn icon" onClick={() => seek(frame - 1)} aria-label="Previous frame">
            ‹
          </button>
          <button className="btn icon" onClick={() => seek(frame + 1)} aria-label="Next frame">
            ›
          </button>
          <span className="timecode">
            {formatTime(frame / video.fps)} · frame {frame}/{lastFrame}
          </span>
          <span className="spacer" />
          <button className="btn" disabled={!prevFlag} onClick={() => prevFlag && seek(prevFlag[0])}>
            ◀ Prev flagged
          </button>
          <button className="btn" disabled={!nextFlag} onClick={() => nextFlag && seek(nextFlag[0])}>
            Next flagged ▶
          </button>
        </div>

        <Timeline
          frameCount={video.frame_count}
          frame={frame}
          tracks={tracks}
          policy={policy}
          overrides={overrides}
          manual={manual}
          selectedTrack={selectedTrack}
          onSeek={(f) => {
            videoRef.current?.pause();
            seek(f);
          }}
          onSelectTrack={setSelectedTrack}
        />
        <p className="legend small muted">
          <span className="dot minor" /> likely minor <span className="dot uncertain" /> uncertain{" "}
          <span className="dot kept" /> kept <span className="dot manual" /> manual · dashed outline =
          interpolated through occlusion · keys: space, ←/→, shift+←/→
        </p>
        <p className="legend small muted">
          People are blurred to their silhouette when the model outlined them, and to their whole
          box on frames without a reliable outline. This preview is drawn over the original video
          in your browser; only the exported file has the blur baked in.
        </p>
      </div>

      <aside className="review-side">
        <div className="panel">
          <h3>Sensitivity</h3>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={thresholdToSensitivity(threshold)}
            onChange={(e) => setThreshold(sensitivityToThreshold(Number(e.target.value)))}
            aria-label="Sensitivity"
          />
          <div className="range-labels small muted">
            <span>Fewer blurs</span>
            <span>More blurs</span>
          </div>
          <p className="small">
            Blur when estimated P(under 18) ≥ <strong>{(threshold * 100).toFixed(0)}%</strong>
            {threshold !== config.default_threshold && (
              <>
                {" "}
                <button className="link" onClick={() => setThreshold(config.default_threshold)}>
                  reset
                </button>
              </>
            )}
          </p>
          <details>
            <summary className="small">Advanced</summary>
            <label className="small field">
              Face observations needed before someone can count as an adult
              <input
                type="number"
                min={1}
                max={100}
                value={minFaceSamples}
                onChange={(e) => setMinFaceSamples(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
          </details>
          <div className="summary">
            <div>
              <strong>{summary.blurred}</strong>
              <span>blurred</span>
            </div>
            <div>
              <strong>{summary.kept}</strong>
              <span>kept</span>
            </div>
            <div>
              <strong>{manual.length}</strong>
              <span>manual</span>
            </div>
          </div>
          <p className="small muted">
            {summary.minor} likely minor · {summary.uncertain} uncertain · {summary.adult} likely adult
          </p>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3>People ({tracks.length})</h3>
            <label className="check inline small">
              <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />
              Reveal thumbnails
            </label>
          </div>
          <TrackList
            thumbnailUrl={(id) => api.thumbnailUrl(job.id, id)}
            fps={video.fps}
            tracks={tracks}
            policy={policy}
            overrides={overrides}
            selected={selectedTrack}
            reveal={reveal}
            onSelect={setSelectedTrack}
            onJump={(f) => {
              videoRef.current?.pause();
              seek(f);
            }}
            onOverride={setOverride}
          />
        </div>

        <div className="panel">
          <h3>Manual regions ({manual.length})</h3>
          {manual.length === 0 && (
            <p className="small muted">
              Missed someone? Pause on the frame, click “+ Add region” and drag over them.
            </p>
          )}
          <ul className="regions">
            {manual.map((m, i) => (
              <li
                key={m.region_id}
                className={selectedRegion === m.region_id ? "selected" : ""}
                onClick={() => setSelectedRegion(m.region_id)}
              >
                <div className="region-head">
                  <strong>Region {i + 1}</strong>
                  <button className="link danger" onClick={() => setManual((ms) => ms.filter((x) => x.region_id !== m.region_id))}>
                    Remove
                  </button>
                </div>
                <div className="region-range small">
                  <label>
                    From
                    <input
                      type="number"
                      value={m.start_frame}
                      onChange={(e) => updateRegion(m.region_id, { start_frame: Number(e.target.value) })}
                    />
                  </label>
                  <button className="link" onClick={() => updateRegion(m.region_id, { start_frame: frame })}>
                    = now
                  </button>
                  <label>
                    to
                    <input
                      type="number"
                      value={m.end_frame}
                      onChange={(e) => updateRegion(m.region_id, { end_frame: Number(e.target.value) })}
                    />
                  </label>
                  <button className="link" onClick={() => updateRegion(m.region_id, { end_frame: frame })}>
                    = now
                  </button>
                </div>
                <button className="link small" onClick={() => seek(m.start_frame)}>
                  {formatTime(m.start_frame / video.fps)}–{formatTime(m.end_frame / video.fps)}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="panel export">
          {keptFlagged > 0 && (
            <p className="warn small" role="alert">
              You chose to keep {keptFlagged} {keptFlagged === 1 ? "person" : "people"} SafeFrame
              flagged as possibly under 18.
            </p>
          )}
          {submitError && <p className="error small">{submitError}</p>}
          <button className="btn primary big" onClick={exportVideo} disabled={submitting}>
            {submitting ? "Starting…" : "Export blurred video"}
          </button>
          <p className="small muted">
            {job.keep_upload
              ? "Your original will be kept so you can export again."
              : "Your original, the preview and thumbnails are deleted once the export is ready."}
          </p>
          <button className="link danger small" onClick={onDiscard}>
            Discard and delete everything
          </button>
        </div>
      </aside>
    </section>
  );
}
