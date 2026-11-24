import { memo, useRef } from "react";
import type { ManualRegion, Override } from "../api/types";
import { classify, shouldBlur, type IndexedTrack, type Policy } from "../review/model";

interface Props {
  frameCount: number;
  frame: number;
  tracks: IndexedTrack[];
  policy: Policy;
  overrides: Record<string, Override>;
  manual: ManualRegion[];
  selectedTrack: number | null;
  onSeek: (frame: number) => void;
  onSelectTrack: (id: number) => void;
}

/** One lane per person (and one for manual regions); click anywhere to seek. */
export const Timeline = memo(function Timeline(props: Props) {
  const { frameCount, frame, tracks, policy, overrides, manual, onSeek } = props;
  const ref = useRef<HTMLDivElement>(null);
  const pct = (f: number) => `${(f / Math.max(frameCount - 1, 1)) * 100}%`;
  const width = (a: number, b: number) =>
    `${Math.max(((b - a + 1) / Math.max(frameCount, 1)) * 100, 0.3)}%`;

  const seekFromEvent = (e: React.PointerEvent) => {
    const box = ref.current!.getBoundingClientRect();
    const x = Math.min(Math.max(e.clientX - box.left, 0), box.width);
    onSeek(Math.round((x / box.width) * (frameCount - 1)));
  };

  return (
    <div
      className="timeline"
      ref={ref}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        seekFromEvent(e);
      }}
      onPointerMove={(e) => e.buttons === 1 && seekFromEvent(e)}
      role="slider"
      aria-label="Timeline"
      aria-valuemin={0}
      aria-valuemax={frameCount - 1}
      aria-valuenow={frame}
    >
      {tracks.map((t) => {
        const blurred = shouldBlur(t, policy, overrides[t.track_id]);
        const status = classify(t, policy);
        return (
          <div className="lane" key={t.track_id}>
            <div
              className={`seg ${blurred ? status : "kept"}${props.selectedTrack === t.track_id ? " selected" : ""}`}
              style={{ left: pct(t.first_frame), width: width(t.first_frame, t.last_frame) }}
              title={`Person #${t.track_id} · ${blurred ? "blurred" : "kept"} (${status.replace("_", " ")})`}
              onPointerDown={(e) => {
                e.stopPropagation();
                props.onSelectTrack(t.track_id);
                onSeek(t.first_frame);
              }}
            />
            {t.band?.hits.map(([f, , ambiguous]) => (
              <div
                key={f}
                className={`band-tick${ambiguous ? " ambiguous" : ""}`}
                style={{ left: pct(f) }}
                title={`Band seen at frame ${f}${ambiguous ? " (ambiguous)" : ""}`}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  props.onSelectTrack(t.track_id);
                  onSeek(f);
                }}
              />
            ))}
          </div>
        );
      })}
      {manual.length > 0 && (
        <div className="lane">
          {manual.map((m) => (
            <div
              key={m.region_id}
              className="seg manual"
              style={{ left: pct(m.start_frame), width: width(m.start_frame, m.end_frame) }}
              title="Manual region"
            />
          ))}
        </div>
      )}
      {tracks.length === 0 && manual.length === 0 && (
        <div className="lane empty muted">No people detected</div>
      )}
      <div className="playhead" style={{ left: pct(frame) }} />
    </div>
  );
});
