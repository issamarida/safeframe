import { memo, useState } from "react";
import type { Override } from "../api/types";
import {
  ageMatters,
  BAND_DECISIONS,
  BLURRED_DECISIONS,
  classify,
  formatTime,
  shouldBlur,
  type Decision,
  type IndexedTrack,
  type Policy,
} from "../review/model";

interface Props {
  thumbnailUrl: (id: number) => string;
  fps: number;
  tracks: IndexedTrack[];
  policy: Policy;
  overrides: Record<string, Override>;
  selected: number | null;
  reveal: boolean;
  onSelect: (id: number) => void;
  onJump: (frame: number) => void;
  onOverride: (id: number, value: Override | null) => void;
}

const STATUS_TEXT: Record<Decision, string> = {
  wearer: "Wearing a band",
  possible: "Possible band",
  not_honoured: "Band not honoured",
  minor: "Likely under 18",
  uncertain: "Age uncertain",
  visible: "No band",
};

const STATUS_HELP: Record<Decision, string> = {
  wearer: "A SafeFrame Band was seen on this person: they asked not to be shown.",
  possible:
    "A band was seen where it could belong to this person or someone next to them, so both are blurred.",
  not_honoured:
    "Guarded place: a band only protects someone who looks under 18 here. A band is a privacy request, not a disguise.",
  minor: "Event mode: this person may be under 18.",
  uncertain: "Event mode: not enough of their face was seen to tell their age.",
  visible: "No band was seen on this person.",
};

export const TrackList = memo(function TrackList(props: Props) {
  const { tracks, policy, overrides, fps } = props;
  const [confirming, setConfirming] = useState<number | null>(null);
  if (tracks.length === 0) {
    return (
      <p className="muted small">
        No people were detected. If someone was missed, add a manual region.
      </p>
    );
  }
  const showAge = ageMatters(policy);
  return (
    <ul className="tracks">
      {tracks.map((t) => {
        const status = classify(t, policy);
        const override = overrides[t.track_id];
        const blurred = shouldBlur(t, policy, override);
        const band = t.band;
        const sightings = band ? band.confirmed + band.possible : 0;
        const choose = (value: Override | null) => {
          // Keeping someone who asked not to be shown needs a second, explicit click.
          if (value === "keep" && BAND_DECISIONS.has(status) && confirming !== t.track_id) {
            setConfirming(t.track_id);
            return;
          }
          setConfirming(null);
          props.onOverride(t.track_id, value);
        };
        return (
          <li
            key={t.track_id}
            className={`track${props.selected === t.track_id ? " selected" : ""}${blurred ? "" : " is-kept"}`}
            onClick={() => props.onSelect(t.track_id)}
          >
            <img
              className={`thumb${props.reveal ? "" : " veiled"}`}
              src={props.thumbnailUrl(t.track_id)}
              alt={`Person #${t.track_id}`}
              loading="lazy"
              onError={(e) => (e.currentTarget.style.visibility = "hidden")}
            />
            <div className="track-body">
              <div className="track-head">
                <strong>Person #{t.track_id}</strong>
                <span className={`badge ${status}`} title={STATUS_HELP[status]}>
                  {STATUS_TEXT[status]}
                </span>
              </div>
              {sightings > 0 && band && (
                <div className="track-meta small">
                  Band seen {sightings}×
                  {band.possible > 0 && <> ({band.possible} ambiguous)</>}
                  {band.first_frame !== null && (
                    <>
                      {" "}
                      ·{" "}
                      <button
                        className="link"
                        onClick={(e) => {
                          e.stopPropagation();
                          props.onJump(band.first_frame!);
                        }}
                      >
                        first at {formatTime(band.first_frame / fps)}
                      </button>
                    </>
                  )}
                </div>
              )}
              {showAge && (t.face_samples > 0 || policy.protectMinors || sightings > 0) && (
                <>
                  <div className="meter" title="Estimated probability of being under 18">
                    <div
                      className="meter-fill"
                      style={{ width: `${(t.p_minor ?? 0) * 100}%` }}
                      data-empty={t.p_minor === null}
                    />
                    <div className="meter-threshold" style={{ left: `${policy.threshold * 100}%` }} />
                  </div>
                  <div className="track-meta muted small">
                    {t.p_minor === null ? "no usable face" : `P(<18) ${(t.p_minor * 100).toFixed(0)}%`}
                    {t.expected_age !== null && <> · ~{Math.round(t.expected_age)} y</>} ·{" "}
                    {t.face_samples} face{t.face_samples === 1 ? "" : "s"}
                  </div>
                </>
              )}
              <div className="track-meta muted small">
                <button
                  className="link"
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onJump(t.first_frame);
                  }}
                >
                  {formatTime(t.first_frame / fps)}–{formatTime(t.last_frame / fps)}
                </button>
              </div>
              <div className="segmented" role="radiogroup" aria-label={`Decision for person #${t.track_id}`}>
                {(
                  [
                    [null, `Auto · ${BLURRED_DECISIONS.has(status) ? "blur" : "keep"}`],
                    ["blur", "Blur"],
                    ["keep", "Keep"],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={label}
                    role="radio"
                    aria-checked={(override ?? null) === value}
                    className={(override ?? null) === value ? "on" : ""}
                    onClick={(e) => {
                      e.stopPropagation();
                      choose(value);
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {confirming === t.track_id && (
                <div className="confirm warn small" role="alert" onClick={(e) => e.stopPropagation()}>
                  <span>
                    Person #{t.track_id} is wearing a band: they asked not to be shown. Keep them
                    visible only if they agreed.
                  </span>
                  <span className="row">
                    <button className="btn small danger" onClick={() => choose("keep")}>
                      Keep anyway
                    </button>
                    <button className="btn small" onClick={() => setConfirming(null)}>
                      Cancel
                    </button>
                  </span>
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
});
