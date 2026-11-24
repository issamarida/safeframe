import { useEffect, useState } from "react";
import { watchJob } from "../api/client";
import type { Job } from "../api/types";
import { formatTime } from "../review/model";
import { ProgressBar } from "./ProgressBar";

interface Props {
  job: Job;
  onUpdate: (job: Job) => void;
  onCancel: () => void;
}

const STEPS = ["Preparing preview", "Loading models", "Finding people and SafeFrame Bands"];

export function ProcessingScreen({ job, onUpdate, onCancel }: Props) {
  const [error, setError] = useState<string | null>(null);
  const [startedAt] = useState(() => Date.now());
  const [now, setNow] = useState(Date.now());

  useEffect(() => watchJob(job.id, onUpdate, (e) => setError(e.message)), [job.id, onUpdate]);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  const elapsed = (now - startedAt) / 1000;
  const eta = job.progress > 0.15 ? (elapsed / job.progress) * (1 - job.progress) : null;
  const current = STEPS.indexOf(job.stage);

  return (
    <section className="card processing">
      <h2>Analysing “{job.filename}”</h2>
      {job.video && job.media !== "photo" && (
        <p className="muted">
          {job.video.width}×{job.video.height} · {job.video.fps.toFixed(2)} fps ·{" "}
          {formatTime(job.video.duration)} · {job.video.frame_count} frames
          {job.video.has_audio ? " · audio" : " · no audio"}
        </p>
      )}
      <ol className="steps">
        {STEPS.map((s, i) => (
          <li key={s} className={i < current ? "done" : i === current ? "active" : ""}>
            {s}
          </li>
        ))}
      </ol>
      <ProgressBar value={job.progress} label={job.state === "queued" ? "Waiting in queue" : job.stage} />
      <p className="muted small">
        Elapsed {formatTime(elapsed)}
        {eta !== null && <> · about {formatTime(eta)} remaining</>}
      </p>
      {error && <p className="error">{error}</p>}
      <button className="btn" onClick={onCancel}>
        Cancel and delete
      </button>
    </section>
  );
}
