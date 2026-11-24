import { useEffect, useState } from "react";
import { api, watchJob } from "../api/client";
import type { AppConfig, Job } from "../api/types";
import { ProgressBar } from "./ProgressBar";

interface Props {
  job: Job;
  config: AppConfig;
  onUpdate: (job: Job) => void;
  onBackToReview: () => void;
  onDelete: () => void;
  onStartOver: () => void;
}

export function ExportScreen({ job, config, onUpdate, onBackToReview, onDelete, onStartOver }: Props) {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (job.state !== "rendering") return;
    return watchJob(job.id, onUpdate, (e) => setError(e.message));
  }, [job.id, job.state, onUpdate]);

  if (job.state === "rendering") {
    return (
      <section className="card processing">
        <h2>Exporting “{job.filename}”</h2>
        <p className="muted">
          {job.media === "photo"
            ? "Blurring and greyscaling everyone protected, at the original resolution."
            : "Blurring and greyscaling everyone protected, then encoding at the original resolution and frame rate with the original audio."}
        </p>
        <ProgressBar value={job.progress} label={job.stage} />
        {error && <p className="error">{error}</p>}
        <button className="btn" onClick={onDelete}>
          Cancel and delete
        </button>
      </section>
    );
  }

  const expires = new Date(job.expires_at * 1000);
  const photo = job.media === "photo";
  const ext = photo ? (job.filename.split(".").pop() ?? "jpg").toUpperCase() : "MP4";
  return (
    <section className="card done">
      <div className="done-icon" aria-hidden>
        ✓
      </div>
      <h2>Your blurred {photo ? "photo" : "video"} is ready</h2>
      <a className="btn primary big" href={api.downloadUrl(job.id)} download>
        Download {ext === "JPG" ? "JPEG" : ext}
      </a>
      <ul className="facts small">
        <li>
          {photo
            ? "Original resolution preserved."
            : "Original resolution, frame rate and audio preserved."}
        </li>
        <li>All metadata removed, including location and device.</li>
        <li>
          {job.upload_deleted
            ? "Your original upload, the preview and thumbnails have been deleted from the server."
            : "Your original upload is being kept, as you asked."}
        </li>
        <li>
          Everything left for this job is deleted automatically after{" "}
          {expires.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}.
        </li>
      </ul>
      {job.sealed_ready && (
        <div className="panel sealed small">
          <strong>Sealed original</strong>
          <p>
            Someone was hidden because of their band, so the unblurred original was sealed instead
            of simply deleted. Only a quorum of independent trustees can open it, for example on a
            court order, and it expires after {config.seal_retention_days} days. Not you, not this
            server: nobody can open it alone.
          </p>
          <a className="btn" href={api.sealedUrl(job.id)} download>
            Download sealed envelope (.sfseal)
          </a>
        </div>
      )}
      <p className="warn small">
        Please {photo ? "check" : "watch"} the result before sharing it: a band that was never
        visible can’t be honoured.
      </p>
      <div className="row">
        {!job.upload_deleted && (
          <button className="btn" onClick={onBackToReview}>
            Back to review
          </button>
        )}
        <button className="btn danger" onClick={onDelete}>
          Delete everything now
        </button>
        <button className="btn" onClick={onStartOver}>
          Process another file
        </button>
      </div>
    </section>
  );
}
