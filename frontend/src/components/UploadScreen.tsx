import { useRef, useState } from "react";
import { api, uploadMedia } from "../api/client";
import type { AppConfig, Job } from "../api/types";
import { ProgressBar } from "./ProgressBar";

interface Props {
  config: AppConfig;
  onUploaded: (job: Job) => void;
}

export function UploadScreen({ config, onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [keep, setKeep] = useState(false);
  const [guarded, setGuarded] = useState(false);
  const [eventMode, setEventMode] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = (f: File | undefined) => {
    setError(null);
    if (!f) return;
    if (f.size > config.max_upload_mb * 1024 * 1024) {
      setError(`That file is larger than the ${config.max_upload_mb} MB limit.`);
      return;
    }
    setFile(f);
  };

  const start = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    const { promise, abort } = uploadMedia(
      file,
      { keepUpload: keep, guarded, protectMinors: eventMode },
      setProgress,
    );
    abortRef.current = abort;
    try {
      onUploaded(await promise);
    } catch (e) {
      setError((e as Error).message);
      setProgress(null);
    } finally {
      abortRef.current = null;
    }
  };

  const uploading = progress !== null;

  return (
    <section className="card upload">
      <h1>Blur everyone who asked not to be shown</h1>
      <p className="lead">
        Anyone wearing a <strong>SafeFrame Band</strong> is blurred in every frame they appear in,
        even where the band itself is hidden. Everyone else and the background stay untouched. You
        review every decision before anything is exported.{" "}
        <a href={api.bandSheetUrl()} target="_blank" rel="noreferrer">
          Print a band
        </a>
      </p>

      <div
        className={`dropzone${dragging ? " dragging" : ""}${file ? " has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files[0]);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        aria-label="Choose a photo or video"
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*,image/jpeg,image/png,image/webp"
          hidden
          onChange={(e) => choose(e.target.files?.[0])}
        />
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB · click to change</span>
          </>
        ) : (
          <>
            <strong>Drop a photo or video here, or click to choose</strong>
            <span className="muted">
              JPEG, PNG, WebP, MP4, MOV, MKV, WebM… up to {config.max_upload_mb} MB
            </span>
          </>
        )}
      </div>

      <label className="check">
        <input type="checkbox" checked={guarded} onChange={(e) => setGuarded(e.target.checked)} />
        <span>
          Taken in a bank, court, airport or other guarded place
          <small className="muted">
            {" "}
            There a band only protects children. If the file records where it was taken, SafeFrame
            checks the guarded-places list itself; you can make the rules stricter, never looser.
          </small>
        </span>
      </label>

      <label className="check">
        <input type="checkbox" checked={eventMode} onChange={(e) => setEventMode(e.target.checked)} />
        <span>
          Event mode: also blur anyone who may be under 18
          <small className="muted">
            {" "}
            For schools, clubs and youth events. Ages are estimated from faces, so this is slower
            and can be wrong; uncertain people are blurred.
          </small>
        </span>
      </label>

      <label className="check">
        <input type="checkbox" checked={keep} onChange={(e) => setKeep(e.target.checked)} />
        <span>
          Keep my original upload after export
          <small className="muted">
            {keep
              ? ` It stays on the server so you can re-export, until you delete it or after ${config.job_ttl_hours} h of inactivity.`
              : " By default the original, the preview and all thumbnails are deleted as soon as the blurred file is ready."}
          </small>
        </span>
      </label>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {uploading ? (
        <div className="stack">
          <ProgressBar value={progress} label="Uploading" />
          <button className="btn" onClick={() => abortRef.current?.()}>
            Cancel upload
          </button>
        </div>
      ) : (
        <button className="btn primary big" disabled={!file} onClick={start}>
          Upload and analyse
        </button>
      )}

      <details className="how">
        <summary>How it works</summary>
        <ol>
          <li>People are detected, outlined and tracked, so each person keeps one identity through head turns and brief occlusions.</li>
          <li>The band’s markers are searched for across the frame and up close around every wrist, then matched to the arm wearing them.</li>
          <li>One sighting is enough: the wearer is blurred for their whole time on screen. Nobody’s face is analysed unless the rules below need it.</li>
          <li>In guarded places a band protects only children, so it can’t be used as a disguise. Security cameras never honour it.</li>
          <li>You review the result, override decisions or draw extra regions; then each protected person gets a strong blur and greyscale. Resolution, frame rate and audio are preserved; all metadata, including location, is removed.</li>
        </ol>
      </details>
    </section>
  );
}
