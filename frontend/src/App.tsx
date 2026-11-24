import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import type { AppConfig, Job } from "./api/types";
import { ExportScreen } from "./components/ExportScreen";
import { Notice } from "./components/Notice";
import { ProcessingScreen } from "./components/ProcessingScreen";
import { ReviewScreen } from "./components/ReviewScreen";
import { UploadScreen } from "./components/UploadScreen";
import type { RenderRequest } from "./api/types";

const JOB_KEY = "safeframe.jobId";

function remember(id: string | null) {
  try {
    if (id) sessionStorage.setItem(JOB_KEY, id);
    else sessionStorage.removeItem(JOB_KEY);
  } catch {
    /* storage unavailable: resuming after reload just won't work */
  }
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Last review settings, so "back to review" after export restores them.
  const [lastRender, setLastRender] = useState<RenderRequest | null>(null);

  useEffect(() => {
    api.config().then(setConfig, (e: Error) => setError(`Cannot reach the SafeFrame server: ${e.message}`));
    let saved: string | null = null;
    try {
      saved = sessionStorage.getItem(JOB_KEY);
    } catch {
      /* ignore */
    }
    if (saved) api.job(saved).then(setJob, () => remember(null));
  }, []);

  const startOver = useCallback(() => {
    remember(null);
    setJob(null);
    setLastRender(null);
    setError(null);
  }, []);

  const onJob = useCallback((j: Job) => {
    remember(j.id);
    setJob(j);
  }, []);

  const deleteJob = useCallback(async () => {
    if (job) await api.remove(job.id).catch(() => undefined);
    startOver();
  }, [job, startOver]);

  let screen;
  if (!config) {
    screen = <div className="card center muted">{error ?? "Loading…"}</div>;
  } else if (!job) {
    screen = <UploadScreen config={config} onUploaded={onJob} />;
  } else if (job.state === "queued" || job.state === "analyzing") {
    screen = <ProcessingScreen job={job} onUpdate={setJob} onCancel={deleteJob} />;
  } else if (job.state === "review") {
    screen = (
      <ReviewScreen
        key={job.id}
        job={job}
        config={config}
        initial={lastRender}
        onRender={(updated, req) => {
          setLastRender(req);
          setJob(updated);
        }}
        onDiscard={deleteJob}
      />
    );
  } else if (job.state === "rendering" || job.state === "done") {
    screen = (
      <ExportScreen
        job={job}
        config={config}
        onUpdate={setJob}
        onBackToReview={() => setJob({ ...job, state: "review" })}
        onDelete={deleteJob}
        onStartOver={startOver}
      />
    );
  } else {
    screen = (
      <div className="card center">
        <h2>Processing {job.state === "cancelled" ? "cancelled" : "failed"}</h2>
        <p className="muted">{job.error ?? "The job could not be completed."}</p>
        <button className="btn primary" onClick={deleteJob}>
          Start over
        </button>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width={28} height={28} />
          <span>SafeFrame</span>
        </div>
        <span className="tagline">Respect people who asked not to be shown</span>
      </header>
      {config && <Notice config={config} />}
      <main>{screen}</main>
      <footer className="footer muted">
        Photos and videos are processed on this server only and deleted automatically.
        {config && <> SafeFrame v{config.version}.</>}
      </footer>
    </div>
  );
}
