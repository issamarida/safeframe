import type { MaskRange } from "../review/masks";
import type { AnalysisData, AppConfig, Job, RenderRequest } from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail, res.status);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  config: () => request<AppConfig>("/config"),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  analysis: (id: string) => request<AnalysisData>(`/jobs/${id}/analysis`),
  masks: (id: string, start: number, end: number) =>
    request<MaskRange>(`/jobs/${id}/masks?start=${start}&end=${end}`),
  render: (id: string, body: RenderRequest) =>
    request<Job>(`/jobs/${id}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  remove: (id: string) => request<void>(`/jobs/${id}`, { method: "DELETE" }),
  previewUrl: (id: string) => `${BASE}/jobs/${id}/preview`,
  thumbnailUrl: (id: string, trackId: number) => `${BASE}/jobs/${id}/tracks/${trackId}/thumbnail`,
  downloadUrl: (id: string) => `${BASE}/jobs/${id}/download`,
};

/**
 * Upload with progress. XHR is used because fetch() cannot report upload
 * progress. The body is sent raw (not multipart) and streamed to disk server-side.
 */
export function uploadVideo(
  file: File,
  keepUpload: boolean,
  onProgress: (fraction: number) => void,
): { promise: Promise<Job>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const params = new URLSearchParams({ filename: file.name, keep_upload: String(keepUpload) });
  const promise = new Promise<Job>((resolve, reject) => {
    xhr.open("POST", `${BASE}/jobs?${params}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: { detail?: string } & Partial<Job> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status === 201) resolve(body as Job);
      else reject(new ApiError(body.detail ?? `Upload failed (${xhr.status})`, xhr.status));
    };
    xhr.onerror = () => reject(new ApiError("Network error during upload", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled", 0));
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}

/**
 * Follow a job's progress via server-sent events, falling back to polling if
 * the event stream drops. Returns an unsubscribe function.
 */
export function watchJob(id: string, onUpdate: (job: Job) => void, onError: (e: Error) => void) {
  let closed = false;
  let poll: ReturnType<typeof setInterval> | undefined;
  const source = new EventSource(`${BASE}/jobs/${id}/events`);

  source.onmessage = (event) => onUpdate(JSON.parse(event.data) as Job);
  source.onerror = () => {
    source.close();
    if (closed || poll) return;
    // The server closes the stream when the job leaves an active state; one
    // final fetch tells us whether that was the case or a real failure.
    const tick = async () => {
      try {
        const job = await api.job(id);
        onUpdate(job);
        if (!["queued", "analyzing", "rendering"].includes(job.state)) stop();
      } catch (e) {
        stop();
        onError(e as Error);
      }
    };
    poll = setInterval(tick, 1000);
    void tick();
  };

  function stop() {
    closed = true;
    source.close();
    if (poll) clearInterval(poll);
  }
  return stop;
}
