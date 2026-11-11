"""Job lifecycle: upload -> analyse -> review -> render -> download -> delete.

Data-retention rules (the privacy contract of the app):

* Each job lives in its own directory under ``data/jobs/<id>``; deleting a job
  removes that directory entirely.
* After a successful render, the original upload, the review preview and the
  per-person thumbnails are deleted unless the user opted to keep the upload.
* Every job, including kept uploads, is deleted after ``job_ttl_hours`` without
  activity, and jobs orphaned by a crash are deleted on the next start-up.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from safeframe.api.schemas import ACTIVE_STATES, JobOut, JobState, RenderRequest, VideoInfoOut
from safeframe.pipeline.analyze import analyze_video
from safeframe.pipeline.interfaces import AgeEstimator, Cancelled, FaceDetector, PersonTracker
from safeframe.pipeline.maskstore import MaskStoreReader, MaskStoreWriter
from safeframe.pipeline.policy import DecisionPolicy
from safeframe.pipeline.render import plan_render, render_video
from safeframe.pipeline.types import Analysis, Box, ManualRegion, VideoInfo
from safeframe.video.io import VideoError, make_preview, probe

log = logging.getLogger(__name__)

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
ComponentsFactory = Callable[[], tuple[PersonTracker, FaceDetector, AgeEstimator]]

# Share of the analysis progress bar spent on the preview encode.
PREVIEW_SHARE = 0.1


class JobError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    id: str
    dir: Path
    filename: str
    keep_upload: bool
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    state: JobState = JobState.QUEUED
    stage: str = "Queued"
    progress: float = 0.0
    error: str | None = None
    upload_deleted: bool = False
    video: VideoInfo | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # --- file layout ---------------------------------------------------------
    @property
    def upload_path(self) -> Path:
        return self.dir / "upload"

    @property
    def preview_path(self) -> Path:
        return self.dir / "preview.mp4"

    @property
    def analysis_path(self) -> Path:
        return self.dir / "analysis.json"

    @property
    def masks_path(self) -> Path:
        """Silhouette store (derived personal data; same lifetime as the analysis)."""
        return self.dir / "masks.sfm"

    @property
    def thumbs_dir(self) -> Path:
        return self.dir / "thumbs"

    @property
    def output_path(self) -> Path:
        return self.dir / "output.mp4"

    @property
    def meta_path(self) -> Path:
        return self.dir / "job.json"

    @property
    def download_name(self) -> str:
        stem = Path(self.filename).stem or "video"
        return f"{stem}-safeframe.mp4"

    def touch(self) -> None:
        self.updated_at = time.time()

    def persist(self) -> None:
        data = {
            "id": self.id,
            "filename": self.filename,
            "keep_upload": self.keep_upload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.value,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "upload_deleted": self.upload_deleted,
            "video": self.video.to_dict() if self.video else None,
        }
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.meta_path)

    @classmethod
    def restore(cls, directory: Path) -> Job:
        data = json.loads((directory / "job.json").read_text())
        video = data.pop("video")
        job = cls(dir=directory, **{k: v for k, v in data.items() if k != "state"})
        job.state = JobState(data["state"])
        job.video = VideoInfo.from_dict(video) if video else None
        return job

    def to_out(self, ttl_seconds: float) -> JobOut:
        v = self.video
        return JobOut(
            id=self.id,
            filename=self.filename,
            state=self.state,
            stage=self.stage,
            progress=round(self.progress, 4),
            error=self.error,
            keep_upload=self.keep_upload,
            upload_deleted=self.upload_deleted,
            output_ready=self.output_path.exists(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.updated_at + ttl_seconds,
            video=VideoInfoOut(**{k: getattr(v, k) for k in VideoInfoOut.model_fields})
            if v
            else None,
        )


def _safe_filename(name: str) -> str:
    name = Path(name or "video").name
    name = re.sub(r"[^\w.\- ]+", "_", name).strip(" .")[:120]
    return name or "video"


class JobManager:
    def __init__(
        self,
        jobs_dir: Path,
        components: ComponentsFactory,
        ttl_seconds: float = 24 * 3600,
        max_upload_bytes: int = 4 * 1024**3,
    ) -> None:
        self.jobs_dir = jobs_dir
        self.components = components
        self.ttl_seconds = ttl_seconds
        self.max_upload_bytes = max_upload_bytes
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        # One worker: jobs share the GPU, and running them in sequence keeps
        # memory use predictable.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="safeframe-job")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._recover()

    # --- lookup --------------------------------------------------------------
    def get(self, job_id: str) -> Job:
        if not JOB_ID.match(job_id):
            raise JobError("job not found", 404)
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError("job not found", 404)
        return job

    def out(self, job: Job) -> JobOut:
        return job.to_out(self.ttl_seconds)

    # --- create --------------------------------------------------------------
    async def create(self, stream, filename: str, keep_upload: bool) -> Job:
        """Stream an uploaded video (async iterable of bytes) to disk, validate it
        with ffprobe and queue analysis. The size limit is enforced while streaming."""
        import anyio

        job_id = secrets.token_hex(16)
        directory = self.jobs_dir / job_id
        directory.mkdir(parents=True)
        job = Job(
            id=job_id, dir=directory, filename=_safe_filename(filename), keep_upload=keep_upload
        )
        try:
            size = 0
            async with await anyio.open_file(job.upload_path, "wb") as fh:
                async for chunk in stream:
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise JobError(
                            f"upload exceeds {self.max_upload_bytes // 1024**2} MB limit", 413
                        )
                    await fh.write(chunk)
            if size == 0:
                raise JobError("empty upload")
            job.video = await anyio.to_thread.run_sync(probe, job.upload_path)
            if job.video.frame_count <= 0:
                raise JobError("the video has no frames")
        except VideoError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise JobError(f"not a readable video: {exc}", 415) from exc
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        job.persist()
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run_analysis, job)
        return job

    # --- analysis ------------------------------------------------------------
    def _set(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.touch()
            if job.dir.exists():
                job.persist()

    def _run_analysis(self, job: Job) -> None:
        if job.cancel.is_set():
            return
        try:
            self._set(job, state=JobState.ANALYZING, stage="Preparing preview", progress=0.0)
            assert job.video is not None
            make_preview(job.upload_path, job.preview_path, job.video)

            self._set(job, stage="Loading models", progress=PREVIEW_SHARE)
            tracker, faces, ages = self.components()

            self._set(job, stage="Detecting, tracking and estimating age")

            def progress(fraction: float) -> None:
                job.progress = PREVIEW_SHARE + (1 - PREVIEW_SHARE) * fraction
                job.touch()

            # The writer removes its partial file if analysis fails or is cancelled.
            with MaskStoreWriter(job.masks_path) as writer:
                result = analyze_video(
                    job.upload_path,
                    tracker,
                    faces,
                    ages,
                    progress=progress,
                    cancel=job.cancel,
                    info=job.video,
                    mask_writer=writer,
                )
                if writer.entries:
                    result.analysis.mask_store = writer.close()
            job.analysis_path.write_text(json.dumps(result.analysis.to_dict()))
            job.thumbs_dir.mkdir(exist_ok=True)
            for track_id, jpeg in result.thumbnails.items():
                (job.thumbs_dir / f"{track_id}.jpg").write_bytes(jpeg)
            self._set(
                job,
                state=JobState.REVIEW,
                stage="Ready for review",
                progress=1.0,
                video=result.analysis.video,
            )
        except Cancelled:
            log.info("job %s cancelled during analysis", job.id)
        except Exception as exc:
            log.exception("analysis failed for job %s", job.id)
            self._fail(job, f"Analysis failed: {exc}")

    def open_masks(self, job: Job, analysis: Analysis) -> MaskStoreReader | None:
        """Validated reader for the job's silhouette store, or None (boxes only)."""
        return MaskStoreReader.for_analysis(analysis, job.dir)

    def load_analysis(self, job: Job) -> Analysis:
        if job.state not in (JobState.REVIEW, JobState.RENDERING, JobState.DONE):
            raise JobError("analysis is not ready", 409)
        if not job.analysis_path.exists():
            raise JobError("analysis data has been deleted", 410)
        return Analysis.from_dict(json.loads(job.analysis_path.read_text()))

    # --- render --------------------------------------------------------------
    def start_render(self, job: Job, request: RenderRequest) -> None:
        with self._lock:
            if job.state not in (JobState.REVIEW, JobState.DONE):
                raise JobError(f"cannot render a job in state '{job.state}'", 409)
            if job.upload_deleted:
                raise JobError("the original upload has already been deleted", 410)
            self._set(
                job,
                state=JobState.RENDERING,
                stage="Queued for rendering",
                progress=0.0,
                error=None,
            )
        self._executor.submit(self._run_render, job, request)

    def _run_render(self, job: Job, request: RenderRequest) -> None:
        if job.cancel.is_set():
            return
        tmp = job.dir / "output.partial.mp4"
        try:
            analysis = Analysis.from_dict(json.loads(job.analysis_path.read_text()))
            policy = DecisionPolicy(request.threshold, request.min_face_samples)
            manual = [
                ManualRegion(r.region_id, r.start_frame, r.end_frame, Box(*r.box))
                for r in request.manual_regions
            ]
            plan = plan_render(analysis, policy, request.overrides, manual)
            self._set(job, stage="Blurring and encoding")

            def progress(fraction: float) -> None:
                job.progress = fraction
                job.touch()

            store = self.open_masks(job, analysis)
            try:
                stats = render_video(
                    job.upload_path,
                    tmp,
                    analysis,
                    plan,
                    progress=progress,
                    cancel=job.cancel,
                    masks=store,
                )
            finally:
                if store is not None:
                    store.close()
            log.info(
                "job %s: rendered %d silhouette and %d box-fallback person-frames",
                job.id,
                stats["mask"],
                stats["box"],
            )
            tmp.replace(job.output_path)
            if not job.keep_upload:
                self._delete_source_material(job)
            self._set(job, state=JobState.DONE, stage="Ready to download", progress=1.0)
        except Cancelled:
            tmp.unlink(missing_ok=True)
            log.info("job %s cancelled during render", job.id)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            log.exception("render failed for job %s", job.id)
            self._fail(job, f"Rendering failed: {exc}", back_to_review=True)

    def _delete_source_material(self, job: Job) -> None:
        """Remove everything that shows unblurred people."""
        job.upload_path.unlink(missing_ok=True)
        job.preview_path.unlink(missing_ok=True)
        job.masks_path.unlink(missing_ok=True)
        shutil.rmtree(job.thumbs_dir, ignore_errors=True)
        job.upload_deleted = True

    def _fail(self, job: Job, message: str, back_to_review: bool = False) -> None:
        if back_to_review and job.analysis_path.exists() and job.upload_path.exists():
            self._set(job, state=JobState.REVIEW, stage="Ready for review", error=message)
        else:
            self._set(job, state=JobState.FAILED, stage="Failed", error=message)

    # --- deletion ------------------------------------------------------------
    def delete(self, job: Job) -> None:
        job.cancel.set()
        with self._lock:
            self._jobs.pop(job.id, None)
            job.state = JobState.CANCELLED
        # Workers notice the cancel flag within one frame; retry removal briefly
        # in case FFmpeg still holds files open.
        for _ in range(20):
            shutil.rmtree(job.dir, ignore_errors=True)
            if not job.dir.exists():
                return
            time.sleep(0.1)
        log.warning("could not fully remove %s; the sweeper will retry", job.dir)

    def sweep(self, now: float | None = None) -> int:
        """Delete expired jobs and stray directories. Returns jobs removed."""
        now = now or time.time()
        with self._lock:
            expired = [
                j
                for j in self._jobs.values()
                if j.state not in ACTIVE_STATES and now - j.updated_at > self.ttl_seconds
            ]
            known = set(self._jobs)
        for job in expired:
            log.info("job %s expired; deleting", job.id)
            self.delete(job)
        for directory in self.jobs_dir.iterdir():
            if directory.name not in known and directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
        return len(expired)

    def _recover(self) -> None:
        """Reload finished jobs after a restart; delete anything interrupted or broken."""
        for directory in self.jobs_dir.iterdir():
            try:
                job = Job.restore(directory)
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                continue
            if job.state in ACTIVE_STATES or job.state in (JobState.FAILED, JobState.CANCELLED):
                shutil.rmtree(directory, ignore_errors=True)
                continue
            self._jobs[job.id] = job

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.state in ACTIVE_STATES:
                job.cancel.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
