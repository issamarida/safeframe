"""FastAPI application."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from safeframe import __version__
from safeframe.api.jobs import ComponentsFactory, Job, JobError, JobManager
from safeframe.api.schemas import (
    ACTIVE_STATES,
    AppConfigOut,
    JobOut,
    JobState,
    RenderRequest,
)
from safeframe.config import Settings, get_settings
from safeframe.pipeline.policy import DEFAULT_MIN_FACE_SAMPLES, DEFAULT_THRESHOLD

log = logging.getLogger(__name__)

# Media responses must never be cached by browsers or proxies.
NO_STORE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def default_components(settings: Settings) -> ComponentsFactory:
    from safeframe.pipeline.factory import ModelBundle, ModelConfig

    bundle = ModelBundle(
        ModelConfig(
            yolo_weights=settings.yolo_weights,
            age_model=settings.age_model,
            age_model_revision=settings.age_model_revision,
            device=settings.device,
        )
    )
    if settings.warm_up_models:
        threading.Thread(target=bundle.warm_up, name="model-warm-up", daemon=True).start()
    return lambda: (bundle.new_tracker(), bundle.new_face_detector(), bundle.age_estimator())


def create_app(
    settings: Settings | None = None,
    components: ComponentsFactory | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = JobManager(
            settings.jobs_dir,
            components or default_components(settings),
            ttl_seconds=settings.job_ttl_hours * 3600,
            max_upload_bytes=settings.max_upload_mb * 1024**2,
        )
        app.state.jobs = manager

        async def sweeper() -> None:
            while True:
                await asyncio.sleep(settings.sweep_interval_seconds)
                await asyncio.to_thread(manager.sweep)

        task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.to_thread(manager.shutdown)

    app = FastAPI(title="SafeFrame", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(JobError)
    async def _job_error(_: Request, exc: JobError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    app.include_router(build_router(settings), prefix="/api")

    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


def manager(request: Request) -> JobManager:
    return request.app.state.jobs


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    def job_dep(job_id: str, jobs: JobManager = Depends(manager)) -> Job:
        return jobs.get(job_id)

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @router.get("/config", response_model=AppConfigOut)
    def config() -> AppConfigOut:
        return AppConfigOut(
            version=__version__,
            max_upload_mb=settings.max_upload_mb,
            job_ttl_hours=settings.job_ttl_hours,
            default_threshold=DEFAULT_THRESHOLD,
            default_min_face_samples=DEFAULT_MIN_FACE_SAMPLES,
        )

    @router.post("/jobs", response_model=JobOut, status_code=201)
    async def create_job(
        request: Request,
        filename: str = Query("video.mp4", max_length=255),
        keep_upload: bool = Query(False),
        jobs: JobManager = Depends(manager),
    ) -> JobOut:
        """Upload a video as the raw request body (``application/octet-stream``).

        Streaming the body straight to the job directory avoids multipart
        temp files in shared system directories and enforces the size limit
        without buffering.
        """
        job = await jobs.create(request.stream(), filename, keep_upload)
        return jobs.out(job)

    @router.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job: Job = Depends(job_dep), jobs: JobManager = Depends(manager)) -> JobOut:
        return jobs.out(job)

    @router.get("/jobs/{job_id}/events")
    async def job_events(
        request: Request, job: Job = Depends(job_dep), jobs: JobManager = Depends(manager)
    ) -> StreamingResponse:
        """Server-sent events with the job snapshot whenever it changes."""

        async def stream() -> AsyncIterator[str]:
            last = None
            while not await request.is_disconnected():
                snapshot = jobs.out(job).model_dump_json()
                if snapshot != last:
                    yield f"data: {snapshot}\n\n"
                    last = snapshot
                if job.state not in ACTIVE_STATES:
                    return
                await asyncio.sleep(0.4)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.get("/jobs/{job_id}/analysis")
    def get_analysis(
        request: Request, job: Job = Depends(job_dep), jobs: JobManager = Depends(manager)
    ) -> Response:
        # Re-serialised from the validated model, so the UI only ever receives
        # masks the renderer would also accept (same frames, same budgets).
        analysis = jobs.load_analysis(job)
        body = analysis.to_dict()
        for track in body["tracks"]:
            track.pop("samples", None)  # per-sample detail isn't needed by the UI
        # Tell the UI whether silhouettes can be fetched by frame range, and how
        # many survived validation. File names and ids stay server-side.
        body.pop("mask_store", None)
        store = jobs.open_masks(job, analysis)
        if store is not None:
            with store:
                if len(store):
                    body["mask_store"] = {"format": "sfm1", "entries": len(store)}
        return _json(request, body)

    @router.get("/jobs/{job_id}/masks")
    def get_masks(
        request: Request,
        start: int = Query(ge=0),
        end: int = Query(gt=0),
        job: Job = Depends(job_dep),
        jobs: JobManager = Depends(manager),
    ) -> Response:
        """Validated silhouettes for frames ``[start, end)``, grouped by track.

        The range is bounded; ``end`` in the response is the exclusive frame
        actually covered, so clients continue from there.
        """
        if end <= start:
            raise JobError("end must be greater than start")
        analysis = jobs.load_analysis(job)
        store = jobs.open_masks(job, analysis)
        if store is None:
            return _json(request, {"start": start, "end": end, "tracks": {}})
        with store:
            covered, rows = store.rows(start, end)
        return _json(
            request,
            {"start": start, "end": covered, "tracks": {str(t): r for t, r in rows.items()}},
        )

    @router.get("/jobs/{job_id}/preview")
    def get_preview(job: Job = Depends(job_dep)) -> FileResponse:
        return _file(job.preview_path, "video/mp4", "the preview has been deleted")

    @router.get("/jobs/{job_id}/tracks/{track_id}/thumbnail")
    def get_thumbnail(track_id: int, job: Job = Depends(job_dep)) -> FileResponse:
        return _file(job.thumbs_dir / f"{track_id}.jpg", "image/jpeg", "thumbnail not available")

    @router.post("/jobs/{job_id}/render", response_model=JobOut, status_code=202)
    def render(
        body: RenderRequest, job: Job = Depends(job_dep), jobs: JobManager = Depends(manager)
    ) -> JobOut:
        jobs.start_render(job, body)
        return jobs.out(job)

    @router.get("/jobs/{job_id}/download")
    def download(job: Job = Depends(job_dep)) -> FileResponse:
        if job.state is not JobState.DONE:
            raise HTTPException(409, "the video has not been rendered yet")
        job.touch()
        return _file(job.output_path, "video/mp4", "output not found", download=job.download_name)

    @router.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job: Job = Depends(job_dep), jobs: JobManager = Depends(manager)) -> Response:
        jobs.delete(job)
        return Response(status_code=204)

    return router


def _file(path: Path, media_type: str, missing: str, download: str | None = None) -> FileResponse:
    if not path.exists():
        raise HTTPException(410, missing)
    return FileResponse(
        path,
        media_type=media_type,
        filename=download,
        headers=NO_STORE,
        content_disposition_type="attachment" if download else "inline",
    )


def _json(request: Request, body: dict) -> Response:
    """Compact JSON, gzip-compressed when large and the client accepts it."""
    payload = json.dumps(body, separators=(",", ":")).encode()
    headers = dict(NO_STORE)
    if len(payload) > 8192 and "gzip" in request.headers.get("accept-encoding", ""):
        # Silhouette run-lengths compress very well.
        payload = gzip.compress(payload, compresslevel=5)
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
    return Response(payload, media_type="application/json", headers=headers)
