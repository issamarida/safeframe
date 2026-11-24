export type JobState =
  | "queued"
  | "analyzing"
  | "review"
  | "rendering"
  | "done"
  | "failed"
  | "cancelled";

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  fps_rational: string;
  frame_count: number;
  duration: number;
  has_audio: boolean;
  /** A photo is analysed as a one-frame video. Absent on legacy analyses. */
  media?: Media;
}

export type Media = "video" | "photo";

/**
 * Where the capture was taken, resolved once at upload (backend
 * pipeline/context.py). The location itself is never stored or returned.
 */
export interface CaptureContext {
  guarded: boolean;
  /** "registry": the location matched a guarded place; "user": declared at upload. */
  guarded_source: "none" | "registry" | "user";
  zone_name: string | null;
  zone_kind: string | null;
  /** Event mode: also protect anyone who may be under 18, band or not. */
  protect_minors: boolean;
  had_location: boolean;
}

export interface Job {
  id: string;
  filename: string;
  state: JobState;
  stage: string;
  progress: number;
  error: string | null;
  keep_upload: boolean;
  upload_deleted: boolean;
  output_ready: boolean;
  created_at: number;
  updated_at: number;
  expires_at: number;
  video: VideoInfo | null;
  media: Media;
  context: CaptureContext;
  /** The unblurred original was sealed for the trustees (see backend seal.py). */
  sealed_ready: boolean;
}

export interface AppConfig {
  version: string;
  max_upload_mb: number;
  job_ttl_hours: number;
  default_threshold: number;
  default_min_face_samples: number;
  default_min_band_hits: number;
  sealing_enabled: boolean;
  seal_retention_days: number;
  band_spec_version: number;
  notice: string;
  intended_use: string;
}

/** [frame, x1, y1, x2, y2, interpolated(0|1)] in source-video pixels. */
export type BoxRow = [number, number, number, number, number, number];

/**
 * Silhouette coverage on one observed frame (see backend pipeline/masks.py):
 * [frame, x, y, width, height, runs]. Integer source pixels, half-open crop.
 * `runs` alternate background/foreground row-major, starting with background
 * (first run may be 0), and sum to width * height.
 */
export type MaskRow = [number, number, number, number, number, number[]];

export type BandVia = "wrist" | "body" | "box";

/** [frame, via, ambiguous(0|1), marker side in px]. */
export type BandHitRow = [number, BandVia, number, number];

export interface BandData {
  /** Sightings attributed to this person alone. */
  confirmed: number;
  /** Sightings that fitted this person and someone else equally well. */
  possible: number;
  first_frame: number | null;
  via: Record<BandVia, number>;
  /** The first sightings (capped server-side); counts above cover all of them. */
  hits: BandHitRow[];
}

export interface TrackData {
  track_id: number;
  first_frame: number;
  last_frame: number;
  p_minor: number | null;
  expected_age: number | null;
  face_samples: number;
  peak_confidence: number;
  boxes: BoxRow[];
  /** Optional; absent on legacy analyses and box-only models. */
  masks_v1?: MaskRow[];
  /** Absent on legacy (age-only) analyses: treated as no band. */
  band?: BandData;
}

export interface AnalysisData {
  video: VideoInfo;
  settings: Record<string, unknown>;
  tracks: TrackData[];
  context?: Partial<CaptureContext>;
  /** Frames where a band was seen but couldn't be attributed to anyone. */
  orphan_band_frames?: number[];
}

export type Override = "blur" | "keep";

/** [x1, y1, x2, y2] in source-video pixels. */
export type Rect = [number, number, number, number];

export interface ManualRegion {
  region_id: string;
  start_frame: number;
  end_frame: number;
  box: Rect;
}

export interface RenderRequest {
  threshold: number;
  min_face_samples: number;
  min_band_hits: number;
  overrides: Record<string, Override>;
  manual_regions: ManualRegion[];
}
