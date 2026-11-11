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
}

export interface AppConfig {
  version: string;
  max_upload_mb: number;
  job_ttl_hours: number;
  default_threshold: number;
  default_min_face_samples: number;
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
}

export interface AnalysisData {
  video: VideoInfo;
  settings: Record<string, unknown>;
  tracks: TrackData[];
  /** Present when silhouettes are served by frame range from GET /masks. */
  mask_store?: { format: "sfm1"; entries: number };
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
  overrides: Record<string, Override>;
  manual_regions: ManualRegion[];
}
