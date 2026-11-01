// ===== 后端 API 类型定义（见 desktop/API_CONTRACT.md）=====

export interface ModelInfo {
  name: string;
  path: string;
  is_root: boolean;
  has_model: boolean;
  size_mb: number;
}

export interface SystemInfo {
  ok: boolean;
  error?: string;
  python_version: string;
  ctranslate2_version: string;
  cuda_available: boolean;
  cuda_device_count: number;
  gpu_names: string[];
  onnxruntime_available: boolean;
  onnx_providers: string[];
  model_dir: string;
  models: ModelInfo[];
  generation_config_path: string;
}

export interface ApiResult {
  ok: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface TaskOptions {
  files: string[];
  task: "translate" | "transcribe";
  model_path?: string;
  device: "auto" | "cpu" | "cuda";
  compute_type: "auto" | "float32" | "float16" | "int8_float16" | "int8";
  sub_formats: string[];
  output_dir?: string;
  overwrite: boolean;
  audio_suffixes?: string;
  enable_batching: boolean;
  batch_size?: number | null;
  max_batch_size: number;
  vad_threshold?: number | null;
  vad_min_speech_duration_ms?: number | null;
  vad_min_silence_duration_ms?: number | null;
  vad_speech_pad_ms?: number | null;
  merge_enabled?: boolean | null;
  merge_max_gap_ms?: number | null;
  merge_max_duration_ms?: number | null;
  smart_split?: boolean | null;
  target_chunk_duration_s?: number | null;
}

export interface ModalOptions {
  input_path: string;
  gpu: string;
  model: "chickenrice" | "jim-ja-transcribe" | "base" | "custom";
  custom_repo?: string;
  custom_target_dir?: string;
  enable_batching: boolean;
  batch_size?: number | null;
  max_batch_size: number;
  timeout_minutes: number;
}

export interface DownloadOptions {
  repo: string;
  target_dir?: string;
  force?: boolean;
  skip_vad?: boolean;
  skip_whisper_base?: boolean;
  /** 下载源端点；留空 = 默认逻辑（官方优先，失败自动切 hf-mirror） */
  endpoint?: string;
}

export interface DownloadSource {
  id: string;
  label: string;
  endpoint: string;
  note: string;
}

export type AppEvent =
  | { type: "task_log"; line: string; stream: "stdout" | "stderr" }
  | { type: "task_progress"; current: number; total: number; file: string }
  | { type: "task_vad_progress"; current: number; total: number; percent: number }
  | { type: "task_exit"; code: number }
  | { type: "download_log"; line: string }
  | { type: "download_exit"; code: number }
  | { type: "modal_log"; line: string }
  | { type: "modal_exit"; code: number };

export interface PyWebViewApi {
  getSystemInfo(): Promise<SystemInfo>;
  runEnvCheck(): Promise<{ ok: boolean; lines: string[]; error?: string }>;
  openPath(path: string): Promise<ApiResult>;
  openLogsFolder(): Promise<ApiResult>;
  selectFiles(title: string): Promise<{ ok: boolean; paths: string[] }>;
  selectFolder(title: string): Promise<{ ok: boolean; path: string | null }>;
  listModels(): Promise<{ ok: boolean; model_dir: string; models: ModelInfo[]; error?: string }>;
  downloadModel(opts: DownloadOptions): Promise<ApiResult>;
  getDownloadSources(): Promise<{ ok: boolean; sources: DownloadSource[] }>;
  cancelDownload(): Promise<ApiResult>;
  startTask(opts: TaskOptions): Promise<ApiResult>;
  cancelTask(): Promise<ApiResult>;
  taskRunning(): Promise<{ ok: boolean; running: boolean }>;
  modalCheck(): Promise<{ ok: boolean; installed: boolean; message?: string }>;
  modalRun(opts: ModalOptions): Promise<ApiResult>;
  modalCancel(): Promise<ApiResult>;
  getGenerationConfig(): Promise<{ ok: boolean; raw: string; path: string; error?: string }>;
  saveGenerationConfig(raw: string): Promise<ApiResult>;
}

export function getApi(): PyWebViewApi {
  const w = window as unknown as { pywebview?: { api: PyWebViewApi } };
  if (!w.pywebview || !w.pywebview.api) {
    throw new Error("pywebview API 尚未就绪");
  }
  return w.pywebview.api;
}

export function onApiReady(cb: () => void): () => void {
  const w = window as unknown as {
    pywebview?: { api: PyWebViewApi };
    addEventListener: (t: string, f: () => void) => void;
    removeEventListener: (t: string, f: () => void) => void;
  };
  if (w.pywebview && w.pywebview.api) {
    cb();
    return () => {};
  }
  const handler = () => cb();
  w.addEventListener("pywebviewready", handler);
  return () => w.removeEventListener("pywebviewready", handler);
}

// ===== 事件订阅 =====

type EventCb = (e: AppEvent) => void;
const listeners = new Set<EventCb>();

export function onEvent(cb: EventCb): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

declare global {
  interface Window {
    __onEvent?: (payload: AppEvent) => void;
  }
}

window.__onEvent = (payload: AppEvent) => {
  listeners.forEach((cb) => cb(payload));
};

export async function call<T extends ApiResult>(
  fn: () => Promise<T>,
  errorPrefix: string
): Promise<T | ApiResult> {
  try {
    const r = await fn();
    return r;
  } catch (e) {
    return { ok: false, error: `${errorPrefix}: ${String(e)}` };
  }
}
