# 桌面版 GUI ↔ 后端 API 契约

前端通过 `window.pywebview.api.<method>(args)` 调用后端（pywebview js_api）。
所有方法返回 JSON 对象；约定统一外层：`{ ok: boolean, error?: string, ...data }`。
后端会通过 `window.__onEvent(payload)` 向前端推送事件（payload 为 JSON 对象）。

## 方法列表

### 系统 / 环境

- `getSystemInfo()` → `{ ok, python_version, ctranslate2_version, cuda_available, cuda_device_count, gpu_names: string[], onnxruntime_available, onnx_providers: string[], model_dir: string, models: ModelInfo[], generation_config_path: string }`
- `runEnvCheck()` → `{ ok, lines: string[] }` 环境诊断输出（同步、可能耗时数秒）
- `openPath(path: string)` → `{ ok }` 在资源管理器中打开路径
- `openLogsFolder()` → `{ ok }`

### 文件选择

- `selectFiles(title: string, filters?: {name: string, patterns: string[]}[])` → `{ ok, paths: string[] }`
- `selectFolder(title: string)` → `{ ok, path: string|null }`
- `selectSavePath(title: string)` → `{ ok, path: string|null }`

### 模型管理

`ModelInfo = { name: string, path: string, is_root: boolean, has_model: boolean, size_mb: number }`

- `listModels()` → `{ ok, model_dir, models: ModelInfo[] }`（扫描 models 目录）
- `downloadModel(opts)` 启动后台下载（子进程）：
  `{ repo: string, target_dir?: string, force?: boolean, skip_vad?: boolean, skip_whisper_base?: boolean }`
  → `{ ok }`；事件见下；完成后 `listModels` 刷新
- `cancelDownload()` → `{ ok }`
- 预设模型（前端内置下拉）：
  - `chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2`（海南鸡v2 日译中，target `chickenrice-v2`）
  - `TransWithAI/whisper-ja-1.5B-ct2`（日文转录 1.5B，target `whisper-ja-1.5B-ct2`）
  - `openai/whisper-base`（基础版）

### 本地转录 / 翻译任务

- `startTask(opts: TaskOptions)` → `{ ok }` 启动后台子进程（同一时间仅允许一个任务）
- `cancelTask()` → `{ ok }`
- `taskRunning()` → `{ ok, running: boolean }`

```ts
interface TaskOptions {
  files: string[];              // 文件或文件夹的绝对路径
  task: 'translate' | 'transcribe';
  model_path?: string;          // 留空 = 默认 models
  device: 'auto' | 'cpu' | 'cuda';
  compute_type: 'auto' | 'float32' | 'float16' | 'int8_float16' | 'int8';
  sub_formats: string[];        // 'srt' | 'vtt' | 'lrc' | 'txt'
  output_dir?: string;          // 空 = 输出到源文件目录
  overwrite: boolean;
  audio_suffixes?: string;      // 逗号分隔；空 = 使用默认全集
  enable_batching: boolean;
  batch_size?: number | null;   // null = 自动探测
  max_batch_size: number;       // 默认 8
  // —— 以下覆盖 generation_config（null/undefined = 不覆盖）——
  vad_threshold?: number | null;              // 0.3~0.7
  vad_min_speech_duration_ms?: number | null;
  vad_min_silence_duration_ms?: number | null;
  vad_speech_pad_ms?: number | null;
  merge_enabled?: boolean | null;
  merge_max_gap_ms?: number | null;
  merge_max_duration_ms?: number | null;
  smart_split?: boolean | null;
  target_chunk_duration_s?: number | null;    // <=30
}
```

### Modal 云端推理

- `modalCheck()` → `{ ok, installed: boolean, message?: string }`
- `modalRun(opts: ModalOptions)` → `{ ok }`（后台子进程）
- `modalCancel()` → `{ ok }`

```ts
interface ModalOptions {
  input_path: string;                       // 文件或文件夹
  gpu: string;                              // 'T4'|'L4'|'L40S'|'A10G'|'A100-40GB'|'A100-80GB'|'H100'|'H200'|'B200'
  model: 'chickenrice' | 'jim-ja-transcribe' | 'base' | 'custom';
  custom_repo?: string;                     // model==='custom' 时必填
  custom_target_dir?: string;               // model==='custom' 时必填，默认 custom-model
  enable_batching: boolean;
  batch_size?: number | null;
  max_batch_size: number;                   // 默认 8
  timeout_minutes: number;                  // 默认 60
}
```

### generation_config.json5

- `getGenerationConfig()` → `{ ok, raw: string, path: string }`（raw 为文件原文，含注释）
- `saveGenerationConfig(raw: string)` → `{ ok }`（原样保存，保留注释）

## 事件（`window.__onEvent(payload)`）

```ts
type Event =
  | { type: 'task_log'; line: string; stream: 'stdout' | 'stderr' }
  | { type: 'task_progress'; current: number; total: number; file: string }
  | { type: 'task_vad_progress'; current: number; total: number; percent: number }
  | { type: 'task_exit'; code: number }
  | { type: 'download_log'; line: string }
  | { type: 'download_exit'; code: number }
  | { type: 'modal_log'; line: string }
  | { type: 'modal_exit'; code: number };
```

进度解析规则（后端实现，前端只消费事件）：
- 任务行 `正在处理（task，i/N）：path` / `Processing (task) (i/N): path` → `task_progress`
- VAD 行 `正在处理第 i/N 块（p%）…` / `Processing chunk i/N (p%) …` → `task_vad_progress`

## 前端技术栈与结构

- React 18 + TypeScript + Vite + Ant Design 5（`antd` + `@ant-design/icons`）
- 页面（Tabs）：任务、模型管理、云端推理(Modal)、生成参数、环境诊断、关于
- `src/api.ts` 已封装调用与事件订阅（`onEvent(cb)`）
- `src/i18n.tsx` 内置 zh-CN / en-US 文案切换
- dev 模式：`pnpm dev`（端口 5173），后端 `python desktop/app.py --dev` 加载该地址
