#!/usr/bin/env python3
"""
Faster Whisper TransWithAI 桌面版 — pywebview 入口与 JS API

开发模式:  python desktop/app.py --dev   （前端 pnpm dev，加载 http://localhost:5173）
生产模式:  PyInstaller 打包后运行，加载 frontend/dist 静态资源
"""

import argparse
import functools
import http.server
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import webview

# ---------- 路径定位 ----------
IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    APP_ROOT = Path(sys.executable).resolve().parent
    # PyInstaller 6 onedir 模式：数据文件位于 exe 同级的 _internal 目录
    BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", APP_ROOT))
else:
    APP_ROOT = Path(__file__).resolve().parent.parent
    BUNDLE_ROOT = APP_ROOT

FRONTEND_DIST_CANDIDATES = [
    APP_ROOT / "frontend" / "dist",       # 用户可手动放置/覆盖
    BUNDLE_ROOT / "frontend" / "dist",    # 打包默认位置（_internal 下）
]
FRONTEND_DIST = next((p for p in FRONTEND_DIST_CANDIDATES if p.exists()), FRONTEND_DIST_CANDIDATES[0])
GENERATION_CONFIG = APP_ROOT / "generation_config.json5"
MODELS_DIR = APP_ROOT / "models"
LOGS_DIR = APP_ROOT / "logs"


def _install_excepthook():
    """windowed 打包模式下把未捕获异常写入文件，便于排查。"""

    def hook(exc_type, exc_value, exc_tb):
        import traceback

        try:
            log = APP_ROOT / "app_error.log"
            with open(log, "a", encoding="utf-8") as f:
                f.write("\n==== %s ====\n" % __import__("datetime").datetime.now())
                traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    if IS_FROZEN:
        sys.excepthook = hook


_install_excepthook()

# 打包模式下强制 pythonnet 使用 Windows 自带的 .NET Framework 运行时，
# 避免 clr_loader 去探测不存在的 dotnet root（CoreCLR）而启动失败。
if IS_FROZEN:
    os.environ.setdefault("PYTHONNET_RUNTIME", "netfx")

if not IS_FROZEN:
    sys.path.insert(0, str(APP_ROOT / "src"))

logger = logging.getLogger("desktop")


def _is_main_model_dir(d: Path) -> bool:
    has_weights = (d / "model.bin").exists() or (d / "model.safetensors").exists()
    return has_weights and (d / "config.json").exists()


class Api:
    """pywebview js_api 实现。所有方法返回 JSON 可序列化对象。"""

    def __init__(self):
        self._task_proc: subprocess.Popen | None = None
        self._task_lock = threading.Lock()
        self._download_proc: subprocess.Popen | None = None
        self._download_lock = threading.Lock()
        self._modal_proc: subprocess.Popen | None = None
        self._modal_lock = threading.Lock()

    # ---------- 内部工具 ----------

    def _window(self) -> webview.Window | None:
        return webview.windows[0] if webview.windows else None

    def _emit(self, payload: dict):
        win = self._window()
        if win is None:
            return
        try:
            win.evaluate_js(
                f"window.__onEvent && window.__onEvent({json.dumps(payload, ensure_ascii=False)})"
            )
        except Exception:
            pass

    def _pump_output(self, proc: subprocess.Popen, prefix: str, stream_name: str):
        """读子进程输出行，推送事件并解析进度。"""
        stream = proc.stdout if stream_name == "stdout" else proc.stderr
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                if prefix == "task":
                    self._emit({"type": "task_log", "line": line, "stream": stream_name})
                    self._parse_task_progress(line)
                elif prefix == "download":
                    self._emit({"type": "download_log", "line": line})
                elif prefix == "modal":
                    self._emit({"type": "modal_log", "line": line})
        except Exception as e:
            logger.debug("pump_output ended: %s", e)

    def _watch_proc(self, proc: subprocess.Popen, prefix: str):
        for name in ("stdout", "stderr"):
            t = threading.Thread(
                target=self._pump_output,
                args=(proc, prefix, name),
                daemon=True,
            )
            t.start()
        code = proc.wait()
        if prefix == "task":
            self._emit({"type": "task_exit", "code": code})
        elif prefix == "download":
            self._emit({"type": "download_exit", "code": code})
        elif prefix == "modal":
            self._emit({"type": "modal_exit", "code": code})

    _RE_TASK_ZH = re.compile(r"，\s*(\d+)\s*/\s*(\d+)[）)]")
    _RE_TASK_EN = re.compile(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)")
    _RE_VAD_ZH = re.compile(r"第\s*(\d+)\s*/\s*(\d+)\s*块[（(]\s*([\d.]+)\s*%")
    _RE_VAD_EN = re.compile(r"chunk\s+(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)\s*%")

    def _parse_task_progress(self, line: str):
        if ("正在处理" in line or "Processing" in line) and ("文件" not in line.split("（")[0].split("(")[0]):
            m = self._RE_TASK_ZH.search(line) or self._RE_TASK_EN.search(line)
            if m:
                self._emit(
                    {
                        "type": "task_progress",
                        "current": int(m.group(1)),
                        "total": int(m.group(2)),
                        "file": line.split("：", 1)[-1].split(": ", 1)[-1].strip(),
                    }
                )
                return
        m = self._RE_VAD_ZH.search(line) or self._RE_VAD_EN.search(line)
        if m:
            self._emit(
                {
                    "type": "task_vad_progress",
                    "current": int(m.group(1)),
                    "total": int(m.group(2)),
                    "percent": float(m.group(3)),
                }
            )

    def _infer_command(self) -> list[str] | None:
        """定位推理入口：打包后为同目录 infer.exe；开发模式为 python infer.py。"""
        if IS_FROZEN:
            exe = APP_ROOT / "infer.exe"
            return [str(exe)] if exe.exists() else None
        return [sys.executable, str(APP_ROOT / "infer.py")]

    def _download_command(self) -> list[str] | None:
        if IS_FROZEN:
            exe = APP_ROOT / "download_models.exe"
            return [str(exe)] if exe.exists() else None
        return [sys.executable, str(APP_ROOT / "download_models.py")]

    def _modal_command(self) -> list[str] | None:
        if IS_FROZEN:
            return None  # 打包版不内置 modal 依赖
        return [sys.executable, str(APP_ROOT / "desktop" / "modal_cli.py")]

    # ---------- 系统 / 环境 ----------

    def get_system_info(self):
        info = {
            "ok": True,
            "python_version": sys.version.split()[0],
            "ctranslate2_version": "",
            "cuda_available": False,
            "cuda_device_count": 0,
            "gpu_names": [],
            "onnxruntime_available": False,
            "onnx_providers": [],
            "model_dir": str(MODELS_DIR),
            "models": self._scan_models(),
            "generation_config_path": str(GENERATION_CONFIG),
        }
        try:
            import ctranslate2

            info["ctranslate2_version"] = ctranslate2.__version__
            info["cuda_device_count"] = ctranslate2.get_cuda_device_count()
            info["cuda_available"] = info["cuda_device_count"] > 0
            if info["cuda_available"]:
                try:
                    names = []
                    for i in range(info["cuda_device_count"]):
                        try:
                            names.append(ctranslate2.get_cuda_device_name(i))
                        except Exception:
                            pass
                    info["gpu_names"] = names
                except Exception:
                    pass
        except Exception:
            pass
        try:
            import onnxruntime as ort

            info["onnxruntime_available"] = True
            info["onnx_providers"] = list(ort.get_available_providers())
        except Exception:
            pass
        return info

    def _scan_models(self):
        models = []
        if not MODELS_DIR.exists():
            return models
        # models 根目录
        if _is_main_model_dir(MODELS_DIR):
            size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
            models.append(
                {
                    "name": "models（根目录）",
                    "path": str(MODELS_DIR),
                    "is_root": True,
                    "has_model": True,
                    "size_mb": size / 1024 / 1024,
                }
            )
        for d in sorted(MODELS_DIR.iterdir()):
            if not d.is_dir():
                continue
            if _is_main_model_dir(d):
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                models.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "is_root": False,
                        "has_model": True,
                        "size_mb": size / 1024 / 1024,
                    }
                )
        return models

    def list_models(self):
        return {"ok": True, "model_dir": str(MODELS_DIR), "models": self._scan_models()}

    def run_env_check(self):
        """进程内运行 infer.py 的诊断函数，捕获打印输出。"""
        buf = io.StringIO()
        lines = []
        try:
            from faster_whisper_transwithai_chickenrice import infer as infer_mod

            saved = sys.stdout
            sys.stdout = buf
            try:
                infer_mod.diagnose_environment()
                infer_mod.check_onnxruntime_detailed()
                infer_mod.test_vad_initialization()
            finally:
                sys.stdout = saved
            lines = buf.getvalue().splitlines()
            return {"ok": True, "lines": lines}
        except Exception as e:
            lines = buf.getvalue().splitlines()
            lines.append(f"[诊断失败] {e}")
            return {"ok": False, "lines": lines, "error": str(e)}

    def open_path(self, path: str):
        try:
            p = Path(path)
            if p.is_dir():
                os.startfile(str(p))  # noqa: S606
            elif p.exists():
                os.startfile(str(p.parent))  # noqa: S606
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def open_logs_folder(self):
        LOGS_DIR.mkdir(exist_ok=True)
        try:
            os.startfile(str(LOGS_DIR))  # noqa: S606
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------- 文件选择 ----------

    def select_files(self, title: str):
        win = self._window()
        if win is None:
            return {"ok": False, "paths": [], "error": "窗口未就绪"}
        result = win.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            title=title or "选择文件",
            file_types=(
                "音视频文件 (*.mp3;*.wav;*.flac;*.m4a;*.aac;*.ogg;*.wma;*.mp4;*.mkv;*.avi;*.mov;*.webm;*.flv;*.wmv)",
                "所有文件 (*.*)",
            ),
        )
        paths = list(result) if result else []
        return {"ok": True, "paths": paths}

    def select_folder(self, title: str):
        win = self._window()
        if win is None:
            return {"ok": False, "path": None, "error": "窗口未就绪"}
        result = win.create_file_dialog(webview.FOLDER_DIALOG, title=title or "选择文件夹")
        if result:
            return {"ok": True, "path": result[0]}
        return {"ok": True, "path": None}

    # ---------- 本地任务 ----------

    def start_task(self, opts: dict):
        cmd = self._infer_command()
        if cmd is None:
            return {"ok": False, "error": "未找到推理入口（infer.exe / infer.py）"}
        with self._task_lock:
            if self._task_proc and self._task_proc.poll() is None:
                return {"ok": False, "error": "已有任务在运行，请先停止当前任务"}
            try:
                args = self._build_task_args(opts)
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "bufsize": 1,
                    "cwd": str(APP_ROOT),
                }
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._task_proc = subprocess.Popen(
                    cmd + args,
                    **kwargs,
                )
            except Exception as e:
                return {"ok": False, "error": f"启动失败: {e}"}
            threading.Thread(
                target=self._watch_proc,
                args=(self._task_proc, "task"),
                daemon=True,
            ).start()
        return {"ok": True}

    @staticmethod
    def _build_task_args(opts: dict) -> list[str]:
        def s(v):
            return str(v)

        args: list[str] = []
        if opts.get("task"):
            args.append(f"--task={opts['task']}")
        if opts.get("device"):
            args.append(f"--device={opts['device']}")
        if opts.get("compute_type"):
            args.append(f"--compute_type={opts['compute_type']}")
        sub_formats = opts.get("sub_formats") or ["srt", "vtt", "lrc"]
        args.append("--sub_formats=" + ",".join(sub_formats))
        if opts.get("output_dir"):
            args.append(f"--output_dir={opts['output_dir']}")
        if opts.get("overwrite"):
            args.append("--overwrite")
        if opts.get("audio_suffixes"):
            args.append(f"--audio_suffixes={opts['audio_suffixes']}")
        if opts.get("model_path"):
            args.append(f"--model_name_or_path={opts['model_path']}")
        if opts.get("enable_batching"):
            args.append("--enable_batching")
            bs = opts.get("batch_size")
            if bs:
                args.append(f"--batch_size={s(bs)}")
            args.append(f"--max_batch_size={s(opts.get('max_batch_size') or 8)}")
        # generation 参数覆盖
        if opts.get("vad_threshold") is not None:
            args.append(f"--vad_threshold={opts['vad_threshold']}")
        if opts.get("vad_min_speech_duration_ms") is not None:
            args.append(f"--vad_min_speech_duration_ms={s(opts['vad_min_speech_duration_ms'])}")
        if opts.get("vad_min_silence_duration_ms") is not None:
            args.append(f"--vad_min_silence_duration_ms={s(opts['vad_min_silence_duration_ms'])}")
        if opts.get("vad_speech_pad_ms") is not None:
            args.append(f"--vad_speech_pad_ms={s(opts['vad_speech_pad_ms'])}")
        me = opts.get("merge_enabled")
        if me is True:
            args.append("--merge_segments")
        elif me is False:
            args.append("--no_merge_segments")
        if opts.get("merge_max_gap_ms") is not None:
            args.append(f"--merge_max_gap_ms={s(opts['merge_max_gap_ms'])}")
        if opts.get("merge_max_duration_ms") is not None:
            args.append(f"--merge_max_duration_ms={s(opts['merge_max_duration_ms'])}")
        ss = opts.get("smart_split")
        if ss is not None:
            args.append(f"--smart_split_with_vad={'true' if ss else 'false'}")
        if opts.get("target_chunk_duration_s") is not None:
            args.append(f"--target_chunk_duration_s={s(opts['target_chunk_duration_s'])}")
        # 位置参数（文件列表）放最后
        args.extend(opts.get("files") or [])
        return args

    def cancel_task(self):
        with self._task_lock:
            proc = self._task_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True}

    def task_running(self):
        with self._task_lock:
            proc = self._task_proc
        return {"ok": True, "running": bool(proc and proc.poll() is None)}

    # ---------- 模型下载 ----------

    def download_model(self, opts: dict):
        cmd = self._download_command()
        if cmd is None:
            return {"ok": False, "error": "未找到 download_models 入口"}
        with self._download_lock:
            if self._download_proc and self._download_proc.poll() is None:
                return {"ok": False, "error": "已有下载任务在运行"}
            args = cmd + ["--hf-model", opts.get("repo", "")]
            if opts.get("target_dir"):
                args += ["--target-dir", opts["target_dir"]]
            if opts.get("force"):
                args.append("--force")
            try:
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "bufsize": 1,
                    "cwd": str(APP_ROOT),
                }
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._download_proc = subprocess.Popen(args, **kwargs)
            except Exception as e:
                return {"ok": False, "error": f"启动下载失败: {e}"}
            threading.Thread(
                target=self._watch_proc,
                args=(self._download_proc, "download"),
                daemon=True,
            ).start()
        return {"ok": True}

    def cancel_download(self):
        with self._download_lock:
            proc = self._download_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True}

    # ---------- Modal 云端 ----------

    def modal_check(self):
        try:
            import modal  # noqa: F401

            return {"ok": True, "installed": True}
        except ImportError:
            return {
                "ok": True,
                "installed": False,
                "message": "pip install modal questionary 后重启应用，并运行 modal token new 完成授权。",
            }

    def modal_run(self, opts: dict):
        cmd = self._modal_command()
        if cmd is None:
            return {
                "ok": False,
                "error": "打包版未内置 Modal 支持；请在 Python 环境中运行 python modal_infer.py，或重新以开发模式启动。",
            }
        with self._modal_lock:
            if self._modal_proc and self._modal_proc.poll() is None:
                return {"ok": False, "error": "已有云端任务在运行"}
            args = cmd + [
                "--input",
                opts.get("input_path", ""),
                "--gpu",
                opts.get("gpu", "T4"),
                "--model",
                opts.get("model", "chickenrice"),
                "--timeout-minutes",
                str(opts.get("timeout_minutes") or 60),
                "--max-batch-size",
                str(opts.get("max_batch_size") or 8),
            ]
            if opts.get("custom_repo"):
                args += ["--custom-repo", opts["custom_repo"]]
            if opts.get("custom_target_dir"):
                args += ["--custom-target-dir", opts["custom_target_dir"]]
            if opts.get("enable_batching"):
                args.append("--enable-batching")
                if opts.get("batch_size"):
                    args += ["--batch-size", str(opts["batch_size"])]
            try:
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "bufsize": 1,
                    "cwd": str(APP_ROOT),
                }
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                self._modal_proc = subprocess.Popen(args, **kwargs)
            except Exception as e:
                return {"ok": False, "error": f"启动失败: {e}"}
            threading.Thread(
                target=self._watch_proc,
                args=(self._modal_proc, "modal"),
                daemon=True,
            ).start()
        return {"ok": True}

    def modal_cancel(self):
        with self._modal_lock:
            proc = self._modal_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True}

    # ---------- generation_config ----------

    def get_generation_config(self):
        try:
            raw = GENERATION_CONFIG.read_text(encoding="utf-8")
            return {"ok": True, "raw": raw, "path": str(GENERATION_CONFIG)}
        except Exception as e:
            return {"ok": False, "raw": "", "path": str(GENERATION_CONFIG), "error": str(e)}

    def save_generation_config(self, raw: str):
        try:
            GENERATION_CONFIG.write_text(raw, encoding="utf-8")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------- 暴露给前端的 camelCase 接口 ----------
    # pywebview 会把 js_api 对象上的 Python 方法名**原样**暴露给 JS，
    # 而 desktop/API_CONTRACT.md 与 frontend/src/api.ts 约定的是 camelCase 名称。
    # 若不转发，前端调用 getApi().listModels() 会抛
    # "TypeError: ... is not a function"，React 首次渲染直接中断、界面全白。
    # 这里保留 snake_case 实现，仅补一层契约别名。

    def getSystemInfo(self):
        return self.get_system_info()

    def runEnvCheck(self):
        return self.run_env_check()

    def openPath(self, path: str):
        return self.open_path(path)

    def openLogsFolder(self):
        return self.open_logs_folder()

    def selectFiles(self, title: str):
        return self.select_files(title)

    def selectFolder(self, title: str):
        return self.select_folder(title)

    def listModels(self):
        return self.list_models()

    def startTask(self, opts: dict):
        return self.start_task(opts)

    def cancelTask(self):
        return self.cancel_task()

    def taskRunning(self):
        return self.task_running()

    def downloadModel(self, opts: dict):
        return self.download_model(opts)

    def cancelDownload(self):
        return self.cancel_download()

    def modalCheck(self):
        return self.modal_check()

    def modalRun(self, opts: dict):
        return self.modal_run(opts)

    def modalCancel(self):
        return self.modal_cancel()

    def getGenerationConfig(self):
        return self.get_generation_config()

    def saveGenerationConfig(self, raw: str):
        return self.save_generation_config(raw)


class _StaticRequestHandler(http.server.SimpleHTTPRequestHandler):
    """前端静态资源处理器。

    - 关闭缓存，避免重新打包后加载到旧资源；
    - 禁止目录列表：历史上窗口曾显示成“文件列表页面”，就是目录索引导致的；
    - 日志走 logger，避免写 stderr（windowed 打包无控制台）。
    """

    def log_message(self, fmt, *args):  # noqa: A003
        logger.debug("[static] " + fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(http.server.HTTPStatus.NOT_FOUND, "Not found")
        return None


def _start_static_server(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    """在 127.0.0.1 的随机端口上提供 root 目录的静态资源，返回 (httpd, url)。"""
    handler = functools.partial(_StaticRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(
        target=httpd.serve_forever,
        name="frontend-static-server",
        daemon=True,
    ).start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    return httpd, f"http://{host}:{port}/"


def main():
    parser = argparse.ArgumentParser(description="TransWithAI 桌面版")
    parser.add_argument("--dev", action="store_true", help="加载 Vite 开发服务器 (http://localhost:5173)")
    parser.add_argument("--debug", action="store_true", help="打开 DevTools")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    httpd = None
    if args.dev:
        url = "http://localhost:5173"
    else:
        index_html = FRONTEND_DIST / "index.html"
        if not index_html.exists():
            webview.create_window(
                "Faster Whisper TransWithAI",
                html=(
                    "<div style='font-family:sans-serif;padding:24px'>"
                    "<h2>前端资源缺失</h2>"
                    "<p>未找到打包后的前端入口，请先在 <code>frontend/</code> 目录执行 "
                    "<code>pnpm build</code> 后重新打包。</p>"
                    f"<p style='color:#888;word-break:break-all'>期望路径：{index_html}</p>"
                    "</div>"
                ),
                width=760,
                height=280,
            )
            webview.start()
            return
        # 注意：不要用 file:// 直接打开。用 file:// 打开目录会显示目录索引（文件列表），
        # 打开 index.html 时 ES module 又会被 CORS 拦截，两者都会导致界面空白。
        # 这里起一个只监听 127.0.0.1 的静态服务，由 WebView 通过 http 加载。
        httpd, url = _start_static_server(FRONTEND_DIST)
        logger.info("前端静态资源服务: %s (root=%s)", url, FRONTEND_DIST)

    window = webview.create_window(
        "Faster Whisper TransWithAI ChickenRice",
        url,
        js_api=Api(),
        width=1280,
        height=860,
        min_size=(1000, 640),
    )
    try:
        webview.start(debug=args.debug)
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
