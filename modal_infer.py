"""feature-modal: 交互式 CLI，完成 Modal App 构建、音频上传、推理执行与结果回传。"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

try:
    import questionary  # type: ignore
    from questionary import Choice  # type: ignore
except ImportError:  # pragma: no cover
    questionary = None  # type: ignore
    Choice = None  # type: ignore

try:
    import modal
except ImportError as exc:  # pragma: no cover
    print("未检测到 modal 包，请先运行 `python -m pip install modal questionary`。")
    raise

APP_NAME = "Faster-Whisper-TransWithAI-ChickenRice"
REPO_URL = "https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice"
VOLUME_NAME = "agent_volume"
VOLUME_ROOT = Path("/agent_volume")
REMOTE_MOUNT = VOLUME_ROOT
APP_ROOT_REL = Path(APP_NAME)
SESSION_SUBDIR = APP_ROOT_REL / "sessions"
REPO_VOLUME_DIR = VOLUME_ROOT / "repo"
SUB_FORMATS = "srt,vtt,lrc"
SUB_SUFFIXES = {".srt", ".vtt", ".lrc"}
AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".wma",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
}
DEFAULT_GPU_CHOICES = [
    "L40S",
    "A10G",
    "A100-40GB",
    "A100-80GB",
    "H100",
    "H200",
    "B200",
    "L4",
    "T4",
]


@dataclass
class ModelProfile:
    key: str
    label: str
    hf_repo: Optional[str]
    target_dir: Optional[str]
    description: str


@dataclass
class UserSelection:
    run_mode: str  # once or persistent
    gpu_choice: str
    input_path: Path
    model_profile: ModelProfile
    custom_repo: Optional[str]
    custom_target_dir: Optional[str]
    enable_batching: bool
    batch_size: Optional[int]
    max_batch_size: int
    timeout_minutes: int


@dataclass
class UploadManifest:
    session_id: str
    source_type: str  # file or directory
    local_source: Path
    remote_inputs_rel: List[Path]
    remote_output_rel: Path
    local_output_dir: Path
    remote_logs_rel: Path


@dataclass
class RemoteResult:
    created_files: Dict[str, List[str]]
    log_file: Optional[str]


def rel_to_volume_path(path: Path) -> str:
    posix = path.as_posix()
    if not posix.startswith("/"):
        posix = "/" + posix
    return posix


def rel_to_container_path(path: Path) -> str:
    return str((REMOTE_MOUNT / path).as_posix())


def volume_path_to_relative(path: str) -> Path:
    return Path(path.lstrip("/"))


def container_to_volume_path(container_path: str) -> str:
    prefix = str(REMOTE_MOUNT)
    if not container_path.startswith(prefix):
        raise ValueError(f"路径 {container_path} 不在挂载点 {prefix} 下")
    rel = container_path[len(prefix) :]
    if not rel.startswith("/"):
        rel = "/" + rel
    return rel


MODEL_PRESETS: Dict[str, ModelProfile] = {
    "chickenrice": ModelProfile(
        key="chickenrice",
        label="海南鸡（日文转中文优化）",
        hf_repo="chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2",
        target_dir="chickenrice-v2",
        description="默认高精度模型，建议 L40S 以上",
    ),
    "base": ModelProfile(
        key="base",
        label="基础版（whisper-base）",
        hf_repo="openai/whisper-base",
        target_dir="whisper-base",
        description="适合低显存/快速校验",
    ),
    "custom": ModelProfile(
        key="custom",
        label="自定义 HuggingFace 模型",
        hf_repo=None,
        target_dir=None,
        description="手动输入 HF repo 与目标目录",
    ),
}


def setup_logger() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"modal_run_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logging.info("日志输出：%s", log_path)
    return log_path


def ensure_questionary():
    if questionary is None or Choice is None:
        raise RuntimeError("需要 questionary，请运行 `python -m pip install questionary`。")


def ask_selection() -> UserSelection:
    ensure_questionary()

    run_mode = questionary.select(
        "选择运行模式：",
        choices=[
            Choice(title="一次性运行（modal run）", value="once"),
            Choice(title="持久化 App（modal deploy）", value="persistent"),
        ],
    ).ask()
    if not run_mode:
        raise KeyboardInterrupt

    gpu_choice = questionary.select(
        "选择 GPU",
        choices=DEFAULT_GPU_CHOICES,
    ).ask()
    if not gpu_choice:
        raise KeyboardInterrupt

    model_key = questionary.select(
        "选择模型：",
        choices=[Choice(title=profile.label, value=key) for key, profile in MODEL_PRESETS.items()],
    ).ask()
    if not model_key:
        raise KeyboardInterrupt

    model_profile = MODEL_PRESETS[model_key]
    custom_repo = None
    custom_target_dir = None
    if model_key == "custom":
        custom_repo = questionary.text("输入 HuggingFace repo（例如 user/repo）").ask()
        if not custom_repo:
            raise KeyboardInterrupt
        custom_target_dir = questionary.text(
            "输入 models 子目录名称（英文/数字）", default="custom-model"
        ).ask()
        if not custom_target_dir:
            raise KeyboardInterrupt

    input_path_str = questionary.path(
        "拖入或输入待处理的本地文件/文件夹路径："
    ).ask()
    if not input_path_str:
        raise KeyboardInterrupt
    input_path = Path(input_path_str).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"路径不存在：{input_path}")

    enable_batching = questionary.confirm(
        "启用批处理以加速（需要更高显存）？", default=False
    ).ask()
    if enable_batching is None:
        raise KeyboardInterrupt

    batch_size = None
    max_batch_size = 8
    if enable_batching:
        batch_size_str = questionary.text(
            "指定批次大小（留空自动探测）", default=""
        ).ask()
        if batch_size_str:
            batch_size = int(batch_size_str)
        max_batch_size_str = questionary.text(
            "最大自动批次大小", default="8"
        ).ask()
        max_batch_size = int(max_batch_size_str or "8")

    timeout_minutes = int(
        questionary.text("任务超时时间（分钟）", default="60").ask() or "60"
    )

    return UserSelection(
        run_mode=run_mode,
        gpu_choice=gpu_choice,
        input_path=input_path,
        model_profile=model_profile,
        custom_repo=custom_repo,
        custom_target_dir=custom_target_dir,
        enable_batching=bool(enable_batching),
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        timeout_minutes=timeout_minutes,
    )


def iter_audio_files(path: Path) -> List[Path]:
    files: List[Path] = []
    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in AUDIO_SUFFIXES:
            files.append(file)
    return files


def validate_audio_path(path: Path) -> None:
    if path.is_file():
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            raise ValueError(f"文件 {path} 不属于支持的音/视频格式。")
    elif path.is_dir():
        if not iter_audio_files(path):
            raise ValueError(f"文件夹 {path} 中没有支持的音/视频文件。")
    else:
        raise ValueError(f"路径 {path} 既不是文件也不是文件夹。")


def prepare_upload(
    volume: modal.Volume,
    selection: UserSelection,
) -> UploadManifest:
    validate_audio_path(selection.input_path)
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    remote_session_rel = SESSION_SUBDIR / session_id
    remote_logs_rel = remote_session_rel / "logs"
    remote_inputs_rel: List[Path] = []

    with volume.batch_upload(force=True) as batch:
        if selection.input_path.is_file():
            remote_rel = APP_ROOT_REL / selection.input_path.name
            logging.info("上传文件 -> %s", rel_to_volume_path(remote_rel))
            batch.put_file(str(selection.input_path), rel_to_volume_path(remote_rel))
            remote_inputs_rel.append(remote_rel)
            remote_output_rel = remote_session_rel
            local_output_dir = selection.input_path.parent
            source_type = "file"
        else:
            remote_input_dir_rel = remote_session_rel / selection.input_path.name
            audio_files = iter_audio_files(selection.input_path)
            for file in audio_files:
                rel = remote_input_dir_rel / file.relative_to(selection.input_path)
                logging.info("上传文件 -> %s", rel_to_volume_path(rel))
                batch.put_file(str(file), rel_to_volume_path(rel))
            remote_inputs_rel.append(remote_input_dir_rel)
            remote_output_rel = remote_session_rel / f"{selection.input_path.name}_out"
            local_output_dir = selection.input_path / f"{selection.input_path.name}_out"
            source_type = "directory"

    return UploadManifest(
        session_id=session_id,
        source_type=source_type,
        local_source=selection.input_path,
        remote_inputs_rel=remote_inputs_rel,
        remote_output_rel=remote_output_rel,
        local_output_dir=local_output_dir,
        remote_logs_rel=remote_logs_rel,
    )


def build_job_payload(selection: UserSelection, manifest: UploadManifest) -> Dict:
    model_profile = selection.model_profile
    hf_repo = selection.custom_repo if model_profile.key == "custom" else model_profile.hf_repo
    target_dir = (
        selection.custom_target_dir if model_profile.key == "custom" else model_profile.target_dir
    ) or "custom-model"

    payload = {
        "session_id": manifest.session_id,
        "mount_root": str(REMOTE_MOUNT),
        "repo_url": REPO_URL,
        "remote_inputs": [rel_to_container_path(p) for p in manifest.remote_inputs_rel],
        "remote_output_dir": rel_to_container_path(manifest.remote_output_rel),
        "output_targets": [
            {
                "remote_dir": rel_to_container_path(manifest.remote_output_rel),
                "extensions": list(SUB_SUFFIXES),
            }
        ],
        "input_mode": manifest.source_type,
        "sub_formats": SUB_FORMATS,
        "enable_batching": selection.enable_batching,
        "batch_size": selection.batch_size,
        "max_batch_size": selection.max_batch_size,
        "timeout_seconds": selection.timeout_minutes * 60,
        "model_profile": {
            "label": model_profile.label,
            "hf_repo": hf_repo,
            "target_dir": target_dir,
        },
        "remote_logs_dir": rel_to_container_path(manifest.remote_logs_rel),
        "output_suffixes": list(SUB_SUFFIXES),
    }
    return payload


def build_modal_image() -> modal.Image:
    return (
        modal.Image.micromamba(python_version="3.10")
        .apt_install("git")
        .micromamba_install(
            spec_file="environment-cuda128.yml",
            channels=["conda-forge", "defaults"],
        )
        .pip_install("modal", "questionary")
    )


def run_remote_pipeline(
    volume: modal.Volume,
    selection: UserSelection,
    manifest: UploadManifest,
    payload: Dict,
) -> RemoteResult:
    image = build_modal_image()
    logging.info("使用 GPU：%s", selection.gpu_choice)
    app = modal.App(APP_NAME)

    @app.function(
        image=image,
        gpu=selection.gpu_choice,
        timeout=selection.timeout_minutes * 60,
        volumes={str(REMOTE_MOUNT): volume},
        serialized=True,
    )
    def modal_pipeline(job_payload: Dict) -> Dict:
        return _remote_pipeline(job_payload)

    with app.run():
        result = modal_pipeline.remote(payload)
    created = {
        remote_dir: files for remote_dir, files in result.get("created", {}).items()
    }
    return RemoteResult(created_files=created, log_file=result.get("log_file"))


def download_outputs(
    manifest: UploadManifest,
    result: RemoteResult,
) -> None:
    def modal_volume_get(remote_path: str, local_dest: Path) -> None:
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info("下载 %s -> %s", remote_path, local_dest)
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dest)],
            check=True,
        )

    for remote_dir, files in result.created_files.items():
        base_rel = Path(remote_dir.lstrip("/"))
        for remote_file in files:
            file_rel = Path(remote_file.lstrip("/"))
            try:
                rel_inside_output = file_rel.relative_to(base_rel)
            except Exception:
                rel_inside_output = file_rel.name
            local_path = manifest.local_output_dir / rel_inside_output
            modal_volume_get(remote_file, local_path)

    if result.log_file:
        local_log = Path("logs") / Path(Path(result.log_file).name)
        modal_volume_get(result.log_file, local_log)


def summarize(manifest: UploadManifest, result: RemoteResult) -> None:
    logging.info("=== 运行完成 ===")
    logging.info("Session: %s", manifest.session_id)
    logging.info("源路径: %s", manifest.local_source)
    logging.info("输出路径: %s", manifest.local_output_dir if manifest.source_type == "directory" else manifest.local_source.parent)
    if result.created_files:
        logging.info("新生成文件：")
        for remote_dir, files in result.created_files.items():
            for file in files:
                logging.info("  %s", file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="feature-modal 运行脚本")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="跳过 questionary，全部通过参数指定（暂未实现）。",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    log_path = setup_logger()
    try:
        selection = ask_selection()
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        manifest = prepare_upload(volume, selection)
        payload = build_job_payload(selection, manifest)
        result = run_remote_pipeline(volume, selection, manifest, payload)
        download_outputs(manifest, result)
        summarize(manifest, result)
        logging.info("✅ 请在上方输出路径查看字幕结果。")
    except KeyboardInterrupt:
        logging.warning("用户中断，未执行任何远程操作。")
        sys.exit(1)
    except Exception as exc:
        logging.exception("运行失败：%s", exc)
        logging.error("日志见：%s", log_path)
        sys.exit(1)


def _remote_pipeline(job: Dict) -> Dict:
    import subprocess
    from pathlib import Path
    import os

    def run(cmd: Sequence[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> None:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=cwd, env=env)

    mount_root = Path(job["mount_root"])
    repo_dir = REPO_VOLUME_DIR
    logs_dir = Path(job["remote_logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "modal_run.log"

    def log(msg: str) -> None:
        line = f"[modal_run] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    if not (repo_dir / ".git").exists():
        log("开始克隆仓库...")
        run(["git", "clone", REPO_URL, str(repo_dir)])
    else:
        log("更新仓库...")
        run(["git", "-C", str(repo_dir), "fetch", "origin", "main"])
        run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"])

    model_profile = job["model_profile"]
    model_path = repo_dir / "models"
    if model_profile.get("hf_repo"):
        target_dir = model_profile["target_dir"]
        model_path = repo_dir / "models" / target_dir
        if not model_path.exists():
            log(f"模型 {model_profile['hf_repo']} 缺失，开始下载...")
            cmd = [
                "python",
                str(repo_dir / "download_models.py"),
                "--hf-model",
                model_profile["hf_repo"],
                "--target-dir",
                target_dir,
            ]
            env = os.environ.copy()
            hf_token = env.get("HF_TOKEN")
            if hf_token:
                env["HF_TOKEN"] = hf_token
            run(cmd, cwd=str(repo_dir), env=env)
    else:
        log("使用仓库默认模型目录。")

    def snapshot(path: str) -> set:
        base = Path(path)
        files = set()
        if base.exists():
            for f in base.rglob("*"):
                if f.is_file():
                    files.add(str(f))
        return files

    before = {
        target["remote_dir"]: snapshot(target["remote_dir"])
        for target in job["output_targets"]
    }

    output_dir = Path(job["remote_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(repo_dir / "infer.py"),
        "--device",
        "cuda",
        "--model_name_or_path",
        str(model_path),
        "--sub_formats",
        job["sub_formats"],
        "--log_level",
        "INFO",
        "--output_dir",
        str(output_dir),
    ]
    if job["enable_batching"]:
        cmd.append("--enable_batching")
        if job["batch_size"]:
            cmd.extend(["--batch_size", str(job["batch_size"])])
        cmd.extend(["--max_batch_size", str(job["max_batch_size"])])

    cmd.extend(job["remote_inputs"])

    log(f"执行推理命令：{' '.join(cmd)}")
    run(cmd, cwd=str(repo_dir))

    def to_volume_path(path_str: str) -> str:
        return container_to_volume_path(path_str)

    created = {}
    for target in job["output_targets"]:
        remote_dir = target["remote_dir"]
        after = snapshot(remote_dir)
        prev = before.get(remote_dir, set())
        new_files = sorted(
            file
            for file in after - prev
            if Path(file).suffix.lower() in SUB_SUFFIXES
        )
        created[to_volume_path(remote_dir)] = [to_volume_path(path) for path in new_files]

    return {"created": created, "log_file": to_volume_path(str(log_file))}


if __name__ == "__main__":  # pragma: no cover
    main()
