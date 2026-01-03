"""feature-modal: 交互式 CLI，完成 Modal App 构建、音频上传、推理执行与结果回传。"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Dict, List, Optional, Sequence, Tuple
=======
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
=======
from typing import Dict, List, Optional, Sequence, Tuple
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
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
<<<<<<< HEAD
VOLUME_NAME = "Faster_Whisper"
VOLUME_ROOT = Path("/Faster_Whisper")
REMOTE_MOUNT = VOLUME_ROOT
APP_ROOT_REL = Path(APP_NAME)
SESSION_SUBDIR = Path("sessions")
=======
VOLUME_NAME = "agent_volume"
VOLUME_ROOT = Path("/agent_volume")
REMOTE_MOUNT = VOLUME_ROOT
APP_ROOT_REL = Path(APP_NAME)
SESSION_SUBDIR = APP_ROOT_REL / "sessions"
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
    ".mp4",
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
=======
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
}
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
VIDEO_NEED_CONVERT = {".mp4"}  # 需要用户手动转换的格式
DEFAULT_GPU_CHOICES = [
    "T4",
    "L4",
=======
DEFAULT_GPU_CHOICES = [
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
    "L40S",
    "A10G",
    "A100-40GB",
    "A100-80GB",
    "H100",
    "H200",
    "B200",
<<<<<<< HEAD
=======
    "L4",
    "T4",
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
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
<<<<<<< HEAD
    original_filename: Optional[str] = None  # 原始文件名（用于恢复空格）


@dataclass
class ScanResult:
    audio_files: List[Path]
    mp4_files: List[Path]


class NoAudioFilesError(Exception):
    pass
=======


@dataclass
class ScanResult:
    audio_files: List[Path]
    mp4_files: List[Path]


@dataclass
class RemoteResult:
    created_files: Dict[str, List[str]]
    log_file: Optional[str]
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)


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

<<<<<<< HEAD
=======
    run_mode = questionary.select(
        "选择运行模式：",
        choices=[
            Choice(title="一次性运行（modal run）", value="once"),
            Choice(title="持久化 App（modal deploy）", value="persistent"),
        ],
    ).ask()
    if not run_mode:
        raise KeyboardInterrupt

>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
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
<<<<<<< HEAD
        run_mode="once",
=======
        run_mode=run_mode,
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
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


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
def scan_audio_files(path: Path) -> ScanResult:
    """扫描目录，返回音频文件和需要转换的 mp4 文件"""
    audio_files: List[Path] = []
    mp4_files: List[Path] = []
<<<<<<< HEAD
    for file in path.rglob("*"):
        if file.is_file():
            suffix = file.suffix.lower()
            if suffix in AUDIO_SUFFIXES:
                audio_files.append(file)
            elif suffix in VIDEO_NEED_CONVERT:
                mp4_files.append(file)
    return ScanResult(audio_files=audio_files, mp4_files=mp4_files)


def validate_audio_path(path: Path) -> ScanResult:
    """验证音频路径，返回扫描结果。如果发现 mp4 文件会打印警告。"""
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in VIDEO_NEED_CONVERT:
            raise ValueError(
                f"文件 {path} 是 mp4 格式，请先使用 ffmpeg 转换为 mp3：\n"
                f"  ffmpeg -i \"{path}\" -vn -acodec libmp3lame \"{path.with_suffix('.mp3')}\""
            )
        if suffix not in AUDIO_SUFFIXES:
            raise ValueError(f"文件 {path} 不属于支持的音/视频格式。")
        return ScanResult(audio_files=[path], mp4_files=[])
    elif path.is_dir():
        scan_result = scan_audio_files(path)
        if scan_result.mp4_files:
            logging.warning("=" * 60)
            logging.warning("发现 %d 个 mp4 文件，这些文件将被跳过：", len(scan_result.mp4_files))
            for mp4_file in scan_result.mp4_files:
                logging.warning("  - %s", mp4_file)
            logging.warning("请使用 ffmpeg 转换为 mp3 后再处理，例如：")
            logging.warning("  ffmpeg -i \"input.mp4\" -vn -acodec libmp3lame \"output.mp3\"")
            logging.warning("=" * 60)
        if not scan_result.audio_files:
            raise NoAudioFilesError(f"输入的文件夹内没有音频文件：{path}")
        return scan_result
=======
def iter_audio_files(path: Path) -> List[Path]:
    files: List[Path] = []
=======
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
    for file in path.rglob("*"):
        if file.is_file():
            suffix = file.suffix.lower()
            if suffix in AUDIO_SUFFIXES:
                audio_files.append(file)
            elif suffix in VIDEO_NEED_CONVERT:
                mp4_files.append(file)
    return ScanResult(audio_files=audio_files, mp4_files=mp4_files)


def validate_audio_path(path: Path) -> ScanResult:
    """验证音频路径，返回扫描结果。如果发现 mp4 文件会打印警告。"""
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in VIDEO_NEED_CONVERT:
            raise ValueError(
                f"文件 {path} 是 mp4 格式，请先使用 ffmpeg 转换为 mp3：\n"
                f"  ffmpeg -i \"{path}\" -vn -acodec libmp3lame \"{path.with_suffix('.mp3')}\""
            )
        if suffix not in AUDIO_SUFFIXES:
            raise ValueError(f"文件 {path} 不属于支持的音/视频格式。")
        return ScanResult(audio_files=[path], mp4_files=[])
    elif path.is_dir():
        scan_result = scan_audio_files(path)
        if scan_result.mp4_files:
            logging.warning("=" * 60)
            logging.warning("发现 %d 个 mp4 文件，这些文件将被跳过：", len(scan_result.mp4_files))
            for mp4_file in scan_result.mp4_files:
                logging.warning("  - %s", mp4_file)
            logging.warning("请使用 ffmpeg 转换为 mp3 后再处理，例如：")
            logging.warning("  ffmpeg -i \"input.mp4\" -vn -acodec libmp3lame \"output.mp3\"")
            logging.warning("=" * 60)
        if not scan_result.audio_files:
            raise ValueError(f"文件夹 {path} 中没有支持的音/视频文件。")
<<<<<<< HEAD
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
=======
        return scan_result
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
    else:
        raise ValueError(f"路径 {path} 既不是文件也不是文件夹。")


<<<<<<< HEAD
<<<<<<< HEAD
def upload_single_file(
    volume: modal.Volume,
    selection: UserSelection,
    audio_file: Path,
    base_dir: Optional[Path] = None,
) -> UploadManifest:
    """上传单个音频文件到 Modal Volume。

    Args:
        volume: Modal Volume 实例
        selection: 用户选择配置
        audio_file: 要上传的音频文件路径
        base_dir: 基础目录（用于文件夹模式，输出到此目录）
    """
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    remote_session_rel = SESSION_SUBDIR / session_id
    remote_logs_rel = remote_session_rel / "logs"

    # 使用固定文件名避免全角字符等问题
    original_filename = audio_file.name
    safe_filename = "todo" + audio_file.suffix.lower()

    with volume.batch_upload(force=True) as batch:
        remote_rel = remote_session_rel / safe_filename
        logging.info("上传文件 -> %s", rel_to_volume_path(remote_rel))
        batch.put_file(str(audio_file), rel_to_volume_path(remote_rel))

    # 如果指定了 base_dir（文件夹模式），输出到 base_dir；否则输出到文件所在目录
    local_output_dir = base_dir if base_dir else audio_file.parent

    return UploadManifest(
        session_id=session_id,
        source_type="file",
        local_source=audio_file,
        remote_inputs_rel=[remote_rel],
        remote_output_rel=remote_session_rel,
        local_output_dir=local_output_dir,
        remote_logs_rel=remote_logs_rel,
        original_filename=original_filename,  # 始终记录原始文件名
=======
def prepare_upload(
=======
def upload_single_file(
>>>>>>> 3f6ae4b (refactor: 优化文件夹处理逻辑，逐文件上传避免连接断开)
    volume: modal.Volume,
    selection: UserSelection,
    audio_file: Path,
    base_dir: Optional[Path] = None,
) -> UploadManifest:
    """上传单个音频文件到 Modal Volume。

    Args:
        volume: Modal Volume 实例
        selection: 用户选择配置
        audio_file: 要上传的音频文件路径
        base_dir: 基础目录（用于文件夹模式，输出到此目录）
    """
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    remote_session_rel = SESSION_SUBDIR / session_id
    remote_logs_rel = remote_session_rel / "logs"

    with volume.batch_upload(force=True) as batch:
        remote_rel = remote_session_rel / audio_file.name
        logging.info("上传文件 -> %s", rel_to_volume_path(remote_rel))
        batch.put_file(str(audio_file), rel_to_volume_path(remote_rel))

    # 如果指定了 base_dir（文件夹模式），输出到 base_dir；否则输出到文件所在目录
    local_output_dir = base_dir if base_dir else audio_file.parent

    return UploadManifest(
        session_id=session_id,
        source_type="file",
        local_source=audio_file,
        remote_inputs_rel=[remote_rel],
        remote_output_rel=remote_session_rel,
        local_output_dir=local_output_dir,
        remote_logs_rel=remote_logs_rel,
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
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
<<<<<<< HEAD
) -> Dict:
    logging.info("=== 开始构建 Modal 镜像 ===")
    image = build_modal_image()
    logging.info("✓ 镜像构建完成")
    logging.info("使用 GPU：%s", selection.gpu_choice)
    logging.info("超时时间：%d 分钟", selection.timeout_minutes)
=======
) -> RemoteResult:
    logging.info("=== 开始构建 Modal 镜像 ===")
    image = build_modal_image()
    logging.info("✓ 镜像构建完成")
    logging.info("使用 GPU：%s", selection.gpu_choice)
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
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

<<<<<<< HEAD
    logging.info("=== 开始远程执行 ===")
    logging.info("正在启动 GPU 容器并执行推理任务...")
    logging.info("（以下为远程容器输出）")
    logging.info("-" * 60)
    with app.run():
        result = modal_pipeline.remote(payload)
    logging.info("-" * 60)
    logging.info("✓ 远程执行完成")
    return result  # 直接返回 Dict，包含 created_files 和 log_content


def process_directory_files(
    volume: modal.Volume,
    selection: UserSelection,
    audio_files: List[Path],
) -> Tuple[int, int]:
    """处理文件夹中的所有音频文件，容器复用。

    Args:
        volume: Modal Volume 实例
        selection: 用户选择配置
        audio_files: 要处理的音频文件列表

    Returns:
        (成功数, 失败数) 元组
    """
    logging.info("=== 开始构建 Modal 镜像 ===")
    image = build_modal_image()
    logging.info("✓ 镜像构建完成")
    logging.info("使用 GPU：%s", selection.gpu_choice)
    logging.info("超时时间：%d 分钟", selection.timeout_minutes)
    logging.info("待处理文件数：%d", len(audio_files))

    app = modal.App(APP_NAME)

    @app.function(
        image=image,
        gpu=selection.gpu_choice,
        timeout=selection.timeout_minutes * 60,
        volumes={str(REMOTE_MOUNT): volume},
        serialized=True,
        min_containers=1,  # 保持容器预热，复用容器
    )
    def modal_pipeline(job_payload: Dict) -> Dict:
        return _remote_pipeline(job_payload)

    success_count = 0
    fail_count = 0
    base_dir = selection.input_path  # 文件夹模式下，输出到源文件夹

    with app.run():
        for i, audio_file in enumerate(audio_files, 1):
            logging.info("=" * 60)
            logging.info("处理文件 [%d/%d]: %s", i, len(audio_files), audio_file.name)
            logging.info("=" * 60)
            try:
                # 1. 上传单个文件
                manifest = upload_single_file(volume, selection, audio_file, base_dir)

                # 2. 构建 payload
                payload = build_job_payload(selection, manifest)

                # 3. 执行推理（复用容器）
                logging.info("正在执行推理...")
                result = modal_pipeline.remote(payload)

                # 4. 写入结果文件到本地
                download_outputs(manifest, result)

                logging.info("✓ 文件 %s 处理完成", audio_file.name)
                success_count += 1
            except Exception as e:
                logging.error("✗ 文件 %s 处理失败: %s", audio_file.name, e)
                fail_count += 1
                continue  # 继续处理下一个文件

    return success_count, fail_count
=======
    with app.run():
        result = modal_pipeline.remote(payload)
    logging.info("-" * 60)
    logging.info("✓ 远程执行完成")
    created = {
        remote_dir: files for remote_dir, files in result.get("created", {}).items()
    }
    return RemoteResult(created_files=created, log_file=result.get("log_file"))
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)


def process_directory_files(
    volume: modal.Volume,
    selection: UserSelection,
    audio_files: List[Path],
) -> Tuple[int, int]:
    """处理文件夹中的所有音频文件，容器复用。

    Args:
        volume: Modal Volume 实例
        selection: 用户选择配置
        audio_files: 要处理的音频文件列表

    Returns:
        (成功数, 失败数) 元组
    """
    logging.info("=== 开始构建 Modal 镜像 ===")
    image = build_modal_image()
    logging.info("✓ 镜像构建完成")
    logging.info("使用 GPU：%s", selection.gpu_choice)
    logging.info("超时时间：%d 分钟", selection.timeout_minutes)
    logging.info("待处理文件数：%d", len(audio_files))

    app = modal.App(APP_NAME)

    @app.function(
        image=image,
        gpu=selection.gpu_choice,
        timeout=selection.timeout_minutes * 60,
        volumes={str(REMOTE_MOUNT): volume},
        serialized=True,
        min_containers=1,  # 保持容器预热，复用容器
    )
    def modal_pipeline(job_payload: Dict) -> Dict:
        return _remote_pipeline(job_payload)

    success_count = 0
    fail_count = 0
    base_dir = selection.input_path  # 文件夹模式下，输出到源文件夹

    with app.run():
        for i, audio_file in enumerate(audio_files, 1):
            logging.info("=" * 60)
            logging.info("处理文件 [%d/%d]: %s", i, len(audio_files), audio_file.name)
            logging.info("=" * 60)
            try:
                # 1. 上传单个文件
                manifest = upload_single_file(volume, selection, audio_file, base_dir)

                # 2. 构建 payload
                payload = build_job_payload(selection, manifest)

                # 3. 执行推理（复用容器）
                logging.info("正在执行推理...")
                result = modal_pipeline.remote(payload)

                # 4. 下载结果到源文件夹
                remote_result = RemoteResult(
                    created_files=result.get("created", {}),
                    log_file=result.get("log_file"),
                )
                download_outputs(manifest, remote_result)

                logging.info("✓ 文件 %s 处理完成", audio_file.name)
                success_count += 1
            except Exception as e:
                logging.error("✗ 文件 %s 处理失败: %s", audio_file.name, e)
                fail_count += 1
                continue  # 继续处理下一个文件

    return success_count, fail_count


def download_outputs(
    manifest: UploadManifest,
<<<<<<< HEAD
    result: Dict,
) -> None:
    """从远程结果中提取文件内容并写入本地"""
    import base64

    created_files = result.get("created_files", {})
    log_content = result.get("log_content")

    # 获取原始文件名的 stem（不含扩展名）
    original_stem = Path(manifest.original_filename).stem if manifest.original_filename else "todo"

    for filename, content_b64 in created_files.items():
        content = base64.b64decode(content_b64)
        # 将 todo.xxx 替换为原始文件名
        if filename.startswith("todo."):
            suffix = Path(filename).suffix
            new_filename = original_stem + suffix
        else:
            new_filename = filename

        local_path = manifest.local_output_dir / new_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        logging.info("写入文件: %s (%d bytes)", local_path, len(content))

    # 写入 log 文件
    if log_content:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"modal_run_{manifest.session_id}.log"
        log_path.write_bytes(base64.b64decode(log_content))
        logging.info("写入日志: %s", log_path)


def summarize(manifest: UploadManifest, result: Dict) -> None:
=======
    result: RemoteResult,
) -> None:
    def modal_volume_get(remote_path: str, local_dest: Path) -> None:
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info("下载 %s -> %s", remote_path, local_dest)
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dest), "--force"],
            check=True,
        )

    for remote_dir, files in result.created_files.items():
        base_rel = Path(remote_dir.lstrip("/"))
        for remote_file in files:
            file_rel = Path(remote_file.lstrip("/"))
            try:
                rel_inside_output = file_rel.relative_to(base_rel)
            except Exception:
                rel_inside_output = Path(file_rel.name)
            local_path = manifest.local_output_dir / rel_inside_output
            modal_volume_get(remote_file, local_path)

            # 如果有原始文件名，恢复原始文件名
            if manifest.original_filename:
                original_stem = Path(manifest.original_filename).stem
                if local_path.stem == "todo":  # 固定文件名
                    new_name = original_stem + local_path.suffix
                    new_path = local_path.parent / new_name
                    logging.info("恢复原始文件名: %s -> %s", local_path.name, new_name)
                    local_path.rename(new_path)

    if result.log_file:
        local_log = Path("logs") / Path(Path(result.log_file).name)
        modal_volume_get(result.log_file, local_log)


def summarize(manifest: UploadManifest, result: RemoteResult) -> None:
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
    logging.info("=== 运行完成 ===")
    logging.info("Session: %s", manifest.session_id)
    logging.info("源路径: %s", manifest.local_source)
    logging.info("输出路径: %s", manifest.local_output_dir if manifest.source_type == "directory" else manifest.local_source.parent)
<<<<<<< HEAD
    created_files = result.get("created_files", {})
    if created_files:
        logging.info("新生成文件：")
        for filename in created_files.keys():
            logging.info("  %s", filename)
=======
    if result.created_files:
        logging.info("新生成文件：")
        for remote_dir, files in result.created_files.items():
            for file in files:
                logging.info("  %s", file)
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="feature-modal 运行脚本")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="跳过 questionary，全部通过参数指定（暂未实现）。",
    )
    return parser.parse_args()


<<<<<<< HEAD
def prompt_exit(enabled: bool) -> None:
    if not enabled:
        return
    try:
        input("输入任意键退出...")
    except EOFError:
        pass


def main() -> int:
    args = parse_args()
    log_path = setup_logger()
    exit_code = 0
    try:
        selection = ask_selection()
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

        # 验证路径并获取扫描结果
        scan_result = validate_audio_path(selection.input_path)

        if selection.input_path.is_dir():
            # 文件夹模式：逐个处理文件，容器复用
            logging.info("检测到文件夹输入，将逐个处理 %d 个音频文件", len(scan_result.audio_files))
            success_count, fail_count = process_directory_files(
                volume, selection, scan_result.audio_files
            )
            logging.info("=" * 60)
            logging.info("=== 批量处理完成 ===")
            logging.info("成功: %d, 失败: %d", success_count, fail_count)
            logging.info("输出路径: %s", selection.input_path)
            logging.info("✅ 请在上方输出路径查看字幕结果。")
        else:
            # 单文件模式：保持原有逻辑
            manifest = upload_single_file(volume, selection, selection.input_path)
            payload = build_job_payload(selection, manifest)
            result = run_remote_pipeline(volume, selection, manifest, payload)
            download_outputs(manifest, result)
            summarize(manifest, result)
            logging.info("✅ 请在上方输出路径查看字幕结果。")
    except KeyboardInterrupt:
        logging.warning("用户中断，未执行任何远程操作。")
        exit_code = 1
    except Exception as exc:
        if isinstance(exc, NoAudioFilesError):
            logging.error("%s", exc)
        else:
            logging.exception("运行失败：%s", exc)
        logging.error("日志见：%s", log_path)
        exit_code = 1


    prompt_exit(not args.non_interactive)
    return exit_code

=======
def main() -> None:
    parse_args()
    log_path = setup_logger()
    try:
        selection = ask_selection()
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

        # 验证路径并获取扫描结果
        scan_result = validate_audio_path(selection.input_path)

        if selection.input_path.is_dir():
            # 文件夹模式：逐个处理文件，容器复用
            logging.info("检测到文件夹输入，将逐个处理 %d 个音频文件", len(scan_result.audio_files))
            success_count, fail_count = process_directory_files(
                volume, selection, scan_result.audio_files
            )
            logging.info("=" * 60)
            logging.info("=== 批量处理完成 ===")
            logging.info("成功: %d, 失败: %d", success_count, fail_count)
            logging.info("输出路径: %s", selection.input_path)
            logging.info("✅ 请在上方输出路径查看字幕结果。")
        else:
            # 单文件模式：保持原有逻辑
            manifest = upload_single_file(volume, selection, selection.input_path)
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


>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
def _remote_pipeline(job: Dict) -> Dict:
    import subprocess
    from pathlib import Path
    import os

<<<<<<< HEAD
    # 强制重新加载 Volume，确保看到最新上传的文件
    from modal import Volume
    volume = Volume.from_name("Faster_Whisper")
    volume.reload()

=======
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
    def run(cmd: Sequence[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> None:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=cwd, env=env)

    mount_root = Path(job["mount_root"])
    repo_dir = REPO_VOLUME_DIR
<<<<<<< HEAD

    # log 文件放在 session 目录下，而不是 logs 子目录
    session_dir = Path(job["remote_output_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = session_dir / "modal_run.log"
=======
    logs_dir = Path(job["remote_logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "modal_run.log"
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)

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

<<<<<<< HEAD
    # 在执行推理前，等待文件同步完成
    import time
    log("等待文件同步...")
    for input_path in job["remote_inputs"]:
        input_file = Path(input_path)
        session_dir = input_file.parent
        max_wait = 180  # 最多等待 3 分钟
        waited = 0
        while not input_file.exists() and waited < max_wait:
            time.sleep(1)
            waited += 1
            if waited % 10 == 0:
                # 打印当前 session 目录下的文件
                if session_dir.exists():
                    files = list(session_dir.iterdir())
                    file_names = [f.name for f in files]
                    log(f"等待文件出现: {input_path} ({waited}s)")
                    log(f"  当前 {session_dir} 下有 {len(files)} 个文件/文件夹: {file_names}")
                else:
                    log(f"等待文件出现: {input_path} ({waited}s)")
                    log(f"  目录不存在: {session_dir}")

        if input_file.exists():
            log(f"文件已就绪: {input_path}")
        else:
            log(f"警告: 文件未出现: {input_path}")

    log(f"执行推理命令：{' '.join(cmd)}")
    try:
        run(cmd, cwd=str(repo_dir))
    except subprocess.CalledProcessError as e:
        # 打印调试信息
        sessions_dir = mount_root / "sessions"
        log(f"推理命令执行失败，错误码: {e.returncode}")
        log(f"=== 调试信息 ===")

        # 统计 sessions 目录下的文件夹数量
        if sessions_dir.exists():
            session_folders = [d for d in sessions_dir.iterdir() if d.is_dir()]
            log(f"sessions 目录下共有 {len(session_folders)} 个文件夹")
        else:
            log(f"sessions 目录不存在: {sessions_dir}")

        # 打印待处理文件所在目录的内容
        for input_path in job["remote_inputs"]:
            input_dir = Path(input_path).parent
            log(f"待处理文件目录: {input_dir}")
            if input_dir.exists():
                log(f"目录内容:")
                for item in input_dir.iterdir():
                    item_type = "目录" if item.is_dir() else "文件"
                    log(f"  [{item_type}] {item.name}")
            else:
                log(f"目录不存在: {input_dir}")

        log(f"=== 调试信息结束 ===")
        raise
=======
    log(f"执行推理命令：{' '.join(cmd)}")
    run(cmd, cwd=str(repo_dir))
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)

    def to_volume_path(path_str: str) -> str:
        return container_to_volume_path(path_str)

<<<<<<< HEAD
    # 收集生成的文件内容（直接返回，避免 volume 同步问题）
    import base64
    created_files = {}  # {filename: base64_content}
=======
    created = {}
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
    for target in job["output_targets"]:
        remote_dir = target["remote_dir"]
        after = snapshot(remote_dir)
        prev = before.get(remote_dir, set())
        new_files = sorted(
            file
            for file in after - prev
            if Path(file).suffix.lower() in SUB_SUFFIXES
        )
<<<<<<< HEAD
        for file_path in new_files:
            file_path = Path(file_path)
            if file_path.exists():
                content = file_path.read_bytes()
                # 文件名使用相对于 remote_dir 的路径
                rel_name = file_path.name
                created_files[rel_name] = base64.b64encode(content).decode()
                log(f"读取生成文件: {file_path} ({len(content)} bytes)")

    # 读取 log 文件内容
    log_content = None
    if log_file.exists():
        log_content = base64.b64encode(log_file.read_bytes()).decode()

    return {
        "created_files": created_files,  # {filename: base64_content}
        "log_content": log_content,
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
=======
        created[to_volume_path(remote_dir)] = [to_volume_path(path) for path in new_files]

    return {"created": created, "log_file": to_volume_path(str(log_file))}


if __name__ == "__main__":  # pragma: no cover
    main()
>>>>>>> fe20a3c (feat: Add Modal cloud GPU inference support)
