#!/usr/bin/env python3
"""
Modal 云端推理 — 非交互 CLI（供桌面 GUI 以子进程方式调用）

复用项目根 modal_infer.py 的全部流水线函数，只是把 questionary 交互
替换为命令行参数。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal_infer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Modal 云端推理（非交互）")
    parser.add_argument("--input", required=True, help="本地音频文件或文件夹路径")
    parser.add_argument("--gpu", default="T4", help=f"GPU 类型，默认 T4（可选: {', '.join(modal_infer.DEFAULT_GPU_CHOICES)}）")
    parser.add_argument(
        "--model",
        default="chickenrice",
        choices=list(modal_infer.MODEL_PRESETS.keys()),
        help="模型预设",
    )
    parser.add_argument("--custom-repo", default=None, help="自定义 HuggingFace repo（model=custom 时必填）")
    parser.add_argument("--custom-target-dir", default="custom-model", help="自定义模型目标目录名")
    parser.add_argument("--enable-batching", action="store_true", help="启用批处理加速")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小（留空自动探测）")
    parser.add_argument("--max-batch-size", type=int, default=8, help="最大自动批次大小")
    parser.add_argument("--timeout-minutes", type=int, default=60, help="任务超时（分钟）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    profile = modal_infer.MODEL_PRESETS[args.model]
    custom_repo = None
    custom_target_dir = None
    if args.model == "custom":
        if not args.custom_repo:
            print("[gui] 自定义模型必须提供 --custom-repo")
            return 2
        custom_repo = args.custom_repo
        custom_target_dir = args.custom_target_dir

    selection = modal_infer.UserSelection(
        run_mode="once",
        gpu_choice=args.gpu,
        input_path=Path(args.input).expanduser().resolve(),
        model_profile=profile,
        custom_repo=custom_repo,
        custom_target_dir=custom_target_dir,
        enable_batching=args.enable_batching,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        timeout_minutes=args.timeout_minutes,
    )

    modal_infer.setup_logger()
    exit_code = 0
    try:
        import modal

        if not selection.input_path.exists():
            raise FileNotFoundError(f"路径不存在：{selection.input_path}")
        volume = modal.Volume.from_name(modal_infer.VOLUME_NAME, create_if_missing=True)
        scan_result = modal_infer.validate_audio_path(selection.input_path)

        if selection.input_path.is_dir():
            print(f"[gui] 文件夹模式，共 {len(scan_result.audio_files)} 个音频文件")
            success_count, fail_count = modal_infer.process_directory_files(
                volume, selection, scan_result.audio_files
            )
            print(f"[gui] 批量处理完成：成功 {success_count}，失败 {fail_count}")
        else:
            manifest = modal_infer.upload_single_file(volume, selection, selection.input_path)
            payload = modal_infer.build_job_payload(selection, manifest)
            result = modal_infer.run_remote_pipeline(volume, selection, manifest, payload)
            modal_infer.download_outputs(manifest, result)
            modal_infer.summarize(manifest, result)
        print("[gui] 请在源文件目录查看字幕结果。")
    except modal_infer.NoAudioFilesError as exc:
        print(f"[gui] {exc}")
        exit_code = 1
    except KeyboardInterrupt:
        print("[gui] 用户中断。")
        exit_code = 1
    except Exception as exc:
        import logging
        import traceback

        traceback.print_exc()
        logging.exception("运行失败：%s", exc)
        print(f"[gui] 运行失败：{exc}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
