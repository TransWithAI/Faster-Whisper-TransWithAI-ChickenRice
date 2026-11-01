#!/usr/bin/env python3
"""桌面版 GUI 一键构建脚本。

流程:
1. 构建 React 前端 (pnpm build)
2. PyInstaller 打包 GUI 主程序（desktop/desktop.spec）
   -> dist/FasterWhisperTransWithAI-GUI/FasterWhisperTransWithAI-GUI.exe
3. 将 GUI exe 与既有 CLI 打包产物（infer.exe、models 等）合并为完整发布目录

用法:
  python desktop/build_desktop.py            # 全部步骤
  python desktop/build_desktop.py --frontend # 仅构建前端
  python desktop/build_desktop.py --skip-frontend
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
GUI_NAME = "FasterWhisperTransWithAI-GUI"


def run(cmd, cwd=None):
    print(f">>> {' '.join(cmd)}  (cwd={cwd or ROOT})")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, shell=(sys.platform == "win32"))


def build_frontend():
    run(["pnpm", "install"], cwd=FRONTEND)
    run(["pnpm", "build"], cwd=FRONTEND)


def build_gui():
    run(["pyinstaller", str(Path("desktop") / "desktop.spec"), "--noconfirm", "--distpath", "dist"])


def merge_with_cli(cli_dist_dir: Path | None):
    """把 GUI（exe + _internal 内容）合并进 CLI 发行目录。

    两个 PyInstaller onedir 产物各有一份 _internal；将 GUI 的 _internal
    内容拷贝进 CLI 的 _internal（同名文件覆盖，二者出自同一环境，依赖一致），
    合并后该目录同时支持 infer.exe 命令行与 GUI 双击运行。
    """
    gui_dir = ROOT / "dist" / GUI_NAME
    gui_exe = gui_dir / f"{GUI_NAME}.exe"
    if not gui_exe.exists():
        print(f"!! 未找到 GUI 产物: {gui_exe}")
        sys.exit(1)

    if cli_dist_dir is None:
        print("!! 未提供 CLI 发行目录（含 infer.exe 的 COLLECT 目录），跳过合并。")
        print(f"   GUI 独立产物: {gui_dir}")
        return

    target = Path(cli_dist_dir).resolve()
    if not (target / "infer.exe").exists():
        print(f"!! 指定目录中未找到 infer.exe: {target}")
        sys.exit(1)

    shutil.copy2(gui_exe, target / f"{GUI_NAME}.exe")
    print("OK  GUI exe 已拷贝")

    # 同一次 COLLECT 产出的其它入口（如 download_models.exe）也要一并带上，
    # 否则 GUI 的「下载模型」会提示“未找到 download_models 入口”。
    other_exes = [p for p in sorted(gui_dir.glob("*.exe")) if p.name != f"{GUI_NAME}.exe"]
    for exe_file in other_exes:
        shutil.copy2(exe_file, target / exe_file.name)
    if other_exes:
        print("OK  其它入口已拷贝: " + ", ".join(p.name for p in other_exes))

    gui_internal = gui_dir / "_internal"
    target_internal = target / "_internal"
    if gui_internal.is_dir():
        target_internal.mkdir(exist_ok=True)
        print(f"    合并 _internal: {gui_internal} -> {target_internal}")
        for item in gui_internal.iterdir():
            dest = target_internal / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    print(f"OK  合并完成: {target}")
    print(f"    双击 {GUI_NAME}.exe 打开桌面应用；infer.exe 仍可命令行/拖放使用。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", action="store_true", help="仅构建前端")
    parser.add_argument("--skip-frontend", action="store_true", help="跳过前端构建")
    parser.add_argument("--merge", type=str, default=None, help="CLI 发行目录（含 infer.exe），将 GUI 合并进去")
    args = parser.parse_args()

    if args.frontend:
        build_frontend()
        return
    if not args.skip_frontend:
        build_frontend()
    build_gui()
    if args.merge:
        merge_with_cli(Path(args.merge))


if __name__ == "__main__":
    main()
