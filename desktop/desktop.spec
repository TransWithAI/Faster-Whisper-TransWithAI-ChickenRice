# -*- mode: python ; coding: utf-8 -*-
"""
桌面版 GUI 打包配置（不影响原有 CLI 打包 project.spec）

产物：dist/FasterWhisperTransWithAI-GUI/ 目录，含 GUI 主程序 infer.exe CLI 与桌面窗口程序。
用法:  pyinstaller desktop/desktop.spec
"""

import os
import sys

block_cipher = None

# PyInstaller 以 spec 所在目录解析相对路径，这里统一换算到项目根。
# 优先使用外部注入的 CHICKENRICE_ROOT；否则用 SPECPATH（spec 文件所在目录）推算。
ROOT = os.environ.get("CHICKENRICE_ROOT") or os.path.abspath(os.path.join(SPECPATH, ".."))
print("Desktop spec ROOT =", ROOT)
assert os.path.isdir(os.path.join(ROOT, "src")), f"ROOT 计算错误: {ROOT}"
os.chdir(ROOT)


# ---------- 复用 project.spec 的依赖收集逻辑 ----------
# 这里直接 import project.spec 会执行两次 Analysis，因此将通用部分内联：
# CUDA DLL、onnxruntime 二进制的收集逻辑与 project.spec 保持一致。

datas = []
binaries = []
hiddenimports = [
    # pywebview Windows 后端
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "clr_loader",
    "pythonnet",
]

# ---- 以下收集逻辑摘自 project.spec（conda CUDA 库） ----
import glob as _glob
import importlib.util
import site
import sysconfig

conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
print(f"Conda environment detected: {conda_prefix}")

lib_dirs = [
    os.path.join(conda_prefix, 'Library', 'bin'),
    os.path.join(conda_prefix, 'bin'),
    os.path.join(conda_prefix, 'DLLs'),
]
try:
    site_packages = site.getsitepackages()
    for sp in site_packages:
        if conda_prefix in sp:
            onnx_capi_path = os.path.join(sp, 'onnxruntime', 'capi')
            if os.path.exists(onnx_capi_path):
                lib_dirs.append(onnx_capi_path)
                print(f"  Added ONNX Runtime path: {onnx_capi_path}")
except Exception:
    pass

cuda_libs_patterns = [
    'cudart64_*.dll',
    'cublas64_*.dll',
    'cublasLt64_*.dll',
    'cudnn64_*.dll',
    'cudnn_ops_infer64_*.dll',
    'cudnn_cnn_infer64_*.dll',
    'cudnn_adv_infer64_*.dll',
    'cudnn_graph64_*.dll',
    'cudnn_heuristic64_*.dll',
    'cudnn_engines_precompiled64_*.dll',
    'cudnn_engines_runtime_compiled64_*.dll',
    'nvrtc64*.dll',
    'nvml.dll',
]

for pattern in cuda_libs_patterns:
    for lib_dir in lib_dirs:
        matches = _glob.glob(os.path.join(lib_dir, pattern))
        for m in matches:
            binaries.append((m, '.'))
            print(f"  Found CUDA binary: {m}")

# onnxruntime capi DLL
for lib_dir in lib_dirs:
    if 'onnxruntime' in lib_dir:
        for m in _glob.glob(os.path.join(lib_dir, '*.dll')):
            binaries.append((m, 'onnxruntime/capi'))

# ---------- 数据文件（datas 源路径相对 spec 目录，必须用绝对路径） ----------
datas += [
    (os.path.join(ROOT, 'src', 'faster_whisper_transwithai_chickenrice'), 'faster_whisper_transwithai_chickenrice'),
    (os.path.join(ROOT, 'locales'), 'locales'),
    (os.path.join(ROOT, 'generation_config.json5'), '.'),
    (os.path.join(ROOT, 'transwithai.ico'), '.'),
]

# 前端构建产物（要求先 pnpm build）
frontend_dist = os.path.join(ROOT, 'frontend', 'dist')
if os.path.isdir(frontend_dist):
    datas.append((frontend_dist, 'frontend/dist'))
else:
    print("WARNING: frontend/dist 不存在，请先在 frontend/ 执行 pnpm build")

# ---------- 入口 ----------
a = Analysis(
    [os.path.join(ROOT, 'desktop', 'app.py')],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(ROOT, 'runtime_hook.py')],
    excludes=[
        'matplotlib', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'notebook', 'jupyter', 'IPython', 'pytest', 'modal', 'questionary',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FasterWhisperTransWithAI-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=os.path.join(ROOT, 'transwithai.ico') if os.path.exists(os.path.join(ROOT, 'transwithai.ico')) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='FasterWhisperTransWithAI-GUI',
)
