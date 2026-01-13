# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
from pathlib import Path
import glob

block_cipher = None

# Collect all data and binaries from critical packages
datas = []
binaries = []
hiddenimports = []

# Function to detect conda environment and CUDA version
def get_conda_cuda_libs():
    """Detect and collect CUDA/cuDNN libraries from the active conda environment"""
    cuda_binaries = []

    # Get the conda environment path
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    print(f"Conda environment detected: {conda_prefix}")

    # Detect CUDA version from environment path or libraries
    cuda_version = None
    if 'cu118' in conda_prefix or 'cuda118' in conda_prefix:
        cuda_version = '11.8'
    elif 'cu122' in conda_prefix or 'cuda122' in conda_prefix:
        cuda_version = '12.2'
    elif 'cu128' in conda_prefix or 'cuda128' in conda_prefix:
        cuda_version = '12.8'
    else:
        # Try to detect from cudart version
        cudart_files = glob.glob(os.path.join(conda_prefix, 'lib', 'libcudart.so.*'))
        if cudart_files:
            cudart_file = os.path.basename(cudart_files[0])
            if '11.8' in cudart_file:
                cuda_version = '11.8'
            elif '12.2' in cudart_file:
                cuda_version = '12.2'
            elif '12.8' in cudart_file:
                cuda_version = '12.8'

    print(f"Detected CUDA version: {cuda_version}")

    # Library paths to check - Windows uses different paths than Linux
    if sys.platform == 'win32':
        lib_dirs = [
            os.path.join(conda_prefix, 'Library', 'bin'),  # Primary location for Windows DLLs
            os.path.join(conda_prefix, 'bin'),              # Alternative location
            os.path.join(conda_prefix, 'DLLs'),             # Python DLLs location
        ]

        # Also check Python site-packages for ONNX Runtime libraries
        import site
        site_packages = site.getsitepackages()
        for sp in site_packages:
            if conda_prefix in sp:
                onnx_capi_path = os.path.join(sp, 'onnxruntime', 'capi')
                if os.path.exists(onnx_capi_path):
                    lib_dirs.append(onnx_capi_path)
                    print(f"  Added ONNX Runtime path: {onnx_capi_path}")

        # Windows CUDA library patterns with version numbers
        cuda_libs_patterns = [
            # CUDA Runtime
            'cudart64_*.dll',
            'cudart32_*.dll',  # 32-bit variant if exists
            # cuBLAS
            'cublas64_*.dll',
            'cublasLt64_*.dll',
            # cuDNN libraries - critical for deep learning
            'cudnn64_*.dll',
            'cudnn_ops_infer64_*.dll',
            'cudnn_ops_train64_*.dll',
            'cudnn_cnn_infer64_*.dll',
            'cudnn_cnn_train64_*.dll',
            'cudnn_adv_infer64_*.dll',
            'cudnn_adv_train64_*.dll',
            # For newer cuDNN versions (9.x)
            'cudnn*.dll',
            # cuFFT
            'cufft64_*.dll',
            'cufftw64_*.dll',
            # cuRAND
            'curand64_*.dll',
            # cuSPARSE
            'cusparse64_*.dll',
            # cuSOLVER
            'cusolver64_*.dll',
            'cusolverMg64_*.dll',
            # NVRTC
            'nvrtc64_*.dll',
            'nvrtc-builtins64_*.dll',
            # NVIDIA Tools Extension
            'nvToolsExt64_*.dll',
            # Additional potential libraries
            'nppc64_*.dll',
            'nppif64_*.dll',
            'npps64_*.dll',
            # ONNX Runtime GPU dependencies (important!)
            'onnxruntime_providers_cuda.dll',
            'onnxruntime_providers_tensorrt.dll',
            'onnxruntime_providers_shared.dll',
            # Python binding for ONNX Runtime
            'onnxruntime_pybind11_state*.pyd',
        ]
    else:
        # Linux/Unix library paths
        lib_dirs = [
            os.path.join(conda_prefix, 'lib'),
            os.path.join(conda_prefix, 'lib', 'stubs'),
        ]

        # Also check Python site-packages for ONNX Runtime libraries
        # This is crucial for finding libonnxruntime_providers_cuda.so, etc.
        import site
        # Try to get the site-packages directory in the conda environment
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages_paths = [
            os.path.join(conda_prefix, 'lib', python_version, 'site-packages'),
            os.path.join(conda_prefix, 'lib', 'python3.10', 'site-packages'),  # Fallback for CI
            os.path.join(conda_prefix, 'lib', 'python3.11', 'site-packages'),  # Alternative version
        ]

        for sp_path in site_packages_paths:
            onnx_capi_path = os.path.join(sp_path, 'onnxruntime', 'capi')
            if os.path.exists(onnx_capi_path):
                lib_dirs.append(onnx_capi_path)
                print(f"  Added ONNX Runtime path: {onnx_capi_path}")
                break

        # Linux CUDA library patterns
        cuda_libs_patterns = [
            'libcudart.so*',
            'libcublas.so*',
            'libcublasLt.so*',
            'libcudnn*.so*',
            'libcufft.so*',
            'libcufftw.so*',
            'libcurand.so*',
            'libcusparse.so*',
            'libcusolver.so*',
            'libnvrtc.so*',
            'libnvToolsExt.so*',
            # ONNX Runtime GPU dependencies
            'libonnxruntime_providers_cuda.so*',
            'libonnxruntime_providers_tensorrt.so*',
            'libonnxruntime_providers_shared.so*',
            # Also check without 'lib' prefix (for files in capi directory)
            'onnxruntime_providers_cuda.so*',
            'onnxruntime_providers_tensorrt.so*',
            'onnxruntime_providers_shared.so*',
            # Python extension module
            'onnxruntime_pybind11_state*.so',
        ]

    # Collect all matching libraries
    for lib_dir in lib_dirs:
        if not os.path.exists(lib_dir):
            continue

        for pattern in cuda_libs_patterns:
            for lib_file in glob.glob(os.path.join(lib_dir, pattern)):
                if os.path.isfile(lib_file) and not os.path.islink(lib_file):
                    # Add to binaries list with destination directory
                    dest_dir = '.'
                    if 'stubs' in lib_file:
                        dest_dir = 'stubs'
                    cuda_binaries.append((lib_file, dest_dir))
                    print(f"  Including CUDA library: {os.path.basename(lib_file)}")


    return cuda_binaries

# Collect CUDA/cuDNN libraries
cuda_binaries = get_conda_cuda_libs()
binaries += cuda_binaries

# Collect CTranslate2 (the actual inference engine for faster-whisper)
try:
    ctranslate2_datas, ctranslate2_binaries, ctranslate2_hiddenimports = collect_all('ctranslate2')
    datas += ctranslate2_datas
    binaries += ctranslate2_binaries
    hiddenimports += ctranslate2_hiddenimports
except:
    pass

# Collect faster-whisper
faster_whisper_datas, faster_whisper_binaries, faster_whisper_hiddenimports = collect_all('faster_whisper')
datas += faster_whisper_datas
binaries += faster_whisper_binaries
hiddenimports += faster_whisper_hiddenimports

# Collect transformers (needed for tokenizers)
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
datas += transformers_datas
binaries += transformers_binaries
hiddenimports += transformers_hiddenimports

# Collect onnxruntime for VAD model
# Note: The Python module is always 'onnxruntime' regardless of whether
# you installed onnxruntime-gpu or onnxruntime via pip
onnx_collected = False
onnx_package = 'onnxruntime'  # Module name is always 'onnxruntime'
try:
    onnx_datas, onnx_binaries, onnx_hiddenimports = collect_all(onnx_package)
    datas += onnx_datas
    binaries += onnx_binaries
    hiddenimports += onnx_hiddenimports
    print(f"Collected {onnx_package} successfully")
    onnx_collected = True

    # Explicitly add ONNX Runtime capi libraries if not already included
    try:
        import importlib.util
        spec = importlib.util.find_spec(onnx_package)
        if spec and spec.origin:
            onnx_path = os.path.dirname(spec.origin)
            capi_path = os.path.join(onnx_path, 'capi')

            if os.path.exists(capi_path):
                print(f"  Found ONNX Runtime capi directory: {capi_path}")
                for file in os.listdir(capi_path):
                    if file.endswith(('.so', '.dll', '.pyd', '.dylib')):
                        src = os.path.join(capi_path, file)
                        # Add to root directory of the bundle
                        binaries.append((src, '.'))
                        print(f"    Added capi library: {file}")
    except Exception as e:
        print(f"  Warning: Could not collect capi libraries: {e}")

except Exception as e:
    print(f"Could not collect {onnx_package}: {e}")
    onnx_collected = False

if not onnx_collected:
    print("WARNING: Could not collect any ONNX Runtime package")

# Collect librosa for audio processing
librosa_datas, librosa_binaries, librosa_hiddenimports = collect_all('librosa')
datas += librosa_datas
binaries += librosa_binaries
hiddenimports += librosa_hiddenimports

# Add numpy
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
datas += numpy_datas
binaries += numpy_binaries
hiddenimports += numpy_hiddenimports

# Add other necessary packages
for package in ['pyjson5', 'scipy', 'soundfile', 'audioread', 'resampy', 'numba', 'av', 'tokenizers']:
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
    except:
        pass

# Collect setuptools and pkg_resources data to fix missing modules
try:
    from PyInstaller.utils.hooks import collect_data_files
    setuptools_datas = collect_data_files('setuptools')
    datas += setuptools_datas
    pkg_resources_datas = collect_data_files('pkg_resources')
    datas += pkg_resources_datas
except:
    pass

# Explicitly collect backports module to fix ModuleNotFoundError
try:
    backports_datas, backports_binaries, backports_hiddenimports = collect_all('backports')
    datas += backports_datas
    binaries += backports_binaries
    hiddenimports += backports_hiddenimports
    print("Collected backports module successfully")
except Exception as e:
    print(f"Could not collect backports module: {e}")
    # Try alternative collection method
    try:
        import backports
        import os
        backports_path = os.path.dirname(backports.__file__)
        datas.append((backports_path, 'backports'))
        print(f"Added backports from path: {backports_path}")
    except:
        print("Warning: backports module not found - may need to be installed")

# Add hidden imports for modules that might not be detected automatically
hiddenimports += [
    'ctranslate2',
    'transformers.models',
    'transformers.models.whisper',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'tokenizers',
    'tokenizers.implementations',
    'tokenizers.models',
    'tokenizers.pre_tokenizers',
    'tokenizers.processors',
    'onnxruntime.capi',
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_providers_cuda',  # Important for GPU
    'onnxruntime.capi.onnxruntime_providers_tensorrt',  # TensorRT if available
    'librosa.core',
    'librosa.feature',
    'scipy.special._ufuncs_cxx',
    'scipy.linalg._fblas',
    'scipy.linalg._flapack',
    'scipy.linalg._cythonized_array_utils',
    'scipy.linalg._solve_toeplitz',
    'scipy.linalg._matfuncs_sqrtm_triu',
    'scipy.linalg._decomp_lu_cython',
    'scipy.linalg._matfuncs_expm',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
    'numba.core',
    'numba.cuda',
    'av.audio',
    'av.container',
    'av.stream',
    'pkg_resources.extern',
    'pkg_resources._vendor',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
    'backports',  # Fix for ModuleNotFoundError
    'backports.functools_lru_cache',  # Common backports module
    'setuptools._vendor.jaraco',  # Include jaraco modules
    'setuptools._vendor.jaraco.text',
    'setuptools._vendor.jaraco.context',
    'setuptools._vendor.jaraco.functools',
    'code',  # For interactive console with --console option
    'readline',  # For better console experience (if available)
    'rlcompleter',  # For tab completion in console
]

# Add project data files
# Note: models directory is excluded and handled separately by CI
datas += [
    ('src/faster_whisper_transwithai_chickenrice', 'faster_whisper_transwithai_chickenrice'),
    ('locales', 'locales'),  # Include the locales directory with translations
]

# Base collections (for infer.exe only)
infer_datas = list(datas)
infer_binaries = list(binaries)
infer_hiddenimports = list(hiddenimports)

# Extend collections for modal_infer.exe only
modal_datas = list(datas)
modal_binaries = list(binaries)
modal_hiddenimports = list(hiddenimports)

# Collect modal / questionary and their tricky deps explicitly (modal_infer only)
try:
    m_datas, m_binaries, m_hiddenimports = collect_all('modal')
    modal_datas += m_datas
    modal_binaries += m_binaries
    modal_hiddenimports += m_hiddenimports
    print("Collected modal successfully")
except:
    print("Warning: could not collect modal")

try:
    s_datas, s_binaries, s_hiddenimports = collect_all('synchronicity')
    modal_datas += s_datas
    modal_binaries += s_binaries
    modal_hiddenimports += s_hiddenimports
except:
    print("Warning: could not collect synchronicity")

try:
    q_datas, q_binaries, q_hiddenimports = collect_all('questionary')
    modal_datas += q_datas
    modal_binaries += q_binaries
    modal_hiddenimports += q_hiddenimports
except:
    print("Warning: could not collect questionary")

modal_hiddenimports += [
    'modal',
    'modal.proto',
    'synchronicity',
    'grpclib',
    'google.protobuf',
    'google.protobuf.internal',
    'toml',
    'rich',
    'typer',
    'click',
    'questionary',
    'prompt_toolkit',
    'prompt_toolkit.styles',
    'prompt_toolkit.key_binding',
    'prompt_toolkit.formatted_text',
    'prompt_toolkit.shortcuts',
    'prompt_toolkit.output',
    'prompt_toolkit.input',
]

a_infer = Analysis(
    ['infer.py'],
    pathex=[],
    binaries=infer_binaries,
    datas=infer_datas,
    hiddenimports=infer_hiddenimports,
    hookspath=[],  # PyInstaller hooks contrib should be auto-detected
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],  # Add runtime hook to set KMP_DUPLICATE_LIB_OK
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'notebook',
        'jupyter',
        'IPython',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_modal = Analysis(
    ['modal_infer.py'],
    pathex=[],
    binaries=modal_binaries,
    datas=modal_datas,
    hiddenimports=modal_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'notebook',
        'jupyter',
        'IPython',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz_infer = PYZ(a_infer.pure, a_infer.zipped_data, cipher=block_cipher)
pyz_modal = PYZ(a_modal.pure, a_modal.zipped_data, cipher=block_cipher)

infer_exe = EXE(
    pyz_infer,
    [a_infer.scripts[0]],
    [],
    exclude_binaries=True,
    name='infer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='transwithai.ico' if os.path.exists('transwithai.ico') else None,
)

modal_exe = EXE(
    pyz_modal,
    [a_modal.scripts[0]],
    [],
    exclude_binaries=True,
    name='modal_infer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='transwithai.ico' if os.path.exists('transwithai.ico') else None,
)

coll = COLLECT(
    infer_exe,
    modal_exe,
    a_infer.binaries,
    a_infer.zipfiles,
    a_infer.datas,
    a_modal.binaries,
    a_modal.zipfiles,
    a_modal.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='faster_whisper_transwithai_chickenrice',
)
