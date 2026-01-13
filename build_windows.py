#!/usr/bin/env python3
"""
Build script for creating Windows executable with PyInstaller
Includes CUDA runtime libraries for GPU acceleration
"""

import os
import sys
import subprocess
from pathlib import Path

# Import the download_models functions
try:
    from download_models import (
        download_vad_model,
        download_whisper_base_for_feature_extractor,
        verify_vad_model,
        verify_whisper_base_feature_extractor
    )
except ImportError:
    print("Warning: download_models.py not found, skipping model download")
    download_vad_model = None
    download_whisper_base_for_feature_extractor = None
    verify_vad_model = None
    verify_whisper_base_feature_extractor = None

def find_cuda_libs():
    """Find CUDA libraries needed for CTranslate2 in the conda environment"""
    cuda_libs = []

    # For conda environments, libraries are in the conda env root
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_path = Path(conda_prefix)
    else:
        conda_path = Path(sys.prefix)

    print(f"Searching for CUDA libraries in: {conda_path}")

    # On Windows, conda installs DLLs in different locations
    # Check Library/bin for Windows conda environments
    lib_dirs = [
        conda_path / "Library" / "bin",  # Windows conda location
        conda_path / "bin",               # Alternative location
        conda_path / "lib",               # Linux/Unix style (for reference)
    ]

    # CUDA and cuDNN library patterns for Windows
    lib_patterns = [
        # CUDA runtime
        "cudart64_*.dll",
        # cuBLAS
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        # cuDNN core libraries - these are critical!
        "cudnn64_*.dll",
        "cudnn_ops_infer64_*.dll",
        "cudnn_cnn_infer64_*.dll",
        "cudnn_adv_infer64_*.dll",
        "cudnn_ops_train64_*.dll",
        "cudnn_cnn_train64_*.dll",
        "cudnn_adv_train64_*.dll",
        # cuDNN without version numbers
        "cudnn*.dll",
        # cuFFT
        "cufft64_*.dll",
        # cuRAND
        "curand64_*.dll",
        # Additional CUDA libraries
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll",
    ]

    # Search in conda library directories
    for lib_dir in lib_dirs:
        if lib_dir.exists():
            print(f"  Checking: {lib_dir}")
            for pattern in lib_patterns:
                found = list(lib_dir.glob(pattern))
                if found:
                    print(f"    Found {len(found)} files matching {pattern}")
                    cuda_libs.extend(found)

    # Also check site-packages for pip-installed CUDA libraries
    site_packages_dirs = [
        conda_path / "Lib" / "site-packages",  # Windows
        conda_path / "lib" / "python*" / "site-packages",  # Unix pattern
    ]

    for site_packages_pattern in site_packages_dirs:
        if "*" in str(site_packages_pattern):
            # Handle glob patterns
            for site_packages in conda_path.glob(str(site_packages_pattern.relative_to(conda_path))):
                if site_packages.exists():
                    # Check nvidia packages
                    nvidia_path = site_packages / "nvidia"
                    if nvidia_path.exists():
                        for subdir in nvidia_path.iterdir():
                            if subdir.is_dir():
                                lib_dir = subdir / "bin"
                                if lib_dir.exists():
                                    for pattern in lib_patterns:
                                        cuda_libs.extend(lib_dir.glob(pattern))

                    # Check ctranslate2.libs for bundled libraries
                    ct2_libs = site_packages / "ctranslate2.libs"
                    if ct2_libs.exists():
                        cuda_libs.extend(ct2_libs.glob("*.dll"))
        else:
            if site_packages_pattern.exists():
                # Check nvidia packages
                nvidia_path = site_packages_pattern / "nvidia"
                if nvidia_path.exists():
                    for subdir in nvidia_path.iterdir():
                        if subdir.is_dir():
                            lib_dir = subdir / "bin"
                            if lib_dir.exists():
                                for pattern in lib_patterns:
                                    cuda_libs.extend(lib_dir.glob(pattern))

                # Check ctranslate2.libs for bundled libraries
                ct2_libs = site_packages_pattern / "ctranslate2.libs"
                if ct2_libs.exists():
                    cuda_libs.extend(ct2_libs.glob("*.dll"))

    # Remove duplicates
    cuda_libs = list(set(cuda_libs))
    cuda_libs.sort(key=lambda x: x.name)

    return cuda_libs

def download_models_if_needed():
    """Download models if they don't exist"""
    if verify_vad_model is None:
        print("Warning: Model download not available")
        return True

    print("\n📦 Checking models...")

    # Check VAD model (always required)
    vad_ok = verify_vad_model() if verify_vad_model else False
    # Check whisper-base feature extractor (required for offline usage)
    whisper_base_ok = verify_whisper_base_feature_extractor() if verify_whisper_base_feature_extractor else False

    if vad_ok and whisper_base_ok:
        print("✓ All models present")
        return True

    print("\n⬇ Downloading missing models...")
    success = True

    # Download VAD model if needed
    if not vad_ok and download_vad_model:
        if not download_vad_model():
            print("❌ Failed to download VAD model")
            success = False

    # Download whisper-base for feature extractor if needed
    if not whisper_base_ok and download_whisper_base_for_feature_extractor:
        if not download_whisper_base_for_feature_extractor():
            print("❌ Failed to download whisper-base feature extractor")
            success = False

    # Final verification
    final_vad_ok = verify_vad_model() if verify_vad_model else False
    final_whisper_base_ok = verify_whisper_base_feature_extractor() if verify_whisper_base_feature_extractor else False

    if final_vad_ok and final_whisper_base_ok:
        print("✅ All models ready")
        return True
    else:
        print("❌ Model download failed. Cannot continue without required models.")
        return False

def build():
    """Main build function"""
    print("Starting Windows build with CUDA support...")

    # Check if we're in a virtual environment or conda environment
    in_conda = os.environ.get('CONDA_PREFIX') is not None
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if in_conda:
        print(f"✓ Using conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    elif in_venv:
        print("✓ Using virtual environment")
    else:
        print("Warning: Not in a conda or virtual environment. Make sure dependencies are installed.")

    print("Note: Models are not included in PyInstaller build and will be handled by CI")
    print("CUDA libraries will be included via spec file's binary collection")

    # Use the project.spec directly
    spec_file = Path("project.spec")
    if not spec_file.exists():
        print("Error: project.spec not found!")
        return 1

    # Build command - using the spec file directly
    build_cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "project.spec"
    ]

    print(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, capture_output=False)

    # Verify build succeeded and check for CUDA libraries
    if result.returncode == 0:
        # Build modal_infer if modal.spec is present (separate target).
        modal_spec = Path("modal.spec")
        if modal_spec.exists():
            # Ensure modal dependencies are available in the current env.
            try:
                import modal  # noqa: F401
                import questionary  # noqa: F401
            except ImportError:
                print("\nmodal/questionary not found; installing for modal.spec build...")
                install_cmd = [
                    sys.executable, "-m", "pip", "install",
                    "modal", "questionary",
                ]
                install_result = subprocess.run(install_cmd, capture_output=False)
                if install_result.returncode != 0:
                    print("\nFailed to install modal/questionary.")
                    return 1

            modal_cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                "--distpath", str(Path("dist") / "faster_whisper_transwithai_chickenrice"),
                "--workpath", str(Path("build") / "modal"),
                str(modal_spec),
            ]
            print(f"\nRunning: {' '.join(modal_cmd)}")
            modal_result = subprocess.run(modal_cmd, capture_output=False)
            if modal_result.returncode != 0:
                print("\nModal build failed!")
                return 1

        dist_root = Path("dist")
        dist_dir = dist_root / "faster_whisper_transwithai_chickenrice"
        engine_dir = dist_root / "engine"
        client_dir = dist_root / "client"

        if dist_dir.exists():
            verify_dirs = [dist_dir]
        else:
            verify_dirs = [p for p in [engine_dir, client_dir] if p.exists()]

        if verify_dirs:
            # Quick verification of critical libraries
            print("\nVerifying CUDA libraries in distribution...")

            critical_libs = ['cudnn', 'cublas', 'cudart']
            found_libs = {}
            missing_libs = []

            # Check in root directory and all subdirectories
            all_dlls = []
            for root_dir in verify_dirs:
                all_dlls.extend(root_dir.glob("**/*.dll"))

            for critical in critical_libs:
                found_in_locations = []
                for dll_path in all_dlls:
                    if critical in dll_path.name.lower():
                        # Get relative path from dist root for consistent display
                        try:
                            rel_path = dll_path.relative_to(dist_root)
                            location = str(rel_path.parent) if str(rel_path.parent) != '.' else 'root'
                        except ValueError:
                            location = str(dll_path.parent)
                        found_in_locations.append(location)

                if found_in_locations:
                    # Remove duplicates and store
                    found_libs[critical] = list(set(found_in_locations))
                else:
                    missing_libs.append(critical)

            if found_libs:
                print(f"  ✓ Found critical libraries:")
                for lib, locations in found_libs.items():
                    locations_str = ', '.join(locations)
                    print(f"    - {lib}: {locations_str}")

            if missing_libs:
                print(f"  ⚠ Missing libraries: {', '.join(missing_libs)}")
                print("     Note: The PyInstaller hooks should have included these.")
                print("     If GPU acceleration doesn't work, check your conda environment.")

            if dist_dir.exists():
                output_locations = [str(dist_dir)]
            else:
                output_locations = [str(p) for p in verify_dirs]
            print(f"\nBuild complete! Output in: {', '.join(output_locations)}")
        else:
            print("Error: dist/engine or dist/client directory not found after build")
            return 1
    else:
        print("\nBuild failed!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(build())
