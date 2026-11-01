#!/usr/bin/env python3
"""
Download required model files from Hugging Face repositories
ONNX VAD model is always downloaded, additional models can be specified via HuggingFace repo path
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin

import requests


# Detect if the environment supports Unicode/emoji
def can_use_unicode():
    """Check if the current environment supports Unicode output"""
    # If we're in a CI environment, be conservative and use ASCII
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        # CI environments often have encoding issues, especially on Windows
        return False

    # Check if UTF-8 is explicitly set
    if os.environ.get("PYTHONIOENCODING", "").lower().startswith("utf"):
        return True

    if sys.platform == "win32":
        # Windows console often doesn't support Unicode well
        # Try to enable UTF-8 on Windows
        try:
            # Test if we can encode an emoji
            test_emoji = "✓"
            test_emoji.encode(sys.stdout.encoding or "utf-8")
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    # On other platforms (Linux, Mac), usually Unicode works
    return True


# Define symbols based on Unicode support
USE_UNICODE = can_use_unicode()

if USE_UNICODE:
    # Unicode/emoji symbols
    CHECKMARK = "✓"
    CROSS = "✗"
    DOWNLOAD = "⬇"
    PACKAGE = "📦"
    SEARCH = "🔍"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠"
else:
    # ASCII fallback symbols
    CHECKMARK = "[OK]"
    CROSS = "[X]"
    DOWNLOAD = "[DOWNLOADING]"
    PACKAGE = "[PACKAGE]"
    SEARCH = "[SEARCH]"
    SUCCESS = "[SUCCESS]"
    ERROR = "[ERROR]"
    WARNING = "[WARNING]"

# Force UTF-8 encoding on stdout/stderr if possible
if sys.platform == "win32" and not USE_UNICODE:
    # On Windows CI, try to set UTF-8 mode
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass  # If this fails, we'll use ASCII symbols anyway


DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_MAX_RETRY_DELAY_SECONDS = 30.0

HF_ENDPOINT_DEFAULT = "https://huggingface.co"
HF_ENDPOINT_CN_MIRROR = "https://hf-mirror.com"


def normalize_endpoint(value: str | None) -> str:
    """把用户输入的下载源规整成可用的 endpoint（补协议、去尾部斜杠）。"""
    endpoint = (value or "").strip()
    if not endpoint:
        return HF_ENDPOINT_DEFAULT
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    return endpoint.rstrip("/")


# 当前使用的 HuggingFace 端点：优先级 --endpoint > 环境变量 HF_ENDPOINT > 官方。
# 兼容 hf-mirror.com 等与官方协议一致的镜像；ModelScope 之类协议不同的站点不能直接用。
HF_ENDPOINT = normalize_endpoint(os.environ.get("HF_ENDPOINT"))


def _fallback_url(url: str) -> str | None:
    """返回同一资源的备用源地址：官方 ↔ hf-mirror 之间互相回退。

    若当前使用的是自定义端点（非上面两者），则不做自动回退。
    """
    current_host = HF_ENDPOINT + "/"
    if not url.startswith(current_host):
        return None
    tail = url[len(current_host) :]
    if HF_ENDPOINT == HF_ENDPOINT_DEFAULT:
        return HF_ENDPOINT_CN_MIRROR + "/" + tail
    if HF_ENDPOINT == HF_ENDPOINT_CN_MIRROR:
        return HF_ENDPOINT_DEFAULT + "/" + tail
    return None


def _get_429_retry_delay(response: requests.Response, retry_index: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, IndexError, OverflowError):
                delay = 0.0

        if delay > 0:
            return min(delay, DOWNLOAD_MAX_RETRY_DELAY_SECONDS)

    exponential_delay = DOWNLOAD_RETRY_BACKOFF_SECONDS * (2**retry_index)
    return min(exponential_delay, DOWNLOAD_MAX_RETRY_DELAY_SECONDS)


def _download_from_url(url: str, dest_path: Path, session: requests.Session) -> bool:
    """Download a single file from one URL, retrying on HTTP 429. Returns True on success."""
    try:
        print(f"  {DOWNLOAD} Downloading {dest_path.name}...", end=" ")

        response = None
        for retry_index in range(DOWNLOAD_MAX_RETRIES + 1):
            response = session.get(url, stream=True, timeout=30)
            if response.status_code != 429 or retry_index == DOWNLOAD_MAX_RETRIES:
                response.raise_for_status()
                break

            retry_delay = _get_429_retry_delay(response, retry_index)
            response.close()
            print(
                f"\r  {WARNING} Rate limited while downloading {dest_path.name}; "
                f"retrying in {retry_delay:.1f}s ({retry_index + 1}/{DOWNLOAD_MAX_RETRIES})"
            )
            time.sleep(retry_delay)
            print(f"  {DOWNLOAD} Downloading {dest_path.name}...", end=" ")

        if response is None:
            raise RuntimeError("Download did not start")

        # Get file size
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress
        downloaded = 0
        last_reported_progress = -1
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        # Only update display every 10%
                        progress_milestone = int(progress // 10) * 10
                        if progress_milestone > last_reported_progress:
                            print(f"\r  {DOWNLOAD} Downloading {dest_path.name}... {progress_milestone}%", end="")
                            last_reported_progress = progress_milestone

        print(f"\r  {CHECKMARK} Downloaded {dest_path.name} ({downloaded / (1024 * 1024):.1f} MB)")
        return True

    except Exception as e:
        print(f"\r  {CROSS} Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_file(url: str, dest_path: Path, session: requests.Session | None = None) -> bool:
    """Download a file with progress indicator.

    Tries the given URL first. If it is a huggingface.co URL and the download
    fails (including after exhausting the HTTP 429 retries), the whole download
    is retried against the hf-mirror.com mirror for another round of attempts.
    """
    if session is None:
        session = requests.Session()

    # Create parent directory if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if file already exists
    if dest_path.exists():
        print(f"  {CHECKMARK} {dest_path.name} already exists")
        return True

    # Try the primary endpoint first, then fall back to the paired mirror
    # (official -> hf-mirror, or hf-mirror -> official).
    candidate_urls = [url]
    fallback_url = _fallback_url(url)
    if fallback_url:
        candidate_urls.append(fallback_url)

    for attempt_index, candidate_url in enumerate(candidate_urls):
        if attempt_index > 0:
            print(f"  {WARNING} Retrying {dest_path.name} from fallback source: {fallback_url}")
        if _download_from_url(candidate_url, dest_path, session):
            return True

    return False


def get_hf_api_files(repo_id: str) -> list[str]:
    """Get list of files from Hugging Face repo using API"""
    api_url = f"{HF_ENDPOINT}/api/models/{repo_id}/tree/main"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        files = response.json()
        return [f["path"] for f in files if f["type"] == "file"]
    except Exception as e:
        print(f"Warning: Could not fetch file list from API: {e}")
        return []


def download_hf_model(repo_id: str, target_dir: str | None = None):
    """Download model files from any HuggingFace repository"""
    base_url = f"{HF_ENDPOINT}/{repo_id}/resolve/main/"

    # Determine target directory
    if target_dir:
        models_dir = Path("models") / target_dir
    else:
        # Use repository name as default subdirectory
        repo_name = repo_id.split("/")[-1]
        models_dir = Path("models") / repo_name

    print(f"\n{PACKAGE} Downloading model from {repo_id}")
    print(f"   Target directory: {models_dir}")

    # Essential file extensions to download for transformer/whisper models
    essential_extensions = [".json", ".bin", ".txt", ".onnx", ".safetensors", ".model"]

    # Try to get full file list from API
    api_files = get_hf_api_files(repo_id)
    if api_files:
        # Filter for essential files
        files_to_download = [f for f in api_files if any(f.endswith(ext) for ext in essential_extensions)]
        print(f"  Found {len(files_to_download)} files in repository")
    else:
        # If API fails, try common file names
        files_to_download = [
            "config.json",
            "model.bin",
            "pytorch_model.bin",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocabulary.json",
            "vocab.json",
            "special_tokens_map.json",
            "merges.txt",
        ]
        print("  Using common file list (API unavailable)")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    success_count = 0
    for filename in files_to_download:
        url = urljoin(base_url, filename)
        dest_path = models_dir / filename
        if download_file(url, dest_path, session):
            success_count += 1

    print(f"  {CHECKMARK} Downloaded {success_count}/{len(files_to_download)} files")
    return success_count > 0


def download_vad_model():
    """Download VAD ONNX model files (always required)"""
    repo_id = "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
    base_url = f"{HF_ENDPOINT}/{repo_id}/resolve/main/"
    models_dir = Path("models")

    print(f"\n{PACKAGE} Downloading VAD ONNX model from {repo_id}")

    # Files to download (renamed to match existing structure)
    files = [
        ("model.onnx", "whisper_vad.onnx"),  # Download as model.onnx, save as whisper_vad.onnx
        (
            "model_metadata.json",
            "whisper_vad_metadata.json",
        ),  # Download as model_metadata.json, save as whisper_vad_metadata.json
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    success_count = 0
    for source_name, dest_name in files:
        url = urljoin(base_url, source_name)
        dest_path = models_dir / dest_name
        if download_file(url, dest_path, session):
            success_count += 1

    print(f"  {CHECKMARK} Downloaded {success_count}/{len(files)} files")
    return success_count == len(files)


def download_whisper_base_for_feature_extractor():
    """Download whisper-base model files specifically for feature extractor (offline usage)"""
    repo_id = "openai/whisper-base"
    models_dir = Path("models") / "whisper-base"
    base_url = f"{HF_ENDPOINT}/{repo_id}/resolve/main/"

    print(f"\n{PACKAGE} Downloading whisper-base for feature extractor (offline usage)")

    # Check if files already exist from main models folder
    existing_models_dir = Path("models")
    if existing_models_dir.exists():
        # Files we can copy from existing models folder if available
        files_to_copy = [
            "preprocessor_config.json",
            "config.json",
            "tokenizer.json",
            "vocab.json",
        ]

        copied = 0
        models_dir.mkdir(parents=True, exist_ok=True)
        for filename in files_to_copy:
            src = existing_models_dir / filename
            dest = models_dir / filename
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)
                print(f"  {CHECKMARK} Copied {filename} from existing models folder")
                copied += 1
            elif dest.exists():
                print(f"  {CHECKMARK} {filename} already exists")
                copied += 1

        if copied >= 2:  # At minimum we need preprocessor_config.json and config.json
            print(f"  {CHECKMARK} Used existing files for whisper-base")
            return True

    # Download ONLY the specific files needed for feature extractor
    # We don't need model weights (.bin, .safetensors) for feature extraction
    required_files = [
        "preprocessor_config.json",  # Required for feature extractor
        "config.json",  # Required for configuration
        "tokenizer.json",  # Optional but useful for tokenization
        "vocab.json",  # Optional but useful for vocabulary
    ]

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading feature extractor files from {repo_id}...")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    success_count = 0
    for filename in required_files:
        url = urljoin(base_url, filename)
        dest_path = models_dir / filename
        if download_file(url, dest_path, session):
            success_count += 1

    print(f"  {CHECKMARK} Downloaded {success_count}/{len(required_files)} feature extractor files")
    return success_count >= 2  # At minimum we need the two required files


def verify_whisper_base_feature_extractor():
    """Verify that whisper-base feature extractor files exist"""
    models_dir = Path("models") / "whisper-base"

    required_files = [
        ("preprocessor_config.json", "Feature extractor config"),
        ("config.json", "Model configuration"),
    ]

    optional_files = [
        ("tokenizer.json", "Tokenizer"),
        ("vocab.json", "Vocabulary"),
    ]

    if not models_dir.exists():
        return False

    print(f"\n{SEARCH} Verifying whisper-base feature extractor files...")
    all_required_present = True

    for filename, description in required_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  {CHECKMARK} {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  {CROSS} {filename} missing - {description}")
            all_required_present = False

    for filename, _description in optional_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  {CHECKMARK} {filename} ({size_kb:.1f} KB) - optional")

    return all_required_present


def verify_vad_model():
    """Verify that required VAD model files exist"""
    models_dir = Path("models")

    required_files = [
        ("whisper_vad.onnx", "VAD ONNX model"),
        ("whisper_vad_metadata.json", "VAD metadata"),
    ]

    print(f"\n{SEARCH} Verifying VAD model files...")
    all_present = True

    for filename, description in required_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {CHECKMARK} {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  {CROSS} {filename} missing - {description}")
            all_present = False

    return all_present


def verify_hf_model(repo_id: str, target_dir: str | None = None):
    """Verify that HuggingFace model files exist"""
    if target_dir:
        models_dir = Path("models") / target_dir
    else:
        repo_name = repo_id.split("/")[-1]
        models_dir = Path("models") / repo_name

    if not models_dir.exists():
        print(f"\n{WARNING} Model directory {models_dir} does not exist")
        return False

    print(f"\n{SEARCH} Verifying model files in {models_dir}...")

    # Check for common model files
    common_files = ["config.json", "model.bin", "pytorch_model.bin", "model.safetensors", "model.onnx"]
    found_files = []

    for file in models_dir.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {CHECKMARK} {file.name} ({size_mb:.1f} MB)")
            found_files.append(file.name)

    # Check if at least one model file exists
    has_model = any(f in found_files for f in common_files)

    if not has_model and found_files:
        print(f"  {WARNING} Warning: No common model files found, but other files exist")
    elif not found_files:
        print(f"  {CROSS} No files found in model directory")
        return False

    return True


def main():
    """Main download function"""
    parser = argparse.ArgumentParser(
        description="Model Downloader for Faster Whisper Custom VAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
    # Download VAD model and whisper-base (both required for offline usage)

  %(prog)s --skip-whisper-base
    # Download only VAD model, skip whisper-base (not recommended)

  %(prog)s --hf-model chickenrice0721/whisper-large-v2-translate-zh-v0.2-st-ct2
    # Download VAD, whisper-base, and Chickenrice Whisper model

  %(prog)s --hf-model TransWithAI/whisper-ja-1.5B-ct2
    # Download VAD, whisper-base, and TransWithAI Japanese transcribe bf16 model

  %(prog)s --hf-model TransWithAI/whisper-ja-1.5B-ct2 --target-dir .
    # Download the main model directly into models/ for the default .bat launchers

  %(prog)s --hf-model openai/whisper-large-v3 --target-dir whisper-v3
    # Download VAD, whisper-base, and Whisper v3 to specific directory

  %(prog)s --force --hf-model myusername/my-custom-model
    # Force re-download everything including VAD, whisper-base, and custom model
        """,
    )

    parser.add_argument(
        "--hf-model", type=str, help='HuggingFace repository path to download (e.g., "openai/whisper-large-v3")'
    )

    parser.add_argument(
        "--target-dir",
        type=str,
        help="Target subdirectory name in models/ for the HuggingFace model (defaults to repo name)",
    )

    parser.add_argument("--force", action="store_true", help="Force re-download even if models already exist")

    parser.add_argument(
        "--endpoint",
        type=str,
        help=(
            "HuggingFace-compatible endpoint. Official: https://huggingface.co ; "
            "China mirror (recommended for CN network): https://hf-mirror.com ; "
            "defaults to $HF_ENDPOINT or the official site."
        ),
    )

    parser.add_argument(
        "--skip-vad", action="store_true", help="Skip downloading VAD model (not recommended, for testing only)"
    )

    parser.add_argument(
        "--skip-whisper-base",
        action="store_true",
        help="Skip downloading whisper-base model for feature extractor (not recommended)",
    )

    args = parser.parse_args()

    # --endpoint 优先级高于环境变量 HF_ENDPOINT
    global HF_ENDPOINT
    if args.endpoint:
        HF_ENDPOINT = normalize_endpoint(args.endpoint)

    print("=" * 60)
    print("Model Downloader for Faster Whisper Custom VAD")
    print("=" * 60)
    print(f"Download source: {HF_ENDPOINT}")
    fallback_preview = _fallback_url(HF_ENDPOINT + "/x")
    if fallback_preview:
        print(f"Fallback source: {fallback_preview.rsplit('/x', 1)[0]}")
    else:
        print("Fallback source: (none, custom endpoint)")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Check if VAD model already exists
    if not args.force and not args.skip_vad and verify_vad_model():
        print(f"\n{CHECKMARK} VAD model files already present")
        vad_exists = True
    else:
        vad_exists = False

    # Check if whisper-base feature extractor already exists
    whisper_base_exists = False
    if not args.skip_whisper_base and not args.force and verify_whisper_base_feature_extractor():
        print(f"\n{CHECKMARK} Whisper-base feature extractor files already present")
        whisper_base_exists = True

    # Check if HF model already exists (if specified)
    hf_exists = False
    if args.hf_model and not args.force and verify_hf_model(args.hf_model, args.target_dir):
        print(f"\n{CHECKMARK} Model {args.hf_model} already present")
        hf_exists = True

    # If everything exists and no force flag, ask user
    all_exists = vad_exists and (not args.hf_model or hf_exists) and (args.skip_whisper_base or whisper_base_exists)
    if all_exists and not args.force:
        response = input("\nAll required models are present. Re-download? (y/N): ").strip().lower()
        if response != "y":
            print("Skipping download.")
            return 0

    # Download models

    # Always download VAD model (unless explicitly skipped)
    if not args.skip_vad:
        if not download_vad_model():
            print(f"{WARNING} Error: VAD model is required and could not be downloaded")
    else:
        print(f"\n{WARNING} Skipping VAD model download (not recommended)")

    # Download whisper-base feature extractor (unless explicitly skipped)
    if not args.skip_whisper_base:
        if not download_whisper_base_for_feature_extractor():
            print(f"{WARNING} Warning: Whisper-base feature extractor could not be downloaded completely")
            # Don't fail completely if feature extractor download has issues
    else:
        print(f"\n{WARNING} Skipping whisper-base download (not recommended for offline usage)")

    # Download HuggingFace model if specified
    if args.hf_model and not download_hf_model(args.hf_model, args.target_dir):
        print(f"{WARNING} Warning: Model {args.hf_model} could not be downloaded completely")
        # Don't fail completely if HF model download has issues

    # Final verification
    print("\n" + "=" * 60)

    # Verify VAD model
    if not args.skip_vad:
        if verify_vad_model():
            print(f"\n{SUCCESS} VAD model downloaded successfully!")
        else:
            print(f"\n{ERROR} Critical: VAD model is missing. Cannot proceed without it.")
            return 1

    # Verify whisper-base feature extractor (unless skipped)
    if not args.skip_whisper_base:
        if verify_whisper_base_feature_extractor():
            print(f"\n{SUCCESS} Whisper-base feature extractor downloaded successfully!")
        else:
            print(f"\n{WARNING} Warning: Some whisper-base feature extractor files may be missing.")

    # Verify HF model if specified
    if args.hf_model:
        if verify_hf_model(args.hf_model, args.target_dir):
            print(f"\n{SUCCESS} Model {args.hf_model} downloaded successfully!")
        else:
            print(f"\n{WARNING} Warning: Some files from {args.hf_model} may be missing.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{ERROR} Error: {e}")
        sys.exit(1)
