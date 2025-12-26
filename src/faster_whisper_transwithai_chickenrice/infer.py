#!/usr/bin/env python3
"""
Inference script with custom VAD injection support
"""

import argparse
import sys
import logging
import os
import json
import code
import platform
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from collections import ChainMap
from typing import Optional, Dict, Any

import pyjson5
from faster_whisper import WhisperModel, BatchedInferencePipeline
import ctranslate2

# Import our VAD injection system
from . import inject_vad, uninject_vad, VadOptionsCompat
from .vad_manager import VadConfig

# Import modern i18n module for translations
from . import i18n_modern as i18n

# Convenience imports
_ = i18n._
format_duration = i18n.format_duration
format_percentage = i18n.format_percentage


def parse_arguments():
    parser = argparse.ArgumentParser(description=_("app.description"))
    parser.add_argument('--model_name_or_path', type=str, default="models",
                       help=_("args.model_path"))
    parser.add_argument('--device', type=str, default='auto',
                       help=_("args.device"))
    parser.add_argument('--compute_type', type=str, default='auto',
                       help=_("args.compute_type"))
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help=_("args.overwrite"))
    parser.add_argument('--audio_suffixes', type=str, default="wav,flac,mp3",
                       help=_("args.audio_extensions"))
    parser.add_argument('--sub_formats', type=str, default="lrc,vtt",
                       help=_("args.subtitle_formats"))
    parser.add_argument('--output_dir', type=str, default=None,
                       help=_("args.output_dir"))
    parser.add_argument('--generation_config', type=str, default="generation_config.json5",
                       help=_("args.config_file"))
    parser.add_argument('--log_level', type=str, default="DEBUG",
                       help=_("args.log_level"))

    # Subtitle post-processing options
    parser.add_argument('--merge_segments', dest='merge_segments', action='store_true', default=None,
                       help="Enable segment merge post-processing (override config file)")
    parser.add_argument('--no_merge_segments', dest='merge_segments', action='store_false', default=None,
                       help="Disable segment merge post-processing (override config file)")
    parser.add_argument('--merge_max_gap_ms', type=int, default=None,
                       help="Max allowed gap (ms) between segments for merging (override config file)")
    parser.add_argument('--merge_max_duration_ms', type=int, default=None,
                       help="Max duration (ms) of a merged segment (override config file)")

    # VAD parameter overrides (whisper_vad is always used)
    parser.add_argument('--vad_threshold', type=float, default=None,
                       help=_("args.vad_threshold"))
    parser.add_argument('--vad_min_speech_duration_ms', type=int, default=None,
                       help=_("args.min_speech_duration"))
    parser.add_argument('--vad_min_silence_duration_ms', type=int, default=None,
                       help=_("args.min_silence_duration"))
    parser.add_argument('--vad_speech_pad_ms', type=int, default=None,
                       help=_("args.speech_padding"))

    # Debug option for interactive console
    parser.add_argument('--console', action='store_true',
                       help="Launch interactive Python console for debugging")

    # Batch inference options
    parser.add_argument('--enable_batching', action='store_true',
                       help="Enable batched inference for faster processing (requires more VRAM)")
    parser.add_argument('--batch_size', type=int, default=None,
                       help="Batch size for batched inference (auto-detect if not specified)")
    parser.add_argument('--max_batch_size', type=int, default=8,
                       help="Maximum batch size to try when auto-detecting (default: 8)")

    parser.add_argument('base_dirs', nargs=argparse.REMAINDER,
                       help=_("args.directories"))
    return parser.parse_args()


def select_best_compute_type(device: str) -> str:
    """
    Automatically select the best compute type based on device and available types.

    Preference order:
    - bfloat16 > float16 > int8 types > float32
    - Prefer int8 over float32 for better memory usage

    Args:
        device: The device to use ('cpu', 'cuda', or 'auto')

    Returns:
        The best available compute type for the device
    """
    # Determine the actual device if 'auto' is specified
    actual_device = device
    if device == 'auto':
        # Check if CUDA devices are actually available
        # First check CUDA_VISIBLE_DEVICES environment variable
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        if cuda_visible == '':
            # Empty string means CUDA is explicitly disabled
            actual_device = 'cpu'
        elif cuda_visible == '-1':
            # -1 also means CUDA is disabled
            actual_device = 'cpu'
        else:
            # Try to check if CUDA is actually available by attempting to get its compute types
            # and checking if we can actually use it
            try:
                # Try to get CUDA compute types
                cuda_types = ctranslate2.get_supported_compute_types('cuda')
                # Also check if we can import and use faster_whisper with CUDA
                # This is a more reliable check
                from faster_whisper import WhisperModel
                # Try to get default device - if CUDA not available, this should fail
                # Note: We're not actually loading a model, just checking device availability
                if cuda_visible is not None:
                    # CUDA_VISIBLE_DEVICES is set to specific devices
                    # Make sure at least one device is visible
                    visible_devices = [d.strip() for d in cuda_visible.split(',') if d.strip()]
                    if not visible_devices:
                        actual_device = 'cpu'
                    else:
                        actual_device = 'cuda'
                else:
                    # CUDA_VISIBLE_DEVICES not set, CUDA should be available if drivers installed
                    actual_device = 'cuda'
            except Exception as e:
                # If we can't get CUDA types or import fails, fall back to CPU
                actual_device = 'cpu'
        logger.info(_("info.auto_detected_device").format(device=actual_device))

    # Get supported compute types for the device
    try:
        supported_types = ctranslate2.get_supported_compute_types(actual_device)
    except Exception as e:
        logger.warning(_("warnings.compute_types_unavailable").format(device=actual_device, error=e))
        # Fallback to safe default
        return 'int8' if actual_device == 'cpu' else 'float16'

    # Define preference order
    # Prefer bfloat16 > float16 > int8 types > float32
    preference_order = [
        'bfloat16',
        'float16',
        'int16',  # For CPU
        'int8_bfloat16',
        'int8_float16',
        'int8_float32',
        'int8',
        'float32'  # Least preferred due to memory usage
    ]

    # Select the best available type based on preference
    for compute_type in preference_order:
        if compute_type in supported_types:
            logger.info(_("info.auto_selected_compute_type").format(compute_type=compute_type, device=actual_device))
            return compute_type

    # If nothing matched (shouldn't happen), use a safe default
    default = 'int8' if actual_device == 'cpu' else 'float16'
    logger.warning(_("warnings.no_preferred_compute_type").format(default=default))
    return default


@dataclass
class Segment:
    start: int  # ms
    end: int    # ms
    text: str


@dataclass(frozen=True)
class SegmentMergeOptions:
    enabled: bool = True
    max_gap_ms: int = 2_000
    max_duration_ms: int = 20_000


def _normalize_merge_text(text: str) -> str:
    return " ".join(text.strip().split())


def merge_segments(segments: list[Segment], options: SegmentMergeOptions | None = None) -> list[Segment]:
    if options is None:
        options = SegmentMergeOptions()

    segments = [s for s in segments if s.text.strip()]
    segments.sort(key=lambda s: (s.start, s.end))
    if not options.enabled:
        return segments

    merged: list[Segment] = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        gap_ms = seg.start - last.end
        if gap_ms > options.max_gap_ms:
            merged.append(seg)
            continue

        merged_duration_ms = seg.end - last.start
        if merged_duration_ms > options.max_duration_ms:
            merged.append(seg)
            continue

        last_norm = _normalize_merge_text(last.text)
        seg_norm = _normalize_merge_text(seg.text)

        if seg_norm.startswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        if last_norm.startswith(seg_norm) or last_norm.endswith(seg_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=last.text)
            continue

        if seg_norm.endswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        merged.append(seg)

    return merged


class SubWriter:
    @classmethod
    def txt(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            lines.append(f"{segment.text}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def lrc(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            start_ts = cls.lrc_timestamp(segment.start)
            end_es = cls.lrc_timestamp(segment.end)
            lines.append(f"[{start_ts}]{segment.text}\n")
            if idx != len(segments) - 1:
                next_start = segments[idx + 1].start
                if next_start is not None and end_es == cls.lrc_timestamp(next_start):
                    continue
            lines.append(f"[{end_es}]\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def lrc_timestamp(ms: int) -> str:
        m = ms // 60_000
        ms = ms - m * 60_000
        s = ms // 1_000
        ms = ms - s * 1_000
        ms = ms // 10
        return f"{m:02d}:{s:02d}.{ms:02d}"

    @classmethod
    def vtt(cls, segments: list[Segment], path: str):
        lines = ["WebVTT\n\n"]
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.vtt_timestamp(segment.start)} --> {cls.vtt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def vtt_timestamp(cls, ms: int):
        return cls._timestamp(ms, '.')

    @classmethod
    def srt(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.srt_timestamp(segment.start)} --> {cls.srt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def srt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ',')

    @classmethod
    def _timestamp(cls, ms: int, delim: str):
        h = ms // 3600_000
        ms -= h * 3600_000
        m = ms // 60_000
        ms -= m * 60_000
        s = ms // 1_000
        ms -= s * 1_000
        return (
            f"{h:02d}:{m:02d}:{s:02d}{delim}{ms:03d}"
        )


@dataclass
class InferenceTask:
    audio_path: str
    sub_prefix: str
    sub_formats: list[str]


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(log_handler)


class Inference:
    sub_writers = {"lrc": SubWriter.lrc, "srt": SubWriter.srt, "vtt": SubWriter.vtt, "txt": SubWriter.txt}

    def __init__(self, args):
        self.args = args
        self.model_name_or_path = args.model_name_or_path
        self.device = args.device
        # Auto-select compute type if 'auto' or 'default' is specified
        if args.compute_type in ['auto', 'default']:
            self.compute_type = select_best_compute_type(self.device)
        else:
            self.compute_type = args.compute_type

        # Batch inference settings
        self.enable_batching = args.enable_batching
        self.batch_size = args.batch_size if args.batch_size else 0
        self.max_batch_size = args.max_batch_size

        self.overwrite = args.overwrite
        self.output_dir = args.output_dir
        if self.output_dir:
            if not os.path.isabs(self.output_dir):
                self.output_dir = os.path.join(os.getcwd(), self.output_dir)
            logger.info(_("info.output_dir", output_dir=self.output_dir))
        self.audio_suffixes = {k: True for k in args.audio_suffixes.split(',')}
        self.sub_formats = []
        for k in args.sub_formats.split(','):
            if k not in self.sub_writers:
                raise ValueError(_("warnings.unknown_format", format=k))
            self.sub_formats.append(k)

        # Load generation config
        self.generation_config, self.segment_merge_options = self._load_generation_config(args)

        # Setup VAD injection if requested
        self._setup_vad_injection(args)

        logger.info(_("info.generation_config", config=self.generation_config))
        logger.info(
            "Segment merge: enabled=%s, max_gap_ms=%s, max_duration_ms=%s",
            self.segment_merge_options.enabled,
            self.segment_merge_options.max_gap_ms,
            self.segment_merge_options.max_duration_ms,
        )

    def _load_generation_config(self, args) -> tuple[Dict[str, Any], SegmentMergeOptions]:
        """Load and process generation configuration"""
        # Default config
        config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
        }

        segment_merge_options = SegmentMergeOptions()

        # Load from file if exists
        if os.path.exists(args.generation_config):
            with open(args.generation_config, "r", encoding='utf-8') as f:
                file_config = pyjson5.decode_io(f)
                file_segment_merge = file_config.pop("segment_merge", None)
                if isinstance(file_segment_merge, dict):
                    segment_merge_options = SegmentMergeOptions(
                        enabled=bool(file_segment_merge.get("enabled", segment_merge_options.enabled)),
                        max_gap_ms=int(file_segment_merge.get("max_gap_ms", segment_merge_options.max_gap_ms)),
                        max_duration_ms=int(
                            file_segment_merge.get("max_duration_ms", segment_merge_options.max_duration_ms)
                        ),
                    )
                config = dict(**ChainMap(file_config, config))

        # Process VAD parameters from config file
        if "vad_parameters" in config:
            vad_params = config.pop("vad_parameters")

            # Convert to VadOptions format
            vad_options = {}

            # Map common parameters
            if "threshold" in vad_params:
                vad_options["threshold"] = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_options["neg_threshold"] = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_options["min_speech_duration_ms"] = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_options["max_speech_duration_s"] = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_options["min_silence_duration_ms"] = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_options["speech_pad_ms"] = vad_params["speech_pad_ms"]

            config["vad_parameters"] = vad_options

        # Override with command line arguments
        if args.vad_threshold is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["threshold"] = args.vad_threshold

        if args.vad_min_speech_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_speech_duration_ms"] = args.vad_min_speech_duration_ms

        if args.vad_min_silence_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_silence_duration_ms"] = args.vad_min_silence_duration_ms

        if args.vad_speech_pad_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["speech_pad_ms"] = args.vad_speech_pad_ms

        # Override segment merge options with command line arguments
        segment_merge_options = SegmentMergeOptions(
            enabled=args.merge_segments if args.merge_segments is not None else segment_merge_options.enabled,
            max_gap_ms=args.merge_max_gap_ms if args.merge_max_gap_ms is not None else segment_merge_options.max_gap_ms,
            max_duration_ms=(
                args.merge_max_duration_ms
                if args.merge_max_duration_ms is not None
                else segment_merge_options.max_duration_ms
            ),
        )

        return config, segment_merge_options

    def _vad_progress_callback(self, chunk_idx, total_chunks, device):
        """Progress callback for VAD processing."""
        progress_pct = (chunk_idx / total_chunks) * 100
        # Use carriage return to update the same line
        print("\r  " + _("progress.vad", current=chunk_idx, total=total_chunks,
            percent=progress_pct, device=device), end="", flush=True)
        if chunk_idx == total_chunks:
            print()  # New line when done

    def _setup_vad_injection(self, args):
        """Setup whisper_vad injection - always enforced"""
        # Always use whisper_vad model
        vad_model = "whisper_vad"

        logger.info(_("info.initializing_vad"))

        # Create VAD config with progress callback
        vad_config = VadConfig(default_model=vad_model)

        # Apply VAD parameters from generation config
        if "vad_parameters" in self.generation_config:
            vad_params = self.generation_config["vad_parameters"]
            if "threshold" in vad_params:
                vad_config.threshold = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_config.neg_threshold = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_config.min_speech_duration_ms = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_config.max_speech_duration_s = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_config.min_silence_duration_ms = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_config.speech_pad_ms = vad_params["speech_pad_ms"]

        # Load ONNX VAD configuration from metadata
        vad_metadata_path = "models/whisper_vad_metadata.json"
        vad_config.onnx_model_path = "models/whisper_vad.onnx"
        vad_config.onnx_metadata_path = vad_metadata_path

        # Read model configuration from metadata JSON if it exists
        if os.path.exists(vad_metadata_path):
            try:
                with open(vad_metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Load model configuration from metadata
                vad_config.whisper_model_name = metadata.get("whisper_model_name", "openai/whisper-base")
                vad_config.frame_duration_ms = metadata.get("frame_duration_ms", 20)
                vad_config.chunk_duration_ms = metadata.get("total_duration_ms", 30000)

                logger.info(_("warnings.loaded_vad_config", path=vad_metadata_path))
            except Exception as e:
                logger.warning(_("warnings.failed_load_vad", path=vad_metadata_path, error=e))
                logger.warning(_("warnings.using_default_vad"))
                # Fallback to defaults
                vad_config.whisper_model_name = "openai/whisper-base"
                vad_config.frame_duration_ms = 20
                vad_config.chunk_duration_ms = 30000
        else:
            # Use defaults if metadata file doesn't exist
            logger.warning(_("warnings.vad_file_not_found", path=vad_metadata_path))
            logger.warning(_("warnings.using_default_vad"))
            vad_config.whisper_model_name = "openai/whisper-base"
            vad_config.frame_duration_ms = 20
            vad_config.chunk_duration_ms = 30000

        # Hardcoded runtime configuration
        vad_config.force_cpu = False
        vad_config.num_threads = 8

        # Inject VAD with progress callback
        inject_vad(model_id=vad_model, config=vad_config, progress_callback=self._vad_progress_callback)
        self.vad_injected = True
        logger.info(_("info.vad_activated", threshold=vad_config.threshold))

    def generates(self, base_dirs):
        if len(base_dirs) == 0:
            logger.warning(_("warnings.provide_directories"))
            return

        tasks = self._scan(base_dirs)
        if len(tasks) == 0:
            logger.info(_("info.no_files_found"))
            return

        logger.info(_("tasks.translation", count=len(tasks)))
        logger.info(_("info.loading_whisper"))

        try:
            model = WhisperModel(self.model_name_or_path, device=self.device, compute_type=self.compute_type)
            logger.info(_("info.model_precision").format(precision=self.compute_type, device=self.device))

            # Setup batched inference if enabled
            batched_model = None
            batch_size_to_use = self.batch_size

            if self.enable_batching:
                try:
                    batched_model = BatchedInferencePipeline(model=model)

                    # Auto-detect batch size if not specified
                    if batch_size_to_use == 0 and len(tasks) > 0:
                        # Use the first audio file as sample for testing
                        batch_size_to_use = self._find_executable_batch_size(
                            model,
                            tasks[0].audio_path,
                            min_batch_size=1,
                            max_batch_size=self.max_batch_size
                        )

                        if batch_size_to_use == 0:
                            logger.warning("Could not find suitable batch size. Falling back to non-batched mode.")
                            batched_model = None

                    if batched_model and batch_size_to_use > 0:
                        logger.info(f"Using batched inference with batch size: {batch_size_to_use}")

                except Exception as e:
                    logger.warning(f"Failed to setup batched inference: {str(e)}. Falling back to non-batched mode.")
                    batched_model = None

            for i, task in enumerate(tasks):
                logger.info(_("info.translating", current=i + 1, total=len(tasks), path=task.audio_path))

                # Use batched or regular inference
                if batched_model and batch_size_to_use > 0:
                    # Use auto-retry with batch size reduction on OOM
                    # This mimics HuggingFace Accelerate's find_executable_batch_size behavior
                    try:
                        _segments, info, actual_batch_size = self._transcribe_with_auto_batch_size(
                            batched_model,
                            task.audio_path,
                            starting_batch_size=batch_size_to_use
                        )
                        # Update batch_size_to_use if it was auto-adjusted
                        if actual_batch_size < batch_size_to_use:
                            logger.info(f"Batch size auto-adjusted from {batch_size_to_use} to {actual_batch_size}")
                            batch_size_to_use = actual_batch_size
                    except Exception as e:
                        logger.warning(f"Batched inference failed: {str(e)}. Falling back to non-batched mode.")
                        # Fallback to non-batched
                        _segments, info = model.transcribe(
                            task.audio_path,
                            **self.generation_config,
                        )
                else:
                    _segments, info = model.transcribe(
                        task.audio_path,
                        **self.generation_config,
                    )

                if info.duration == info.duration_after_vad or info.duration_after_vad == 0:
                    logger.info(_("info.duration", duration=format_duration(info.duration)))
                else:
                    rate = info.duration_after_vad / info.duration
                    logger.info(_("info.duration_filtered",
                        original=format_duration(info.duration),
                        filtered=format_duration(info.duration_after_vad),
                        percent=format_percentage(rate)))

                segments = []
                for _segment in _segments:
                    segment = Segment(
                        start=int(_segment.start*1_000),
                        end=int(_segment.end*1_000),
                        text=_segment.text.strip(),
                    )
                    segments.append(segment)
                    logger.debug(f"[{SubWriter.lrc_timestamp(segment.start)} --> "
                               f"{SubWriter.lrc_timestamp(segment.end)}] {segment.text}")

                segments = merge_segments(segments, self.segment_merge_options)
                os.makedirs(os.path.dirname(task.sub_prefix), exist_ok=True)
                for sub_suffix in task.sub_formats:
                    sub_path = f"{task.sub_prefix}.{sub_suffix}"
                    logger.info(_("info.writing", path=sub_path))
                    self.sub_writers[sub_suffix](segments, sub_path)

        finally:
            # Clean up VAD injection
            if self.vad_injected:
                uninject_vad()
                logger.info(_("info.vad_deactivated"))

    def _find_executable_batch_size(self, model, sample_audio_path, min_batch_size=1, max_batch_size=64):
        """
        Find the maximum executable batch size for batched inference.
        Starts from max_batch_size and works down exponentially on OOM.

        Args:
            model: WhisperModel instance
            sample_audio_path: Path to a sample audio file for testing
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try

        Returns:
            Optimal batch size that fits in memory
        """
        if not self.enable_batching:
            return 0

        logger.info(_("batch.finding_optimal", min_size=min_batch_size, max_size=max_batch_size))

        # Start from max and work down on failure (like HuggingFace Accelerate)
        current_batch_size = max_batch_size

        while current_batch_size >= min_batch_size:
            try:
                logger.info(_("batch.testing_size", size=current_batch_size))

                # Try to create batched pipeline with this batch size
                batched_model = BatchedInferencePipeline(model=model)

                # Test transcription with this batch size
                # Note: batch_size is passed separately to BatchedInferencePipeline.transcribe()
                # It's NOT part of generation_config
                segments, info = batched_model.transcribe(
                    sample_audio_path,
                    batch_size=current_batch_size,  # batch_size is a separate parameter
                    **self.generation_config  # generation_config doesn't include batch_size
                )

                # Force evaluation by converting to list
                list(segments)

                # Success! This batch size works
                logger.info(_("batch.size_successful", size=current_batch_size))
                logger.info(_("batch.optimal_found", size=current_batch_size))
                return current_batch_size

            except RuntimeError as e:
                # If OOM, reduce batch size exponentially
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    logger.warning(_("batch.oom_error", size=current_batch_size))
                else:
                    logger.warning(_("batch.runtime_error", size=current_batch_size, error=error_msg))

                # Reduce batch size by half (exponential backoff)
                new_batch_size = current_batch_size // 2

                # Ensure we reduce by at least 1
                if new_batch_size == current_batch_size:
                    new_batch_size = current_batch_size - 1

                if new_batch_size < min_batch_size:
                    logger.error(_("batch.no_suitable_size", min_size=min_batch_size))
                    return 0

                logger.info(_("batch.reducing_size", old_size=current_batch_size, new_size=new_batch_size))
                current_batch_size = new_batch_size

            except Exception as e:
                logger.warning(_("batch.unexpected_error", size=current_batch_size, error=str(e)))

                # Reduce batch size by half on unexpected errors too
                new_batch_size = current_batch_size // 2
                if new_batch_size < min_batch_size:
                    return 0
                current_batch_size = new_batch_size

        # Should not reach here
        logger.error(_("batch.no_suitable_size", min_size=min_batch_size))
        return 0

    def _transcribe_with_auto_batch_size(self, batched_model, audio_path, starting_batch_size=None):
        """
        Transcribe with automatic batch size reduction on OOM.
        Similar to HuggingFace Accelerate's find_executable_batch_size decorator.

        This function automatically retries with smaller batch sizes if OOM occurs,
        implementing the same behavior as Accelerate's find_executable_batch_size.

        Args:
            batched_model: BatchedInferencePipeline instance
            audio_path: Path to audio file
            starting_batch_size: Initial batch size to try (uses self.batch_size if not specified)

        Returns:
            Tuple of (segments, info, actual_batch_size_used)
        """
        batch_size = starting_batch_size or self.batch_size or 32
        min_batch_size = 1

        while batch_size >= min_batch_size:
            try:
                logger.debug(_("batch.attempting_transcription", size=batch_size))

                # Try transcription with current batch size
                segments, info = batched_model.transcribe(
                    audio_path,
                    batch_size=batch_size,
                    **self.generation_config
                )

                # Success! Return results with the batch size that worked
                if batch_size < (starting_batch_size or self.batch_size or 32):
                    logger.info(_("batch.auto_adjusted", size=batch_size))

                return segments, info, batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    # Reduce batch size by 0.8 (20% reduction, similar to Accelerate's 0.9 but more aggressive)
                    new_batch_size = int(batch_size * 0.8)

                    # Ensure we reduce by at least 1
                    if new_batch_size == batch_size:
                        new_batch_size = batch_size - 1

                    logger.warning(_("batch.oom_reducing", old_size=batch_size, new_size=new_batch_size))

                    batch_size = new_batch_size

                    if batch_size < min_batch_size:
                        logger.error(_("batch.cannot_run_min", min_size=min_batch_size))
                        raise RuntimeError(_("batch.inference_failed", min_size=min_batch_size)) from e
                else:
                    # Not an OOM error, re-raise
                    raise

        # Should not reach here
        raise RuntimeError("Failed to find executable batch size")

    def _scan(self, base_dirs) -> list[InferenceTask]:
        tasks: list[InferenceTask] = []

        def process(base_path, audio_path):
            nonlocal tasks
            p = Path(audio_path)
            suffix = p.suffix.lower().lstrip('.')

            logger.debug(_("debug.processing", path=audio_path))
            logger.debug(_("debug.file_suffix", suffix=suffix))
            logger.debug(_("debug.valid_suffixes", suffixes=self.audio_suffixes))

            if suffix not in self.audio_suffixes:
                logger.debug(_("debug.skipped_suffix", suffix=suffix))
                return

            rel_path = p.relative_to(base_path)
            abs_path = Path(os.path.join(self.output_dir or base_path, rel_path))
            sub_formats = []

            for suffix in self.sub_formats:
                sub_path = abs_path.parent / f"{abs_path.stem}.{suffix}"
                if sub_path.exists() and not self.overwrite:
                    logger.debug(_("debug.subtitle_exists", path=sub_path))
                    continue
                sub_formats.append(suffix)

            if len(sub_formats) == 0:
                logger.debug(_("debug.skipped_all_exist"))
                return

            logger.debug(_("debug.added_task", formats=sub_formats))
            tasks.append(InferenceTask(audio_path, str(abs_path.parent / abs_path.stem), sub_formats))

        for base_dir in base_dirs:
            # Expand user home directory
            base_dir = os.path.expanduser(base_dir)
            logger.debug(_("debug.scanning", path=base_dir))

            parent_dir = os.path.dirname(base_dir)
            if os.path.isdir(base_dir):
                for root, dirs, files in os.walk(base_dir, topdown=True):
                    for file in files:
                        process(parent_dir, os.path.join(root, file))
            else:
                process(parent_dir, base_dir)

        logger.info(_("files.found", count=len(tasks)))
        return tasks


def diagnose_environment():
    """Run comprehensive environment diagnostics for debugging"""
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)

    # System info
    print("\n1. System Information:")
    print(f"   Platform: {platform.system()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print(f"   Frozen: {getattr(sys, 'frozen', False)}")

    if getattr(sys, 'frozen', False):
        print(f"   Bundle Dir: {getattr(sys, '_MEIPASS', 'Unknown')}")

    # CUDA environment
    print("\n2. CUDA Environment Variables:")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'CUDNN_HOME', 'LD_LIBRARY_PATH', 'PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH' and value != 'Not set':
            # Just show cuda-related paths
            cuda_paths = [p for p in value.split(os.pathsep) if 'cuda' in p.lower() or 'nvidia' in p.lower()]
            value = os.pathsep.join(cuda_paths) if cuda_paths else 'No CUDA paths in PATH'
        print(f"   {var}: {value}")

    # Check for nvidia-smi
    print("\n3. NVIDIA GPU Detection:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,cuda_version', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"   GPU Info: {result.stdout.strip()}")
        else:
            print("   nvidia-smi failed")
    except FileNotFoundError:
        print("   nvidia-smi not found in PATH")
    except Exception as e:
        print(f"   Error: {e}")


def check_onnxruntime_detailed():
    """Detailed ONNX Runtime check for debugging"""
    print("\n" + "=" * 60)
    print("ONNX RUNTIME DIAGNOSTICS")
    print("=" * 60)

    try:
        import onnxruntime as ort
        print(f"\n✓ onnxruntime imported successfully")
        print(f"  Version: {ort.__version__}")
        print(f"  Location: {ort.__file__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"\n  Available providers: {providers}")

        # Check for GPU support
        has_cuda = 'CUDAExecutionProvider' in providers
        has_tensorrt = 'TensorrtExecutionProvider' in providers
        has_directml = 'DmlExecutionProvider' in providers

        print(f"\n  GPU Support:")
        print(f"    CUDA: {'✓ Available' if has_cuda else '✗ Not Available'}")
        print(f"    TensorRT: {'✓ Available' if has_tensorrt else '✗ Not Available'}")
        print(f"    DirectML: {'✓ Available' if has_directml else '✗ Not Available'}")

        if not has_cuda and sys.platform != 'darwin':
            print("\n  ⚠️ CUDA not available. This might be because:")
            print("    1. onnxruntime (CPU) is installed instead of onnxruntime-gpu")
            print("    2. CUDA libraries are missing or not in PATH")
            print("    3. Incompatible CUDA/cuDNN versions")

        # Check bundled libraries if frozen
        if getattr(sys, 'frozen', False):
            bundle_dir = getattr(sys, '_MEIPASS', '')
            print(f"\n  Checking bundled libraries in: {bundle_dir}")

            cuda_libs = []
            onnx_libs = []

            try:
                for root, dirs, files in os.walk(bundle_dir):
                    for file in files:
                        if any(x in file.lower() for x in ['cuda', 'cudnn', 'cublas', 'cufft']):
                            cuda_libs.append(file)
                        elif 'onnx' in file.lower():
                            onnx_libs.append(file)

                if cuda_libs:
                    print(f"\n  Found {len(cuda_libs)} CUDA-related libraries:")
                    for lib in cuda_libs[:10]:
                        print(f"    - {lib}")
                    if len(cuda_libs) > 10:
                        print(f"    ... and {len(cuda_libs) - 10} more")
                else:
                    print("\n  ⚠️ No CUDA libraries found in bundle")
            except Exception as e:
                print(f"  Error scanning bundle: {e}")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import onnxruntime: {e}")
        print("\nSuggestions:")
        print("  1. Install onnxruntime-gpu for GPU support")
        print("  2. Check if package is bundled correctly in PyInstaller")
        return False
    except Exception as e:
        print(f"\n✗ Error during ONNX Runtime check: {e}")
        traceback.print_exc()
        return False


def test_vad_initialization():
    """Test VAD model initialization for debugging"""
    print("\n" + "=" * 60)
    print("VAD MODEL TEST")
    print("=" * 60)

    try:
        from .vad_manager import WhisperVADOnnxWrapper, VadModelManager
        print("✓ VAD modules imported successfully")

        # Check for model files
        model_paths = [
            'models/whisper_vad.onnx',
            'models/vad/whisper_vad.onnx',
            os.path.join(os.path.dirname(sys.executable), 'models', 'whisper_vad.onnx'),
        ]

        # If frozen, also check in bundle directory
        if getattr(sys, 'frozen', False):
            bundle_dir = getattr(sys, '_MEIPASS', '')
            model_paths.extend([
                os.path.join(bundle_dir, 'models', 'whisper_vad.onnx'),
                os.path.join(bundle_dir, 'whisper_vad.onnx'),
            ])

        model_path = None
        print("\nSearching for VAD model:")
        for path in model_paths:
            exists = os.path.exists(path)
            print(f"  {path}: {'Found' if exists else 'Not found'}")
            if exists and model_path is None:
                model_path = path

        if model_path:
            print(f"\n✓ Using model: {model_path}")

            # Try to initialize
            print("\nTesting VAD initialization (GPU if available):")
            try:
                wrapper = WhisperVADOnnxWrapper(
                    model_path=model_path,
                    force_cpu=False,
                    num_threads=1
                )
                print(f"  ✓ Device: {wrapper.device}")
                print(f"  ✓ Providers: {wrapper.session.get_providers()}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Test with forced CPU for comparison
            print("\nTesting VAD initialization (Force CPU):")
            try:
                wrapper_cpu = WhisperVADOnnxWrapper(
                    model_path=model_path,
                    force_cpu=True,
                    num_threads=1
                )
                print(f"  ✓ Device: {wrapper_cpu.device}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print("\n✗ No VAD model file found")
            print("  Download the model using download_models.py")

    except ImportError as e:
        print(f"✗ Failed to import VAD modules: {e}")
    except Exception as e:
        print(f"✗ Error during VAD test: {e}")
        traceback.print_exc()


def launch_debug_console():
    """Launch interactive Python console for debugging"""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEBUG CONSOLE")
    print("=" * 60)
    print("\nYou now have access to an interactive Python console.")
    print("\nAvailable commands:")
    print("  diagnose()       - Run environment diagnostics")
    print("  check_onnx()     - Check ONNX Runtime status")
    print("  test_vad()       - Test VAD initialization")
    print("  import X         - Try importing any module")
    print("  exit() or Ctrl+D - Exit console and continue")
    print("\nUseful variables:")
    print("  sys.path         - Python module search paths")
    print("  os.environ       - Environment variables")
    print("  sys.frozen       - Check if running from PyInstaller")
    print("=" * 60 + "\n")

    # Create namespace with useful functions
    namespace = {
        'diagnose': diagnose_environment,
        'check_onnx': check_onnxruntime_detailed,
        'test_vad': test_vad_initialization,
        'sys': sys,
        'os': os,
        'platform': platform,
    }

    # Launch interactive console
    code.InteractiveConsole(locals=namespace).interact(banner="")


def main():
    """Main entry point for the script"""
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        # When run as a module, don't change directory
        pass

    args = parse_arguments()

    # Display open-source notice
    print("=" * 70)
    print("⚠️  重要声明 / IMPORTANT NOTICE")
    print("=" * 70)
    print("本软件开源于: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice")
    print("开发团队: AI汉化组 (https://t.me/transWithAI)")
    print("任何第三方非免费下载均为智商税")
    print("=" * 70)
    print()

    # Check if console mode requested
    if args.console:
        # Run diagnostics first
        diagnose_environment()
        check_onnxruntime_detailed()
        test_vad_initialization()

        # Launch interactive console
        launch_debug_console()

        # After console exits, ask if user wants to continue with normal operation
        print("\nDebug console exited.")
        try:
            response = input("Continue with normal inference? (y/n): ").strip().lower()
            if response != 'y':
                print("Exiting...")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

    # Normal operation
    logger.setLevel(args.log_level)

    # Add file logging to latest.log in current working directory
    # This helps users report issues by providing a log file
    log_file_path = os.path.join(os.getcwd(), 'latest.log')
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(args.log_level)

    # Add file handler to the module logger
    logger.addHandler(file_handler)

    logger.info(_("info.logging_to_file").format(path=log_file_path))
    logger.info(_("info.program_version").format(version="v1.3"))
    logger.info(_("info.python_version").format(version=sys.version))
    logger.info(_("info.platform").format(platform=platform.platform()))
    logger.info(_("info.arguments").format(args=vars(args)))

    if len(args.base_dirs) == 0:
        logger.warning(_("warnings.drag_files"))
        sys.exit(1)

    inference = Inference(args)
    inference.generates(args.base_dirs)
    sys.exit(0)


if __name__ == '__main__':
    # When run directly as a script
    import os
    os.chdir(os.path.dirname(__file__))
    main()
