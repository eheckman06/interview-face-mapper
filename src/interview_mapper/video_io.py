from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import imageio_ffmpeg
import numpy as np


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(
    video_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
    max_duration_s: Optional[float] = None,
) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
    ]
    if max_duration_s is not None:
        cmd.extend(["-t", str(max_duration_s)])
    cmd.append(str(output_wav))
    subprocess.run(cmd, check=True, capture_output=True)
    return output_wav


def get_video_metadata(video_path: Path) -> Tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0.0
    cap.release()
    return fps, duration


def iter_frames(
    video_path: Path,
    sample_fps: float = 2.0,
    max_duration_s: Optional[float] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / sample_fps)), 1)
    max_frame = None
    if max_duration_s is not None:
        max_frame = int(max_duration_s * native_fps)
    frame_idx = 0

    while True:
        if max_frame is not None and frame_idx > max_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / native_fps
            yield timestamp, frame
        frame_idx += 1

    cap.release()


def iter_frames_in_range(
    video_path: Path,
    start_s: float,
    end_s: float,
    sample_fps: float = 8.0,
) -> Iterator[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(native_fps / sample_fps)), 1)
    start_frame = int(start_s * native_fps)
    end_frame = int(end_s * native_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % step == 0:
            yield frame_idx / native_fps, frame
        frame_idx += 1

    cap.release()
