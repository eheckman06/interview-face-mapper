from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from .faces import FaceObservation
from .speech_names import (
    MIN_FRAME_BRIGHTNESS,
    SEARCH_OFFSETS_S,
    _frame_brightness,
    _read_video_frame,
)
from .video_io import ffmpeg_path, get_video_metadata


INTRO_PRE_ROLL_S = 4.0
INTRO_MIN_DURATION_S = 24.0
INTRO_MAX_DURATION_S = 48.0
INTRO_POST_WIDE_PAD_S = 12.0
INTRO_SEARCH_AHEAD_S = 150.0
WIDE_SHOT_SEGMENT_S = 14.0
WIDE_SHOT_PRE_ROLL_S = 3.0
AUDIO_VISUAL_LAG_THRESHOLD_S = 8.0
REALIGNED_VIDEO_DURATION_S = 30.0
REALIGNED_VIDEO_PRE_ROLL_S = 3.0
REALIGNED_AUDIO_PRE_ROLL_S = 2.0
REALIGNED_AUDIO_DURATION_S = 14.0


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "intro"


def _load_observations(output_dir: Path) -> List[FaceObservation]:
    faces_path = output_dir / "faces.json"
    if not faces_path.exists():
        return []
    payload = json.loads(faces_path.read_text(encoding="utf-8"))
    return [FaceObservation(**item) for item in payload.get("observations", [])]


def _face_presence_windows(
    observations: List[FaceObservation],
    start_s: float,
    end_s: float,
) -> List[Tuple[float, int]]:
    buckets: dict[int, set[str]] = {}
    for obs in observations:
        if not obs.cluster_id:
            continue
        if obs.timestamp < start_s or obs.timestamp > end_s:
            continue
        bucket = int(round(obs.timestamp))
        buckets.setdefault(bucket, set()).add(obs.cluster_id)
    return [(float(second), len(clusters)) for second, clusters in sorted(buckets.items())]


def _peak_window(
    windows: List[Tuple[float, int]],
) -> Tuple[float, int]:
    if not windows:
        return 0.0, 0
    peak_t, peak_count = max(windows, key=lambda item: (item[1], item[0]))
    return peak_t, peak_count


def _global_wide_shot(
    observations: List[FaceObservation],
) -> Tuple[float, int]:
    if not observations:
        return 0.0, 0
    min_ts = min(obs.timestamp for obs in observations)
    max_ts = max(obs.timestamp for obs in observations)
    windows = _face_presence_windows(observations, min_ts, max_ts)
    return _peak_window(windows)


def _first_visible_timestamp(
    video_path: Path,
    timestamp_s: float,
    max_ahead_s: float = 120.0,
) -> Optional[float]:
    coarse_offsets = [offset for offset in SEARCH_OFFSETS_S if offset <= max_ahead_s]
    if max_ahead_s not in coarse_offsets:
        coarse_offsets.append(max_ahead_s)

    earliest: Optional[float] = None
    for index, offset in enumerate(coarse_offsets):
        lower_bound = timestamp_s + (coarse_offsets[index - 1] if index else 0.0)
        upper_bound = timestamp_s + offset
        step_start = int(max(lower_bound, 0.0))
        step_end = int(upper_bound)
        for second in range(step_start, step_end + 1):
            frame = _read_video_frame(video_path, float(second))
            if frame is None or _frame_brightness(frame) < MIN_FRAME_BRIGHTNESS:
                continue
            earliest = float(second)
            break
        if earliest is not None:
            break
    return earliest


def _needs_audio_video_realign(
    video_path: Path,
    timestamp_s: float,
    face_id: str = "",
    lip_score: float = 0.0,
) -> Tuple[bool, Optional[float]]:
    visual_s = _first_visible_timestamp(video_path, timestamp_s)
    if visual_s is None or visual_s <= timestamp_s + AUDIO_VISUAL_LAG_THRESHOLD_S:
        return False, visual_s
    if face_id and lip_score > 0:
        return False, visual_s
    return True, visual_s


def _encode_av_realigned_segment(
    video_path: Path,
    output_path: Path,
    audio_start_s: float,
    audio_duration_s: float,
    video_start_s: float,
    video_duration_s: float,
) -> Path:
    _, video_duration = get_video_metadata(video_path)
    if video_duration > 0:
        video_start_s = min(video_start_s, max(video_duration - 1.0, 0.0))
        video_duration_s = min(video_duration_s, max(video_duration - video_start_s, 1.0))
        audio_start_s = min(audio_start_s, max(video_duration - 1.0, 0.0))
        audio_duration_s = min(audio_duration_s, max(video_duration - audio_start_s, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-ss",
        f"{video_start_s:.3f}",
        "-i",
        str(video_path),
        "-ss",
        f"{audio_start_s:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{min(video_duration_s, audio_duration_s):.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _clip_window_for_intro(
    timestamp_s: float,
    observations: List[FaceObservation],
    cluster_count: int = 0,
) -> Tuple[float, float]:
    search_start = max(timestamp_s - 10.0, 0.0)
    search_end = timestamp_s + INTRO_SEARCH_AHEAD_S
    windows = _face_presence_windows(observations, search_start, search_end)
    peak_t, peak_count = _peak_window(windows)

    target_count = cluster_count or peak_count
    best_t = peak_t
    best_count = peak_count
    for second, count in windows:
        if count >= best_count:
            best_t = float(second)
            best_count = count
        if target_count and count >= target_count:
            best_t = float(second)
            break

    start_s = max(timestamp_s - INTRO_PRE_ROLL_S, 0.0)
    end_s = max(timestamp_s + 18.0, best_t + INTRO_POST_WIDE_PAD_S)
    duration_s = min(INTRO_MAX_DURATION_S, max(INTRO_MIN_DURATION_S, end_s - start_s))
    return start_s, duration_s


def _encode_segment(
    video_path: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
) -> Path:
    _, video_duration = get_video_metadata(video_path)
    if video_duration > 0:
        start_s = min(start_s, max(video_duration - 1.0, 0.0))
        duration_s = min(duration_s, max(video_duration - start_s, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _concat_segments(segment_paths: List[Path], output_path: Path) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        concat_list = Path(handle.name)
        for segment in segment_paths:
            handle.write(f"file '{segment.resolve()}'\n")

    try:
        cmd = [
            ffmpeg_path(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        concat_list.unlink(missing_ok=True)
    return output_path


def extract_intro_clip(
    video_path: Path,
    output_path: Path,
    timestamp_s: float,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    pre_roll_s: float = INTRO_PRE_ROLL_S,
    observations: Optional[List[FaceObservation]] = None,
    cluster_count: int = 0,
    face_id: str = "",
    lip_score: float = 0.0,
) -> Tuple[Path, int, bool, bool]:
    if not video_path.exists():
        raise FileNotFoundError(f"Missing source video: {video_path}")

    realign, visual_s = _needs_audio_video_realign(
        video_path,
        timestamp_s,
        face_id=face_id,
        lip_score=lip_score,
    )

    if realign and visual_s is not None:
        audio_start_s = max(timestamp_s - REALIGNED_AUDIO_PRE_ROLL_S, 0.0)
        video_start_s = max(visual_s - REALIGNED_VIDEO_PRE_ROLL_S, 0.0)
        segment_duration_s = REALIGNED_VIDEO_DURATION_S
        primary_peak = 0
        if observations:
            primary_peak = max(
                (
                    count
                    for _, count in _face_presence_windows(
                        observations,
                        video_start_s,
                        video_start_s + segment_duration_s,
                    )
                ),
                default=0,
            )
        global_wide_t, global_wide_count = (0.0, 0)
        if observations:
            global_wide_t, global_wide_count = _global_wide_shot(observations)
        needs_wide_tail = (
            observations
            and global_wide_count > 0
            and global_wide_count > primary_peak
            and global_wide_t > video_start_s + segment_duration_s - 5.0
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            intro_segment = temp_root / "intro.mp4"
            _encode_av_realigned_segment(
                video_path,
                intro_segment,
                audio_start_s=audio_start_s,
                audio_duration_s=REALIGNED_AUDIO_DURATION_S,
                video_start_s=video_start_s,
                video_duration_s=segment_duration_s,
            )
            if needs_wide_tail:
                wide_segment = temp_root / "wide.mp4"
                wide_start = max(global_wide_t - WIDE_SHOT_PRE_ROLL_S, 0.0)
                _encode_segment(video_path, wide_segment, wide_start, WIDE_SHOT_SEGMENT_S)
                _concat_segments([intro_segment, wide_segment], output_path)
                return (
                    output_path,
                    max(primary_peak, global_wide_count),
                    True,
                    True,
                )
            _concat_segments([intro_segment], output_path)
        return output_path, primary_peak, False, True

    if start_s is None or duration_s is None:
        if observations is None:
            start_s, duration_s = max(timestamp_s - pre_roll_s, 0.0), INTRO_MIN_DURATION_S
        else:
            start_s, duration_s = _clip_window_for_intro(timestamp_s, observations, cluster_count)

    primary_peak = 0
    if observations:
        primary_peak = max(
            (count for _, count in _face_presence_windows(observations, start_s, start_s + duration_s)),
            default=0,
        )

    global_wide_t, global_wide_count = (0.0, 0)
    if observations:
        global_wide_t, global_wide_count = _global_wide_shot(observations)

    needs_wide_tail = (
        observations
        and global_wide_count > 0
        and global_wide_count > primary_peak
        and global_wide_t > start_s + duration_s - 5.0
    )

    if not needs_wide_tail:
        _encode_segment(video_path, output_path, start_s, duration_s)
        return output_path, primary_peak, False, False

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        intro_segment = temp_root / "intro.mp4"
        wide_segment = temp_root / "wide.mp4"
        _encode_segment(video_path, intro_segment, start_s, duration_s)
        wide_start = max(global_wide_t - WIDE_SHOT_PRE_ROLL_S, 0.0)
        _encode_segment(video_path, wide_segment, wide_start, WIDE_SHOT_SEGMENT_S)
        _concat_segments([intro_segment, wide_segment], output_path)

    return output_path, max(primary_peak, global_wide_count), True, False


def export_intro_clips(output_dir: Path) -> List[dict]:
    detections_path = output_dir / "all_detected_names.json"
    if not detections_path.exists():
        return []

    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    observations = _load_observations(output_dir)
    faces_path = output_dir / "faces.json"
    cluster_count = 0
    if faces_path.exists():
        cluster_count = len(json.loads(faces_path.read_text(encoding="utf-8")).get("face_clusters", []))

    clip_dir = output_dir / "intro_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    for detection in detections:
        source_video = Path(detection.get("source_video", ""))
        if not source_video.exists() and faces_path.exists():
            source_video = Path(json.loads(faces_path.read_text(encoding="utf-8"))["video"])
        if not source_video.exists():
            detection["intro_clip_file"] = ""
            detection["intro_clip_faces_visible"] = 0
            detection["intro_clip_includes_full_cast"] = False
            detection["intro_clip_audio_realigned"] = False
            continue

        timestamp = float(detection["timestamp"])
        start_s, duration_s = _clip_window_for_intro(timestamp, observations, cluster_count)
        safe_name = _safe_stem(detection.get("name", "intro"))
        clip_tag = _safe_stem(detection.get("clip_label", "clip"))
        clip_name = f"{safe_name}_{int(timestamp)}_{clip_tag}.mp4"
        clip_path = clip_dir / clip_name
        try:
            _, peak_visible, includes_full_cast, audio_realigned = extract_intro_clip(
                source_video,
                clip_path,
                timestamp,
                start_s=start_s,
                duration_s=duration_s,
                observations=observations,
                cluster_count=cluster_count,
                face_id=detection.get("face_id", ""),
                lip_score=float(detection.get("lip_score", 0.0) or 0.0),
            )
            detection["intro_clip_file"] = str(Path("intro_clips") / clip_name)
            detection["intro_clip_start"] = round(start_s, 2)
            detection["intro_clip_duration"] = round(duration_s, 2)
            detection["intro_clip_faces_visible"] = peak_visible
            detection["intro_clip_includes_full_cast"] = includes_full_cast
            detection["intro_clip_audio_realigned"] = audio_realigned
        except (subprocess.CalledProcessError, OSError, ValueError):
            detection["intro_clip_file"] = ""
            detection["intro_clip_faces_visible"] = 0
            detection["intro_clip_includes_full_cast"] = False
            detection["intro_clip_audio_realigned"] = False

    detections_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return detections
