from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .correlate import LowerThirdSegment, apply_labels, correlate_speakers_to_faces, save_correlation
from .export import export_lower_thirds
from .face_sheet import export_face_name_sheet
from .faces import analyze_faces, rebuild_face_clusters
from .label_ui import generate_label_ui
from .screengrab import generate_named_screengrab
from .speech_names import extract_names_from_speech
from .speakers import diarize_speakers

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


@dataclass
class AnalysisResult:
    job_id: str
    output_dir: Path
    main_video: Path
    face_cluster_count: int
    speaker_segment_count: int
    extra_video_count: int
    clip_count: int


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "video"


def save_upload(upload_name: str, data: bytes, uploads_dir: Path, prefix: str) -> Path:
    suffix = Path(upload_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or '(none)'}. Upload MP4, MOV, or similar video files.")

    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / f"{prefix}_{_safe_stem(upload_name)}{suffix}"
    target.write_bytes(data)
    return target


def finalize_upload_path(staged_path: Path, uploads_dir: Path, prefix: str, upload_name: str) -> Path:
    suffix = Path(upload_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or '(none)'}. Upload MP4, MOV, or similar video files.")
    target = uploads_dir / f"{prefix}_{_safe_stem(upload_name)}{suffix}"
    staged_path.replace(target)
    return target


def _save_clips_manifest(
    output_dir: Path,
    main_video: Path,
    extra_videos: List[Path],
) -> None:
    payload = {
        "main_video": str(main_video),
        "extra_videos": [str(item) for item in extra_videos],
        "all_clips": [str(main_video), *[str(item) for item in extra_videos]],
    }
    (output_dir / "clips.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clip_paths_from_output(output_dir: Path, main_video: Path) -> List[Path]:
    clips_path = output_dir / "clips.json"
    if clips_path.exists():
        payload = json.loads(clips_path.read_text(encoding="utf-8"))
        return [Path(item) for item in payload.get("all_clips", [])]
    faces_path = output_dir / "faces.json"
    if faces_path.exists():
        payload = json.loads(faces_path.read_text(encoding="utf-8"))
        sampled = payload.get("sampled_videos", [])
        if sampled:
            return [Path(item) for item in sampled]
    return [main_video]


def run_analysis(
    main_video: Path,
    output_dir: Path,
    extra_videos: Optional[List[Path]] = None,
    max_speakers: int = 6,
    face_sample_fps: float = 2.0,
    max_duration_s: Optional[float] = None,
    extra_max_duration_s: Optional[float] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    def update(message: str) -> None:
        if progress:
            progress(message)

    output_dir.mkdir(parents=True, exist_ok=True)
    extras = extra_videos or []
    clip_count = 1 + len(extras)
    _save_clips_manifest(output_dir, main_video, extras)

    if clip_count > 1:
        update(
            f"Detecting faces across {clip_count} clips (step 1 of 4). "
            "This is the slowest step..."
        )
    else:
        update("Detecting faces (step 1 of 4). This is the slowest step...")
    faces_payload = analyze_faces(
        main_video,
        output_dir,
        sample_fps=face_sample_fps,
        extra_videos=extras,
        max_duration_s=max_duration_s,
        extra_max_duration_s=extra_max_duration_s or max_duration_s,
    )
    estimated_people = max(len(faces_payload.get("face_clusters", [])), 2)
    update("Diarizing speakers on main clip (step 2 of 4)...")
    speaker_segments = diarize_speakers(
        main_video,
        output_dir,
        max_speakers=max(max_speakers, estimated_people),
        max_duration_s=max_duration_s,
    )
    update("Matching speakers to faces on main clip (step 3 of 4)...")
    mappings, lower_thirds = correlate_speakers_to_faces(main_video, faces_payload, speaker_segments)
    save_correlation(output_dir, mappings, lower_thirds)

    if clip_count > 1:
        update(
            f"Listening for name introductions across {clip_count} clips (step 4 of 4)..."
        )
    else:
        update("Listening for on-camera name introductions (step 4 of 4)...")
    spoken_labels = extract_names_from_speech(
        main_video,
        output_dir,
        speaker_segments,
        faces_payload,
        intro_window_s=None,
        extra_videos=extras if extras else None,
    )
    if spoken_labels:
        (output_dir / "labels.json").write_text(json.dumps(spoken_labels, indent=2), encoding="utf-8")

    generate_label_ui(output_dir, faces_payload)
    face_sheet = export_face_name_sheet(output_dir, labels=spoken_labels or None)

    job_id = output_dir.name
    return AnalysisResult(
        job_id=job_id,
        output_dir=output_dir,
        main_video=main_video,
        face_cluster_count=len(faces_payload["face_clusters"]),
        speaker_segment_count=len(speaker_segments),
        extra_video_count=len(extras),
        clip_count=clip_count,
    )


def run_finalize(output_dir: Path, labels: dict) -> dict:
    correlation_path = output_dir / "correlation.json"
    if not correlation_path.exists():
        raise FileNotFoundError("Missing correlation.json. Run analysis first.")

    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    correlation = json.loads(correlation_path.read_text(encoding="utf-8"))
    lower_thirds = [LowerThirdSegment(**item) for item in correlation["lower_thirds"]]
    labeled = apply_labels(lower_thirds, labels)
    export_lower_thirds(output_dir, labeled)
    screengrab_path = generate_named_screengrab(output_dir, labels=labels)
    face_sheet = export_face_name_sheet(output_dir, labels=labels)

    return {
        "lower_thirds_json": str(output_dir / "lower_thirds.json"),
        "lower_thirds_csv": str(output_dir / "lower_thirds.csv"),
        "lower_third_markers_edl": str(output_dir / "lower_third_markers.edl"),
        "named_screengrab": str(screengrab_path),
        "face_name_sheet_csv": face_sheet["face_name_sheet_csv"],
        "face_name_sheet_html": face_sheet["face_name_sheet_html"],
        "face_name_sheet_json": face_sheet["face_name_sheet_json"],
        "cast_sheet": face_sheet["cast_sheet"],
        "intro_roster_html": face_sheet.get("intro_roster_html", ""),
        "intro_roster_image": face_sheet.get("intro_roster_image", ""),
    }


def rebuild_analysis(output_dir: Path, progress: Optional[Callable[[str], None]] = None) -> dict:
    def update(message: str) -> None:
        if progress:
            progress(message)

    update("Re-clustering faces with improved wide-shot detection...")
    faces_payload = rebuild_face_clusters(output_dir)
    main_video = Path(faces_payload["video"])
    extra_clips = [Path(item) for item in _clip_paths_from_output(output_dir, main_video)]
    extra_clips = [item for item in extra_clips if item.resolve() != main_video.resolve()]

    update("Re-matching speakers to faces on main clip...")
    speaker_segments = []
    speakers_path = output_dir / "speakers.json"
    if speakers_path.exists():
        from .speakers import SpeakerSegment

        speaker_segments = [
            SpeakerSegment(**item)
            for item in json.loads(speakers_path.read_text(encoding="utf-8"))["segments"]
        ]

    mappings, lower_thirds = correlate_speakers_to_faces(main_video, faces_payload, speaker_segments)
    save_correlation(output_dir, mappings, lower_thirds)

    if extra_clips:
        update(f"Listening for name introductions across {1 + len(extra_clips)} clips...")
    else:
        update("Listening for on-camera name introductions...")
    spoken_labels = extract_names_from_speech(
        main_video,
        output_dir,
        speaker_segments,
        faces_payload,
        extra_videos=extra_clips if extra_clips else None,
    )
    if spoken_labels:
        (output_dir / "labels.json").write_text(json.dumps(spoken_labels, indent=2), encoding="utf-8")

    generate_label_ui(output_dir, faces_payload)
    face_sheet = export_face_name_sheet(output_dir, labels=spoken_labels or None)

    return {
        "face_clusters": len(faces_payload["face_clusters"]),
        "face_name_sheet_html": face_sheet["face_name_sheet_html"],
        "cast_sheet": face_sheet["cast_sheet"],
        "intro_roster_html": face_sheet.get("intro_roster_html", ""),
        "intro_roster_image": face_sheet.get("intro_roster_image", ""),
    }


def create_job_root(base_dir: Path) -> Path:
    job_id = uuid.uuid4().hex[:12]
    job_dir = base_dir / job_id
    (job_dir / "uploads").mkdir(parents=True, exist_ok=True)
    return job_dir
