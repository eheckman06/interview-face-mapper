from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .faces import FaceAnalyzer, FaceObservation
from .speakers import SpeakerSegment
from .video_io import extract_audio, ffmpeg_path


# Higher priority patterns are tried first (larger base confidence).
NAME_RULES: List[Tuple[re.Pattern[str], float]] = [
    (
        re.compile(
            r"\bmy name is\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)",
            re.IGNORECASE,
        ),
        40.0,
    ),
    (
        re.compile(
            r"\bi'?m\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?),?\s+and\s+i'?m\s+(?:also\s+)?from",
            re.IGNORECASE,
        ),
        35.0,
    ),
    (
        re.compile(
            r"\b(?!i'?m\b)([A-Za-z][A-Za-z'\-]+)\s+and\s+i'?m\s+from",
            re.IGNORECASE,
        ),
        35.0,
    ),
    (
        re.compile(
            r"\bi'?m\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)\s+and\s+i'?m\s+(?:very|currently|also|not)",
            re.IGNORECASE,
        ),
        32.0,
    ),
    (
        re.compile(r"\b(?:hi|hey|hello),?\s*i'?m\s+([A-Za-z][A-Za-z'\-]+)", re.IGNORECASE),
        30.0,
    ),
    (
        re.compile(
            r"\bi'?m\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)\s+and\s+i'?m\s+from",
            re.IGNORECASE,
        ),
        28.0,
    ),
    (
        re.compile(r"\bthis is\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)", re.IGNORECASE),
        25.0,
    ),
    (
        re.compile(r"\bcall me\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)", re.IGNORECASE),
        25.0,
    ),
]

BLOCKED_NAME_WORDS = {
    "here",
    "back",
    "good",
    "great",
    "sure",
    "yes",
    "no",
    "the",
    "a",
    "an",
    "and",
    "with",
    "from",
    "your",
    "our",
    "when",
    "going",
    "first",
    "what",
    "that",
    "this",
    "there",
    "where",
    "how",
    "why",
    "just",
    "very",
    "really",
    "well",
    "so",
    "oh",
    "sorry",
    "involved",
    "currently",
    "not",
    "also",
    "excited",
    "going",
    "here",
    "telling",
    "view",
    "voting",
    "sorry",
    "all",
    "right",
    "okay",
}


def _clean_name(raw: str) -> Optional[str]:
    name = raw.strip().strip(".,!?")
    if re.match(r"^i'?m\b", name, re.IGNORECASE):
        return None
    name = re.split(r"\s+and\s+i'?m\b", name, flags=re.IGNORECASE)[0]
    name = re.split(r"\s+from\b", name, flags=re.IGNORECASE)[0]
    name = " ".join(part for part in name.split() if part)
    if not name:
        return None
    words = name.split()
    if len(words) > 3:
        return None
    for word in words:
        if len(word) < 2 or word.lower() in BLOCKED_NAME_WORDS:
            return None
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", word):
            return None
    return name


def extract_spoken_names(text: str) -> List[Tuple[str, float]]:
    cleaned = text.strip()
    if not cleaned:
        return []
    found: List[Tuple[str, float]] = []
    seen: set[str] = set()
    for pattern, base_confidence in NAME_RULES:
        for match in pattern.finditer(cleaned):
            name = _clean_name(match.group(1))
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            found.append((name, base_confidence))
    return found


def _extract_wav_segment(source_wav: Path, start_s: float, end_s: float, destination: Path) -> Path:
    duration = max(end_s - start_s, 0.25)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path(),
        "-y",
        "-ss",
        str(max(start_s, 0.0)),
        "-t",
        str(duration),
        "-i",
        str(source_wav),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(destination),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return destination


def _load_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel("base", device="cpu", compute_type="int8")


def _audio_duration(wav_path: Path) -> float:
    return float(sf.info(wav_path).duration)


def _cluster_centroids(observations: List[FaceObservation]) -> Dict[str, np.ndarray]:
    buckets: Dict[str, List[np.ndarray]] = {}
    for obs in observations:
        if obs.cluster_id:
            buckets.setdefault(obs.cluster_id, []).append(np.array(obs.embedding, dtype=np.float32))
    return {cluster_id: np.mean(vectors, axis=0) for cluster_id, vectors in buckets.items()}


def _nearest_cluster(observation: FaceObservation, centroids: Dict[str, np.ndarray]) -> str:
    obs_vec = np.array(observation.embedding, dtype=np.float32)
    best_id = ""
    best_dist = float("inf")
    for cluster_id, centroid in centroids.items():
        dist = np.linalg.norm(obs_vec - centroid)
        if dist < best_dist:
            best_dist = dist
            best_id = cluster_id
    return best_id


def _rank_faces_at_timestamp(
    video_path: Path,
    timestamp: float,
    centroids: Dict[str, np.ndarray],
    analyzer: FaceAnalyzer,
) -> List[Tuple[str, float]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return []

    observations = analyzer.analyze_frame(timestamp, frame)
    ranked: List[Tuple[str, float]] = []
    for obs in observations:
        cluster_id = _nearest_cluster(obs, centroids)
        if not cluster_id:
            continue
        x1, y1, x2, y2 = obs.bbox
        area = float((x2 - x1) * (y2 - y1))
        score = obs.lip_openness * 1000.0 + area
        ranked.append((cluster_id, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _save_intro_thumbnail(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    centroids: Dict[str, np.ndarray],
    analyzer: FaceAnalyzer,
) -> Tuple[str, Optional[str]]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "", None
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return "", None

    observations = analyzer.analyze_frame(timestamp, frame)
    if not observations:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(output_path, quality=90)
        return "", str(output_path)

    best_obs = max(
        observations,
        key=lambda obs: obs.lip_openness * 1000 + (obs.bbox[2] - obs.bbox[0]) * (obs.bbox[3] - obs.bbox[1]),
    )
    face_id = _nearest_cluster(best_obs, centroids)
    x1, y1, x2, y2 = best_obs.bbox
    crop = frame[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if crop.size > 0:
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output_path, quality=90)
    else:
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(output_path, quality=90)
    return face_id, str(output_path)


def _dedupe_events(events: List[dict], window_s: float = 90.0) -> List[dict]:
    kept: List[dict] = []
    for event in sorted(events, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for existing in kept:
            if existing["name"].lower() != event["name"].lower():
                continue
            if existing.get("source_video") != event.get("source_video"):
                continue
            if abs(existing["timestamp"] - event["timestamp"]) <= window_s:
                duplicate = True
                break
        if not duplicate:
            kept.append(event)
    kept.sort(key=lambda item: item["timestamp"])
    return kept


def _clip_label(video_path: Path) -> str:
    return video_path.stem


def _wav_for_clip(video_path: Path, output_dir: Path, is_main: bool) -> Path:
    if is_main:
        wav_path = output_dir / "audio.wav"
        if wav_path.exists():
            return wav_path
    clip_audio_dir = output_dir / "clip_audio"
    clip_audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = clip_audio_dir / f"{_clip_label(video_path)}.wav"
    if not wav_path.exists():
        extract_audio(video_path, wav_path)
    return wav_path


def _events_from_wav(
    whisper_model,
    wav_path: Path,
    source_video: Path,
    scan_duration: Optional[float],
) -> List[dict]:
    duration = scan_duration or _audio_duration(wav_path)
    whisper_segments, _ = whisper_model.transcribe(
        str(wav_path),
        beam_size=1,
        vad_filter=True,
    )
    raw_events: List[dict] = []
    for whisper_segment in whisper_segments:
        if whisper_segment.start > duration:
            break
        transcript = whisper_segment.text.strip()
        if not transcript:
            continue
        timestamp = whisper_segment.start + 0.35
        for name, base_confidence in extract_spoken_names(transcript):
            confidence = base_confidence
            if timestamp < 120:
                confidence += 5
            raw_events.append(
                {
                    "name": name,
                    "timestamp": timestamp,
                    "transcript": transcript,
                    "confidence": confidence,
                    "source_video": str(source_video),
                }
            )
    return raw_events


def extract_names_from_speech(
    video_path: Path,
    output_dir: Path,
    speaker_segments: List[SpeakerSegment],
    faces_payload: dict,
    intro_window_s: Optional[float] = None,
    extra_videos: Optional[List[Path]] = None,
) -> Dict[str, str]:
    del speaker_segments
    clip_paths = [video_path]
    if extra_videos:
        clip_paths.extend(extra_videos)
    elif faces_payload.get("sampled_videos"):
        sampled = [Path(item) for item in faces_payload["sampled_videos"]]
        clip_paths = sampled

    observations = [FaceObservation(**item) for item in faces_payload["observations"]]
    centroids = _cluster_centroids(observations)
    if not centroids:
        return {}

    whisper_model = _load_whisper_model()
    analyzer = FaceAnalyzer()
    raw_events: List[dict] = []
    all_detections: List[dict] = []

    main_video_path = Path(faces_payload.get("video", str(video_path)))
    try:
        for clip_path in clip_paths:
            is_main = clip_path.resolve() == main_video_path.resolve()
            wav_path = _wav_for_clip(clip_path, output_dir, is_main=is_main)
            scan_duration = intro_window_s
            if scan_duration is None and not is_main:
                scan_duration = _audio_duration(wav_path)
            raw_events.extend(
                _events_from_wav(whisper_model, wav_path, clip_path, scan_duration)
            )

        raw_events = _dedupe_events(raw_events)
        face_labels: Dict[str, str] = {}
        assigned_faces: set[str] = set()
        intro_thumb_dir = output_dir / "name_intro_thumbnails"
        intro_thumb_dir.mkdir(parents=True, exist_ok=True)

        for event in raw_events:
            clip_path = Path(event["source_video"])
            ranked_faces = _rank_faces_at_timestamp(
                clip_path, event["timestamp"], centroids, analyzer
            )

            chosen_face = ""
            lip_score = 0.0
            for face_id, score in ranked_faces:
                if face_id not in assigned_faces:
                    chosen_face = face_id
                    lip_score = score
                    break
            if not chosen_face and ranked_faces:
                chosen_face, lip_score = ranked_faces[0]

            clip_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", _clip_label(clip_path)).strip("._") or "clip"
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", event["name"]).strip("._") or "name"
            thumb_path = intro_thumb_dir / f"{safe_name}_{int(event['timestamp'])}_{clip_tag}.jpg"
            thumb_face, thumb_file = _save_intro_thumbnail(
                clip_path,
                event["timestamp"],
                thumb_path,
                centroids,
                analyzer,
            )
            if not chosen_face and thumb_face:
                chosen_face = thumb_face

            detection = {
                "name": event["name"],
                "timestamp": event["timestamp"],
                "transcript": event["transcript"],
                "confidence": event["confidence"],
                "face_id": chosen_face,
                "lip_score": lip_score,
                "source_video": event["source_video"],
                "clip_label": _clip_label(clip_path),
                "thumbnail_file": str(Path("name_intro_thumbnails") / thumb_path.name),
            }
            all_detections.append(detection)

            if chosen_face and chosen_face not in assigned_faces and chosen_face not in face_labels:
                assigned_faces.add(chosen_face)
                face_labels[chosen_face] = event["name"]
            elif (
                chosen_face
                and chosen_face not in face_labels
                and event["confidence"] >= 30
            ):
                assigned_faces.add(chosen_face)
                face_labels[chosen_face] = event["name"]
            elif (
                chosen_face
                and face_labels.get(chosen_face, "").lower() != event["name"].lower()
                and event["confidence"] >= 35
            ):
                face_labels[chosen_face] = event["name"]

        (output_dir / "all_detected_names.json").write_text(
            json.dumps(all_detections, indent=2),
            encoding="utf-8",
        )
    finally:
        analyzer.close()

    (output_dir / "spoken_names.json").write_text(
        json.dumps({"labels": face_labels, "all_detections": all_detections}, indent=2),
        encoding="utf-8",
    )
    return face_labels
