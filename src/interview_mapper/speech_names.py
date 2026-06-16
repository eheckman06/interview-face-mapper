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


LETTER_SPELLING_PATTERN = re.compile(
    r"\b(?:[A-Z](?:\s*-\s*|\s+)){2,}[A-Z]\b",
    re.IGNORECASE,
)

SPELLING_CONTEXT_WINDOW_S = 12.0

SPELLING_CLARIFICATION_PATTERNS = [
    re.compile(r"\bwith\s+an?\s+([A-Z])\b", re.IGNORECASE),
    re.compile(r"\bspelled\s+with\s+an?\s+([A-Z])\b", re.IGNORECASE),
    re.compile(r"\bit'?s\s+an?\s+([A-Z])\b", re.IGNORECASE),
]

LOCATION_PATTERNS = [
    re.compile(r"\bi'?m\s+(?:also\s+)?from\s+([^.!?]+)", re.IGNORECASE),
    re.compile(r"\band\s+i'?m\s+from\s+([^.!?]+)", re.IGNORECASE),
    re.compile(r"\bfrom\s+([^.!?]+)", re.IGNORECASE),
]

PRONUNCIATION_PATTERNS = [
    re.compile(r"\bpronounced\s+(?:as\s+)?([^.,!?]+)", re.IGNORECASE),
    re.compile(r"\bpronunciation\s+(?:is\s+)?([^.,!?]+)", re.IGNORECASE),
    re.compile(r"\bit'?s\s+pronounced\s+([^.,!?]+)", re.IGNORECASE),
]


def _normalize_location(raw: str) -> str:
    location = raw.strip().strip(".,!?")
    location = re.split(r"\s+and\s+i'?m\b", location, flags=re.IGNORECASE)[0]
    location = re.split(r"\s+but\b", location, flags=re.IGNORECASE)[0]
    return " ".join(location.split())


def extract_location(text: str) -> str:
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        location = _normalize_location(match.group(1))
        if location and location.lower() not in BLOCKED_NAME_WORDS:
            return location
    return ""


def extract_letter_spellings(text: str) -> List[str]:
    spellings: List[str] = []
    seen: set[str] = set()
    for match in LETTER_SPELLING_PATTERN.finditer(text):
        spelling = match.group(0).upper()
        spelling = re.sub(r"\s*-\s*", "-", spelling)
        spelling = re.sub(r"\s+", "-", spelling)
        spelling = re.sub(r"-+", "-", spelling).strip("-")
        if spelling not in seen:
            seen.add(spelling)
            spellings.append(spelling)
    return spellings


def _letters_only(value: str) -> str:
    return re.sub(r"[^A-Za-z]", "", value).lower()


def _format_name_from_spelling(spelling: str) -> str:
    letters = _letters_only(spelling)
    if not letters:
        return ""
    if len(letters) <= 3:
        return letters.upper()
    return letters[0].upper() + letters[1:].lower()


def _spelling_relates_to_name(spelling: str, name: str) -> bool:
    spelling_letters = _letters_only(spelling)
    name_letters = _letters_only(name)
    if not spelling_letters or not name_letters:
        return False
    if name_letters in spelling_letters or spelling_letters in name_letters:
        return True
    if len(spelling_letters) == len(name_letters):
        differences = sum(left != right for left, right in zip(spelling_letters, name_letters))
        if differences <= 2:
            return True
    if len(spelling_letters) >= 4 and len(name_letters) >= 4:
        if spelling_letters[-4:] == name_letters[-4:]:
            return True
        if spelling_letters[-3:] == name_letters[-3:]:
            return True
    return False


def _extract_spelling_clarifications(text: str) -> List[str]:
    notes: List[str] = []
    for pattern in SPELLING_CLARIFICATION_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        letter = matches[-1].group(1).upper()
        note = f"Spelled with a {letter} on camera"
        if note not in notes:
            notes.append(note)
    return notes


def _apply_letter_clarification(name: str, clarifications: List[str]) -> str:
    if not name or not clarifications:
        return name
    for note in clarifications:
        match = re.search(r"Spelled with a ([A-Z]) on camera", note)
        if not match:
            continue
        letter = match.group(1)
        if name and name[0].upper() != letter:
            return letter + name[1:]
    return name


def _best_spelling_for_name(spellings: List[str], name: str) -> str:
    related = [spelling for spelling in spellings if _spelling_relates_to_name(spelling, name)]
    if not related:
        return ""
    return max(related, key=lambda item: len(_letters_only(item)))


def _pronunciation_from_spelling(spelling: str) -> str:
    chunks = [part for part in spelling.split("-") if part]
    if not chunks:
        return ""
    return " ".join(chunk.upper() for chunk in chunks)


def _apply_clarification_to_spelling(spelling: str, clarifications: List[str]) -> str:
    if not spelling or not clarifications:
        return spelling
    for note in reversed(clarifications):
        match = re.search(r"Spelled with a ([A-Z]) on camera", note)
        if not match:
            continue
        letter = match.group(1)
        parts = [part for part in spelling.split("-") if part]
        if not parts:
            return spelling
        if parts[0].upper() != letter:
            parts[0] = letter
            return "-".join(parts)
    return spelling


def _name_from_clarification(name: str, clarifications: List[str]) -> str:
    if not name or not clarifications:
        return name
    for note in reversed(clarifications):
        match = re.search(r"Spelled with a ([A-Z]) on camera", note)
        if not match:
            continue
        letter = match.group(1)
        if name[0].upper() != letter:
            return letter + name[1:]
    return name


def _should_apply_spelling_as_name(spelling: str, name: str) -> bool:
    if not _spelling_relates_to_name(spelling, name):
        return False
    spelling_letters = _letters_only(spelling)
    name_letters = _letters_only(name)
    if len(name.split()) > 1 and len(spelling_letters) > len(name_letters) + 2:
        return False
    return abs(len(spelling_letters) - len(name_letters)) <= 3


def _merge_transcript_context(primary: str, context: str = "") -> str:
    parts: List[str] = []
    seen: set[str] = set()
    for chunk in (primary, context):
        cleaned = " ".join(chunk.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            parts.append(cleaned)
    return " ".join(parts)


def extract_intro_details(
    transcript: str,
    name: str,
    context_transcript: str = "",
) -> Dict[str, str]:
    combined = _merge_transcript_context(transcript, context_transcript)
    location = extract_location(combined)
    spellings = extract_letter_spellings(combined)
    clarifications = _extract_spelling_clarifications(combined)

    name_spelling = _best_spelling_for_name(spellings, name)
    spelling_note = ""
    if not name_spelling and spellings:
        name_spelling = spellings[-1]
        spelling_note = "Spelling offered on camera (may refer to another name in this clip)"

    name_spelling = _apply_clarification_to_spelling(name_spelling, clarifications)

    display_name = name
    if name_spelling and _should_apply_spelling_as_name(name_spelling, name):
        display_name = _format_name_from_spelling(name_spelling)
    elif clarifications:
        display_name = _name_from_clarification(name, clarifications)

    if clarifications and _name_from_clarification(name, clarifications) == display_name:
        clarification_note = "; ".join(clarifications)
        spelling_note = (
            f"{spelling_note}; {clarification_note}".strip("; ")
            if spelling_note
            else clarification_note
        )

    pronunciation = ""
    for pattern in PRONUNCIATION_PATTERNS:
        match = pattern.search(combined)
        if match:
            pronunciation = match.group(1).strip()
            break
    if not pronunciation and name_spelling and _spelling_relates_to_name(name_spelling, name):
        pronunciation = _pronunciation_from_spelling(name_spelling)
    elif not pronunciation and name_spelling:
        pronunciation = _pronunciation_from_spelling(name_spelling)

    result = {
        "location": location,
        "name_spelling": name_spelling,
        "spelling_note": spelling_note,
        "pronunciation": pronunciation,
        "name": display_name,
    }
    if display_name != name:
        result["name_heard"] = name
    return result


def enrich_detection_record(record: dict) -> dict:
    heard_names = extract_spoken_names(record.get("transcript", ""))
    base_name = heard_names[0][0] if heard_names else record.get("name", "")
    details = extract_intro_details(
        record.get("transcript", ""),
        base_name,
        record.get("context_transcript", ""),
    )
    enriched = {**record, **details}
    return enriched


def refresh_detection_details(output_dir: Path) -> List[dict]:
    detections_path = output_dir / "all_detected_names.json"
    if not detections_path.exists():
        return []

    detections = json.loads(detections_path.read_text(encoding="utf-8"))
    faces_path = output_dir / "faces.json"
    context_by_timestamp: Dict[float, str] = {}
    if faces_path.exists():
        faces_payload = json.loads(faces_path.read_text(encoding="utf-8"))
        video_path = Path(faces_payload.get("video", ""))
        if video_path.exists():
            whisper_model = _load_whisper_model()
            wav_path = _wav_for_clip(video_path, output_dir, is_main=True)
            context_by_timestamp = _spelling_context_by_timestamp(
                whisper_model,
                wav_path,
                [float(item["timestamp"]) for item in detections],
            )

    enriched = []
    for item in detections:
        timestamp = float(item["timestamp"])
        item["context_transcript"] = context_by_timestamp.get(timestamp, "")
        heard_names = extract_spoken_names(item.get("transcript", ""))
        base_name = heard_names[0][0] if heard_names else item.get("name_heard") or item.get("name", "")
        item["name"] = base_name
        enriched.append(enrich_detection_record(item))

    detections_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return enriched


def _collect_whisper_segments(
    whisper_model,
    wav_path: Path,
    scan_duration: Optional[float],
) -> List[dict]:
    duration = scan_duration or _audio_duration(wav_path)
    whisper_segments, _ = whisper_model.transcribe(
        str(wav_path),
        beam_size=1,
        vad_filter=True,
    )
    segments: List[dict] = []
    for whisper_segment in whisper_segments:
        if whisper_segment.start > duration:
            break
        transcript = whisper_segment.text.strip()
        if not transcript:
            continue
        segments.append(
            {
                "start": float(whisper_segment.start),
                "end": float(whisper_segment.end),
                "text": transcript,
            }
        )
    return segments


def _context_transcript_for_timestamp(
    segments: List[dict],
    timestamp: float,
    window_after_s: float = SPELLING_CONTEXT_WINDOW_S,
    next_intro_timestamp: Optional[float] = None,
) -> str:
    end_time = timestamp + window_after_s
    if next_intro_timestamp is not None:
        end_time = min(end_time, next_intro_timestamp - 0.5)

    parts: List[str] = []
    seen: set[str] = set()
    for segment in segments:
        if segment["start"] < timestamp - 2.0:
            continue
        if segment["start"] > end_time:
            break
        text = segment["text"]
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return " ".join(parts)


def _spelling_context_by_timestamp(
    whisper_model,
    wav_path: Path,
    timestamps: List[float],
) -> Dict[float, str]:
    sorted_ts = sorted(timestamps)
    contexts: Dict[float, str] = {}
    for index, timestamp in enumerate(sorted_ts):
        next_intro = sorted_ts[index + 1] if index + 1 < len(sorted_ts) else None
        segment_wav = wav_path.parent / f"_spelling_ctx_{int(timestamp)}.wav"
        window_end = timestamp + SPELLING_CONTEXT_WINDOW_S
        if next_intro is not None:
            window_end = min(window_end, next_intro - 0.5)
        try:
            _extract_wav_segment(
                wav_path,
                max(timestamp - 2.0, 0.0),
                max(window_end, timestamp + 4.0),
                segment_wav,
            )
            segments = _collect_whisper_segments(whisper_model, segment_wav, scan_duration=None)
            offset = max(timestamp - 2.0, 0.0)
            adjusted = [
                {
                    "start": segment["start"] + offset,
                    "end": segment["end"] + offset,
                    "text": segment["text"],
                }
                for segment in segments
            ]
            contexts[timestamp] = _context_transcript_for_timestamp(
                adjusted,
                timestamp,
                next_intro_timestamp=next_intro,
            )
        finally:
            segment_wav.unlink(missing_ok=True)
    return contexts


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


MIN_FRAME_BRIGHTNESS = 12.0
SEARCH_OFFSETS_S = [
    0, 0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0,
    10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0,
]


def _frame_brightness(frame) -> float:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_usable_thumbnail(path: Path, min_brightness: float = 20.0) -> bool:
    if not path.exists():
        return False
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        if width > 900 and height > 500:
            return False
        pixels = list(rgb.getdata())
        avg = sum(sum(channel) / 3.0 for channel in pixels) / max(len(pixels), 1)
        return avg >= min_brightness and max(width, height) >= 80


def _read_video_frame(video_path: Path, timestamp: float):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def _embedding_face_id(observation: FaceObservation, centroids: Dict[str, np.ndarray]) -> str:
    obs_vec = np.array(observation.embedding, dtype=np.float32)
    best_id = ""
    best_dist = float("inf")
    for cluster_id, centroid in centroids.items():
        dist = float(np.linalg.norm(obs_vec - np.array(centroid, dtype=np.float32)))
        if dist < best_dist:
            best_dist = dist
            best_id = cluster_id
    return best_id


def _obs_matches_face_id(
    observation: FaceObservation,
    face_id: str,
    centroids: Dict[str, np.ndarray],
) -> bool:
    if not face_id:
        return False
    return _embedding_face_id(observation, centroids) == face_id


def _thumbnail_exclude_attempts(assigned_faces: set[str], chosen_face: str) -> List[set[str]]:
    if chosen_face:
        return [set()]
    attempts: List[set[str]] = []
    if assigned_faces:
        attempts.append(set(assigned_faces))
        for face_id in sorted(assigned_faces, key=lambda item: int(item.split("_")[1]), reverse=True):
            attempts.append({face_id})
    attempts.append(set())
    unique: List[set[str]] = []
    seen: set[tuple[str, ...]] = set()
    for attempt in attempts:
        key = tuple(sorted(attempt))
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return unique


def _validate_saved_crop(
    crop_path: Path,
    expected_face_id: str,
    centroids: Dict[str, np.ndarray],
    analyzer: FaceAnalyzer,
) -> bool:
    import cv2

    image = cv2.imread(str(crop_path))
    if image is None:
        return False
    observations = analyzer.analyze_frame(0.0, image)
    if not observations:
        return False
    return _obs_matches_face_id(observations[0], expected_face_id, centroids)


def _save_face_crop(frame, bbox: Tuple[int, int, int, int], output_path: Path) -> bool:
    import cv2
    from PIL import Image

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output_path, quality=90)
    return True


def _pick_observation(
    observations: List[FaceObservation],
    centroids: Dict[str, np.ndarray],
    preferred_face_id: str = "",
    exclude_face_ids: Optional[set[str]] = None,
) -> Tuple[FaceObservation, str]:
    ranked: List[Tuple[FaceObservation, str, float]] = []
    for obs in observations:
        face_id = _nearest_cluster(obs, centroids)
        if not face_id or (exclude_face_ids and face_id in exclude_face_ids):
            continue
        x1, y1, x2, y2 = obs.bbox
        area = float((x2 - x1) * (y2 - y1))
        score = obs.lip_openness * 1000.0 + area
        if face_id == preferred_face_id:
            score += 100000.0
        ranked.append((obs, face_id, score))
    if not ranked:
        raise ValueError("No observations to pick from")
    ranked.sort(key=lambda item: item[2], reverse=True)
    best_obs, best_face, _ = ranked[0]
    return best_obs, best_face


def _find_face_capture(
    video_path: Path,
    timestamp: float,
    centroids: Dict[str, np.ndarray],
    analyzer: FaceAnalyzer,
    preferred_face_id: str = "",
    exclude_face_ids: Optional[set[str]] = None,
) -> Tuple[float, object, FaceObservation, str]:
    for offset in SEARCH_OFFSETS_S:
        capture_time = timestamp + offset
        if capture_time < 0:
            continue
        frame = _read_video_frame(video_path, capture_time)
        if frame is None or _frame_brightness(frame) < MIN_FRAME_BRIGHTNESS:
            continue
        observations = analyzer.analyze_frame(capture_time, frame)
        if preferred_face_id:
            observations = [
                obs
                for obs in observations
                if _nearest_cluster(obs, centroids) == preferred_face_id
            ]
        elif exclude_face_ids:
            observations = [
                obs
                for obs in observations
                if _nearest_cluster(obs, centroids) not in exclude_face_ids
            ]
        if not observations:
            continue
        best_obs, face_id = _pick_observation(
            observations,
            centroids,
            preferred_face_id=preferred_face_id,
            exclude_face_ids=exclude_face_ids,
        )
        return capture_time, frame, best_obs, face_id
    raise ValueError(f"No usable face frame found near {timestamp:.1f}s in {video_path}")


def _save_wide_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    import cv2
    from PIL import Image

    for offset in SEARCH_OFFSETS_S:
        capture_time = timestamp + offset
        if capture_time < 0:
            continue
        frame = _read_video_frame(video_path, capture_time)
        if frame is None or _frame_brightness(frame) < MIN_FRAME_BRIGHTNESS:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(output_path, quality=90)
        return True
    return False


def _copy_cluster_thumbnail(output_dir: Path, face_id: str, output_path: Path) -> bool:
    if not face_id:
        return False
    source = output_dir / "face_thumbnails" / f"{face_id}.jpg"
    if not source.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(source.read_bytes())
    return True


def _save_intro_thumbnail(
    video_path: Path,
    timestamp: float,
    output_path: Path,
    centroids: Dict[str, np.ndarray],
    analyzer: FaceAnalyzer,
    preferred_face_id: str = "",
    output_dir: Optional[Path] = None,
    exclude_face_ids: Optional[set[str]] = None,
    allow_cluster_fallback: bool = True,
) -> Tuple[str, Optional[str]]:
    exclude_attempts = [exclude_face_ids or set()]
    if not preferred_face_id and exclude_face_ids is not None:
        exclude_attempts = _thumbnail_exclude_attempts(exclude_face_ids, chosen_face="")

    for attempt_exclude in exclude_attempts:
        try:
            _, frame, best_obs, face_id = _find_face_capture(
                video_path,
                timestamp,
                centroids,
                analyzer,
                preferred_face_id=preferred_face_id,
                exclude_face_ids=attempt_exclude or None,
            )
            expected_face = preferred_face_id or face_id
            if preferred_face_id and not _obs_matches_face_id(best_obs, preferred_face_id, centroids):
                if (
                    allow_cluster_fallback
                    and output_dir
                    and _copy_cluster_thumbnail(output_dir, preferred_face_id, output_path)
                ):
                    return preferred_face_id, str(output_path)
                continue
            if not _save_face_crop(frame, best_obs.bbox, output_path):
                continue
            if preferred_face_id and not _validate_saved_crop(
                output_path, preferred_face_id, centroids, analyzer
            ):
                if (
                    allow_cluster_fallback
                    and output_dir
                    and _copy_cluster_thumbnail(output_dir, preferred_face_id, output_path)
                ):
                    return preferred_face_id, str(output_path)
                continue
            return expected_face, str(output_path)
        except ValueError:
            continue

    if (
        allow_cluster_fallback
        and preferred_face_id
        and output_dir
        and _copy_cluster_thumbnail(output_dir, preferred_face_id, output_path)
    ):
        return preferred_face_id, str(output_path)

    return "", None


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
    segments = _collect_whisper_segments(whisper_model, wav_path, scan_duration)
    duration = scan_duration or _audio_duration(wav_path)
    raw_events: List[dict] = []
    for segment in segments:
        if segment["start"] > duration:
            break
        transcript = segment["text"]
        timestamp = segment["start"] + 0.35
        context_transcript = _context_transcript_for_timestamp(segments, timestamp)
        for name, base_confidence in extract_spoken_names(transcript):
            confidence = base_confidence
            if timestamp < 120:
                confidence += 5
            raw_events.append(
                {
                    "name": name,
                    "timestamp": timestamp,
                    "transcript": transcript,
                    "context_transcript": context_transcript,
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

            multiple_faces_visible = len(ranked_faces) > 1 and ranked_faces[0][0] in assigned_faces
            allow_cluster_fallback = lip_score <= 0 or multiple_faces_visible

            clip_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", _clip_label(clip_path)).strip("._") or "clip"
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", event["name"]).strip("._") or "name"
            thumb_path = intro_thumb_dir / f"{safe_name}_{int(event['timestamp'])}_{clip_tag}.jpg"
            thumb_face, _ = _save_intro_thumbnail(
                clip_path,
                event["timestamp"],
                thumb_path,
                centroids,
                analyzer,
                preferred_face_id=chosen_face,
                output_dir=output_dir,
                exclude_face_ids=assigned_faces if not chosen_face else set(),
                allow_cluster_fallback=allow_cluster_fallback,
            )
            if not chosen_face and thumb_face and thumb_face not in assigned_faces:
                chosen_face = thumb_face

            detection = enrich_detection_record(
                {
                    "name": event["name"],
                    "timestamp": event["timestamp"],
                    "transcript": event["transcript"],
                    "context_transcript": event.get("context_transcript", ""),
                    "confidence": event["confidence"],
                    "face_id": chosen_face,
                    "lip_score": lip_score,
                    "source_video": event["source_video"],
                    "clip_label": _clip_label(clip_path),
                    "thumbnail_file": str(Path("name_intro_thumbnails") / thumb_path.name),
                }
            )
            all_detections.append(detection)

            if chosen_face and chosen_face not in face_labels:
                assigned_faces.add(chosen_face)
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


def resolve_display_thumbnail(output_dir: Path, row: dict) -> Path:
    intro_path = output_dir / row.get("thumbnail_file", "")
    if is_usable_thumbnail(intro_path):
        return intro_path

    face_id = row.get("face_id", "")
    cluster_path = output_dir / "face_thumbnails" / f"{face_id}.jpg"
    if face_id and cluster_path.exists():
        return cluster_path

    return intro_path


def refresh_intro_thumbnails(output_dir: Path) -> List[dict]:
    detections_path = output_dir / "all_detected_names.json"
    if not detections_path.exists():
        return []

    faces_path = output_dir / "faces.json"
    if not faces_path.exists():
        return refresh_detection_details(output_dir)

    faces_payload = json.loads(faces_path.read_text(encoding="utf-8"))
    video_path = Path(faces_payload["video"])
    observations = [FaceObservation(**item) for item in faces_payload["observations"]]
    centroids = _cluster_centroids(observations)
    detections = [enrich_detection_record(item) for item in json.loads(detections_path.read_text(encoding="utf-8"))]

    analyzer = FaceAnalyzer()
    try:
        for detection in detections:
            thumb_path = output_dir / detection.get("thumbnail_file", "")
            if is_usable_thumbnail(thumb_path):
                continue

            clip_path = Path(detection.get("source_video") or video_path)
            preferred_face = detection.get("face_id", "")
            lip_score = float(detection.get("lip_score", 0.0))
            allow_cluster_fallback = lip_score <= 0 or preferred_face != "face_0"
            _save_intro_thumbnail(
                clip_path,
                float(detection["timestamp"]),
                thumb_path,
                centroids,
                analyzer,
                preferred_face_id=preferred_face,
                output_dir=output_dir,
                allow_cluster_fallback=allow_cluster_fallback,
            )
    finally:
        analyzer.close()

    detections_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return detections
