from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from .mediapipe_models import face_landmarker_model
from .video_io import get_video_metadata, iter_frames


@dataclass
class FaceObservation:
    timestamp: float
    bbox: Tuple[int, int, int, int]
    embedding: List[float]
    lip_openness: float
    cluster_id: str = ""
    source_video: str = ""


@dataclass
class FaceCluster:
    cluster_id: str
    sample_count: int
    representative_bbox: Tuple[int, int, int, int]
    thumbnail_path: str


class FaceAnalyzer:
    def __init__(self) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_landmarker_model())),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=8,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def _landmark_embedding(self, landmarks: Sequence, width: int, height: int) -> np.ndarray:
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def _lip_openness(self, landmarks: Sequence) -> float:
        upper = landmarks[13]
        lower = landmarks[14]
        return abs(upper.y - lower.y)

    def _bbox_from_landmarks(self, landmarks: Sequence, width: int, height: int) -> Tuple[int, int, int, int]:
        xs = [lm.x * width for lm in landmarks]
        ys = [lm.y * height for lm in landmarks]
        x1 = max(int(min(xs)) - 10, 0)
        y1 = max(int(min(ys)) - 10, 0)
        x2 = min(int(max(xs)) + 10, width - 1)
        y2 = min(int(max(ys)) + 10, height - 1)
        return x1, y1, x2, y2

    def analyze_frame(self, timestamp: float, frame: np.ndarray) -> List[FaceObservation]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return []

        height, width = frame.shape[:2]
        observations: List[FaceObservation] = []
        for landmarks in result.face_landmarks:
            embedding = self._landmark_embedding(landmarks, width, height)
            bbox = self._bbox_from_landmarks(landmarks, width, height)
            lip_openness = self._lip_openness(landmarks)
            observations.append(
                FaceObservation(
                    timestamp=timestamp,
                    bbox=bbox,
                    embedding=embedding.tolist(),
                    lip_openness=lip_openness,
                )
            )
        return observations


def _faces_per_timestamp(observations: List[FaceObservation]) -> Dict[float, int]:
    counts: Dict[float, int] = defaultdict(int)
    for obs in observations:
        counts[obs.timestamp] += 1
    return counts


def _estimate_person_count(observations: List[FaceObservation]) -> int:
    if not observations:
        return 0
    per_frame = _faces_per_timestamp(observations)
    max_faces = max(per_frame.values())
    if max_faces <= 1:
        return 1
    return max_faces


def _cluster_features(
    observation: FaceObservation,
    frame_width: float,
    frame_height: float,
) -> np.ndarray:
    embedding = np.array(observation.embedding, dtype=np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    x1, y1, x2, y2 = observation.bbox
    cx = ((x1 + x2) / 2) / max(frame_width, 1.0)
    cy = ((y1 + y2) / 2) / max(frame_height, 1.0)
    width = (x2 - x1) / max(frame_width, 1.0)
    spatial = np.array([cx, cy, width], dtype=np.float32) * 2.5
    return np.concatenate([embedding * 0.35, spatial])


def cluster_face_observations(
    observations: List[FaceObservation],
    frame_width: float,
    frame_height: float,
) -> Tuple[Dict[str, List[FaceObservation]], List[FaceObservation], List[int]]:
    if not observations:
        return {}, [], []

    person_count = _estimate_person_count(observations)
    matrix = np.array(
        [_cluster_features(obs, frame_width, frame_height) for obs in observations],
        dtype=np.float32,
    )

    if person_count <= 1:
        labels = np.zeros(len(observations), dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=min(person_count, len(observations)),
            linkage="ward",
        ).fit_predict(matrix)

        # If DBSCAN-style merge left too few clusters, keep agglomerative result.
        dbscan_labels = DBSCAN(eps=0.55, min_samples=2, metric="euclidean").fit_predict(matrix)
        dbscan_clusters = len({label for label in dbscan_labels if label != -1})
        if dbscan_clusters > person_count:
            labels = dbscan_labels

    clusters: Dict[str, List[FaceObservation]] = {}
    noise: List[FaceObservation] = []
    label_list: List[int] = []
    for obs, label in zip(observations, labels):
        label_list.append(int(label))
        if label == -1:
            noise.append(obs)
            continue
        key = f"face_{label}"
        clusters.setdefault(key, []).append(obs)

    # Assign stray noise points to nearest cluster centroid.
    if noise and clusters:
        centroids = {
            cluster_id: np.mean([np.array(item.embedding) for item in items], axis=0)
            for cluster_id, items in clusters.items()
        }
        for obs in noise:
            obs_vec = np.array(obs.embedding, dtype=np.float32)
            best_id = min(
                centroids.items(),
                key=lambda item: np.linalg.norm(obs_vec - item[1]),
            )[0]
            clusters[best_id].append(obs)
            label_list[observations.index(obs)] = int(best_id.split("_")[1])

    return clusters, noise, label_list


def save_thumbnails(
    clusters: Dict[str, List[FaceObservation]],
    output_dir: Path,
    fallback_video: Path,
) -> List[FaceCluster]:
    output_dir.mkdir(parents=True, exist_ok=True)
    caps: Dict[str, cv2.VideoCapture] = {}

    def _capture_for(path: Path) -> cv2.VideoCapture:
        key = str(path)
        if key not in caps:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            caps[key] = cap
        return caps[key]

    summaries: List[FaceCluster] = []
    for cluster_id, items in sorted(clusters.items()):
        best = max(items, key=lambda item: item.lip_openness)
        source_path = Path(best.source_video) if best.source_video else fallback_video
        cap = _capture_for(source_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, best.timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        x1, y1, x2, y2 = best.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        thumb_path = output_dir / f"{cluster_id}.jpg"
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(thumb_path, quality=90)
        summaries.append(
            FaceCluster(
                cluster_id=cluster_id,
                sample_count=len(items),
                representative_bbox=best.bbox,
                thumbnail_path=f"face_thumbnails/{cluster_id}.jpg",
            )
        )

    for cap in caps.values():
        cap.release()
    return summaries


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


def discover_videos(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Video path not found: {path}")
    videos = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files found in {path}")
    return videos


def _video_frame_size(video_path: Path) -> Tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1920.0, 1080.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0
    cap.release()
    return float(width), float(height)


def _sample_faces_from_video(
    analyzer: FaceAnalyzer,
    source_video: Path,
    sample_fps: float,
    max_duration_s: Optional[float],
    intro_boost: bool = True,
) -> List[FaceObservation]:
    observations: List[FaceObservation] = []
    source_key = str(source_video)

    for timestamp, frame in iter_frames(
        source_video,
        sample_fps=sample_fps,
        max_duration_s=max_duration_s,
    ):
        for obs in analyzer.analyze_frame(timestamp, frame):
            observations.append(
                FaceObservation(
                    timestamp=obs.timestamp,
                    bbox=obs.bbox,
                    embedding=obs.embedding,
                    lip_openness=obs.lip_openness,
                    source_video=source_key,
                )
            )

    if intro_boost:
        intro_limit = min(max_duration_s or 180.0, 180.0)
        for timestamp, frame in iter_frames(
            source_video,
            sample_fps=max(sample_fps, 2.5),
            max_duration_s=intro_limit,
        ):
            for obs in analyzer.analyze_frame(timestamp, frame):
                observations.append(
                    FaceObservation(
                        timestamp=obs.timestamp,
                        bbox=obs.bbox,
                        embedding=obs.embedding,
                        lip_openness=obs.lip_openness,
                        source_video=source_key,
                    )
                )
    return observations


def analyze_faces(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 2.0,
    extra_videos: Optional[List[Path]] = None,
    max_duration_s: Optional[float] = None,
    extra_max_duration_s: Optional[float] = None,
) -> Dict[str, object]:
    frame_width, frame_height = _video_frame_size(video_path)
    analyzer = FaceAnalyzer()
    observations: List[FaceObservation] = []
    sampled_videos = [video_path, *(extra_videos or [])]
    try:
        for index, source_video in enumerate(sampled_videos):
            duration_limit = max_duration_s if index == 0 else (extra_max_duration_s or max_duration_s)
            observations.extend(
                _sample_faces_from_video(
                    analyzer,
                    source_video,
                    sample_fps=sample_fps,
                    max_duration_s=duration_limit,
                    intro_boost=True,
                )
            )
    finally:
        analyzer.close()

    clusters, noise, labels = cluster_face_observations(observations, frame_width, frame_height)
    labeled_observations: List[FaceObservation] = []
    for obs, label in zip(observations, labels):
        cluster_id = f"face_{label}" if label != -1 else ""
        labeled_observations.append(
            FaceObservation(
                timestamp=obs.timestamp,
                bbox=obs.bbox,
                embedding=obs.embedding,
                lip_openness=obs.lip_openness,
                cluster_id=cluster_id,
                source_video=obs.source_video,
            )
        )

    thumb_dir = output_dir / "face_thumbnails"
    summaries = save_thumbnails(clusters, thumb_dir, fallback_video=video_path)

    payload = {
        "video": str(video_path),
        "sampled_videos": [str(item) for item in sampled_videos],
        "face_clusters": [asdict(item) for item in summaries],
        "observations": [asdict(item) for item in labeled_observations],
        "noise_count": len(noise),
        "estimated_people": len(summaries),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "faces.json").write_text(json.dumps(payload, indent=2))
    return payload


def rebuild_face_clusters(output_dir: Path) -> Dict[str, object]:
    faces_path = output_dir / "faces.json"
    if not faces_path.exists():
        raise FileNotFoundError("Missing faces.json.")

    payload = json.loads(faces_path.read_text(encoding="utf-8"))
    video_path = Path(payload["video"])
    frame_width, frame_height = _video_frame_size(video_path)

    observations = [
        FaceObservation(
            timestamp=item["timestamp"],
            bbox=tuple(item["bbox"]),
            embedding=item["embedding"],
            lip_openness=item["lip_openness"],
            source_video=item.get("source_video", ""),
        )
        for item in payload["observations"]
    ]

    clusters, noise, labels = cluster_face_observations(observations, frame_width, frame_height)
    labeled_observations: List[FaceObservation] = []
    for obs, label in zip(observations, labels):
        cluster_id = f"face_{label}" if label != -1 else ""
        labeled_observations.append(
            FaceObservation(
                timestamp=obs.timestamp,
                bbox=obs.bbox,
                embedding=obs.embedding,
                lip_openness=obs.lip_openness,
                cluster_id=cluster_id,
                source_video=obs.source_video,
            )
        )

    thumb_dir = output_dir / "face_thumbnails"
    summaries = save_thumbnails(clusters, thumb_dir, fallback_video=video_path)

    new_payload = {
        **payload,
        "face_clusters": [asdict(item) for item in summaries],
        "observations": [asdict(item) for item in labeled_observations],
        "noise_count": len(noise),
        "estimated_people": len(summaries),
    }
    faces_path.write_text(json.dumps(new_payload, indent=2), encoding="utf-8")
    return new_payload
