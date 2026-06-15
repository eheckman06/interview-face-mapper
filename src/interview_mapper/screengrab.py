from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .faces import FaceAnalyzer, FaceObservation
from .video_io import iter_frames


def _load_observations(output_dir: Path) -> Tuple[Path, List[FaceObservation]]:
    faces_path = output_dir / "faces.json"
    if not faces_path.exists():
        raise FileNotFoundError("Missing faces.json. Run `interview-mapper analyze` first.")

    payload = json.loads(faces_path.read_text(encoding="utf-8"))
    video_path = Path(payload["video"])
    observations = [FaceObservation(**item) for item in payload["observations"]]
    return video_path, observations


def _labels_for_output(output_dir: Path, labels: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if labels is not None:
        return labels
    labels_path = output_dir / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))
    return {}


def _best_timestamp(observations: List[FaceObservation]) -> float:
    by_time: Dict[float, List[FaceObservation]] = defaultdict(list)
    for obs in observations:
        if obs.cluster_id:
            by_time[obs.timestamp].append(obs)

    if not by_time:
        return 0.0

    best_ts = 0.0
    best_score = -1.0
    for timestamp, items in by_time.items():
        cluster_ids = {item.cluster_id for item in items}
        score = len(cluster_ids) * 10 + len(items)
        if score > best_score:
            best_score = score
            best_ts = timestamp
    return best_ts


def _observations_near(
    observations: List[FaceObservation],
    timestamp: float,
    tolerance_s: float = 0.6,
) -> List[FaceObservation]:
    nearby = [
        obs
        for obs in observations
        if obs.cluster_id and abs(obs.timestamp - timestamp) <= tolerance_s
    ]
    if not nearby:
        return []

    by_cluster: Dict[str, FaceObservation] = {}
    for obs in nearby:
        current = by_cluster.get(obs.cluster_id)
        if current is None or abs(obs.timestamp - timestamp) < abs(current.timestamp - timestamp):
            by_cluster[obs.cluster_id] = obs
    return list(by_cluster.values())


def _draw_label_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = bbox
    color = (79, 124, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    top = max(y1 - text_h - baseline - pad * 2, 0)
    bottom = top + text_h + baseline + pad * 2
    right = min(x1 + text_w + pad * 2, frame.shape[1] - 1)

    cv2.rectangle(frame, (x1, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + pad, bottom - baseline - pad),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def generate_named_screengrab(
    output_dir: Path,
    labels: Optional[Dict[str, str]] = None,
    timestamp: Optional[float] = None,
    output_name: str = "named_screengrab.jpg",
) -> Path:
    video_path, observations = _load_observations(output_dir)
    label_map = _labels_for_output(output_dir, labels)
    chosen_ts = timestamp if timestamp is not None else _best_timestamp(observations)
    frame_obs = _observations_near(observations, chosen_ts)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, chosen_ts * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {chosen_ts:.2f}s from {video_path}")

    if not frame_obs:
        analyzer = FaceAnalyzer()
        try:
            live_obs = analyzer.analyze_frame(chosen_ts, frame)
            cluster_map = _cluster_centroids(observations)
            for obs in live_obs:
                cluster_id = _nearest_cluster(obs, cluster_map)
                if cluster_id:
                    obs.cluster_id = cluster_id
            frame_obs = live_obs
        finally:
            analyzer.close()

    for obs in frame_obs:
        name = label_map.get(obs.cluster_id, obs.cluster_id or "Unknown")
        _draw_label_box(frame, obs.bbox, name)

    footer = f"Frame @ {_format_timestamp(chosen_ts)}"
    cv2.putText(
        frame,
        footer,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), frame)
    meta = {
        "video": str(video_path),
        "timestamp": chosen_ts,
        "timestamp_tc": _format_timestamp(chosen_ts),
        "labels": {obs.cluster_id: label_map.get(obs.cluster_id, obs.cluster_id) for obs in frame_obs},
        "image": str(output_path),
    }
    (output_dir / "named_screengrab.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_path


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _cluster_centroids(observations: List[FaceObservation]) -> Dict[str, np.ndarray]:
    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    for obs in observations:
        if obs.cluster_id:
            buckets[obs.cluster_id].append(np.array(obs.embedding, dtype=np.float32))
    return {cluster_id: np.mean(vectors, axis=0) for cluster_id, vectors in buckets.items()}


def _nearest_cluster(observation: FaceObservation, centroids: Dict[str, np.ndarray]) -> str:
    obs_vec = np.array(observation.embedding, dtype=np.float32)
    best_id = ""
    best_dist = float("inf")
    for cluster_id, centroid in centroids.items():
        dist = 1.0 - float(
            np.dot(obs_vec, centroid) / (np.linalg.norm(obs_vec) * np.linalg.norm(centroid) + 1e-8)
        )
        if dist < best_dist:
            best_dist = dist
            best_id = cluster_id
    return best_id
