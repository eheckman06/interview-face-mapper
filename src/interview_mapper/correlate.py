from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .faces import FaceObservation
from .speakers import SpeakerSegment


@dataclass
class SpeakerFaceMapping:
    speaker_id: str
    face_id: str
    confidence: float
    evidence_segments: int


@dataclass
class LowerThirdSegment:
    start: float
    end: float
    speaker_id: str
    face_id: str
    name: str
    confidence: float


def _observations_for_segment(
    observations: List[FaceObservation],
    start: float,
    end: float,
) -> List[FaceObservation]:
    return [
        obs
        for obs in observations
        if obs.cluster_id and start <= obs.timestamp <= end
    ]


def _best_face_for_segment(segment_observations: List[FaceObservation]) -> str:
    lip_activity: Dict[str, List[float]] = {}
    for obs in segment_observations:
        lip_activity.setdefault(obs.cluster_id, []).append(obs.lip_openness)

    ranked = sorted(
        lip_activity.items(),
        key=lambda item: np.std(item[1]) + np.mean(item[1]),
        reverse=True,
    )
    return ranked[0][0]


def correlate_speakers_to_faces(
    video_path: Path,
    faces_payload: dict,
    speaker_segments: List[SpeakerSegment],
) -> Tuple[List[SpeakerFaceMapping], List[LowerThirdSegment]]:
    del video_path  # kept for API compatibility; correlation uses cached face samples only
    observations = [FaceObservation(**item) for item in faces_payload["observations"]]
    if not observations:
        return [], []

    speaker_scores: Dict[str, Dict[str, float]] = {}
    segment_evidence: Dict[Tuple[str, str], int] = {}

    for segment in speaker_segments:
        segment_observations = _observations_for_segment(observations, segment.start, segment.end)
        if not segment_observations:
            continue

        best_face = _best_face_for_segment(segment_observations)
        speaker_scores.setdefault(segment.speaker_id, {})
        speaker_scores[segment.speaker_id][best_face] = (
            speaker_scores[segment.speaker_id].get(best_face, 0.0) + 1.0
        )
        segment_evidence[(segment.speaker_id, best_face)] = (
            segment_evidence.get((segment.speaker_id, best_face), 0) + 1
        )

    mappings: List[SpeakerFaceMapping] = []
    for speaker_id, face_counts in speaker_scores.items():
        face_id, count = max(face_counts.items(), key=lambda item: item[1])
        total = sum(face_counts.values())
        mappings.append(
            SpeakerFaceMapping(
                speaker_id=speaker_id,
                face_id=face_id,
                confidence=count / max(total, 1),
                evidence_segments=segment_evidence.get((speaker_id, face_id), 0),
            )
        )

    lower_thirds: List[LowerThirdSegment] = []
    speaker_to_face = {item.speaker_id: item for item in mappings}
    for segment in speaker_segments:
        mapping = speaker_to_face.get(segment.speaker_id)
        if not mapping:
            continue
        lower_thirds.append(
            LowerThirdSegment(
                start=segment.start,
                end=segment.end,
                speaker_id=segment.speaker_id,
                face_id=mapping.face_id,
                name=mapping.face_id,
                confidence=mapping.confidence,
            )
        )

    return mappings, lower_thirds


def apply_labels(lower_thirds: List[LowerThirdSegment], labels: Dict[str, str]) -> List[LowerThirdSegment]:
    updated: List[LowerThirdSegment] = []
    for segment in lower_thirds:
        updated.append(
            LowerThirdSegment(
                start=segment.start,
                end=segment.end,
                speaker_id=segment.speaker_id,
                face_id=segment.face_id,
                name=labels.get(segment.face_id, segment.face_id),
                confidence=segment.confidence,
            )
        )
    return updated


def save_correlation(
    output_dir: Path,
    mappings: List[SpeakerFaceMapping],
    lower_thirds: List[LowerThirdSegment],
) -> None:
    payload = {
        "speaker_face_mappings": [asdict(item) for item in mappings],
        "lower_thirds": [asdict(item) for item in lower_thirds],
    }
    (output_dir / "correlation.json").write_text(json.dumps(payload, indent=2))
