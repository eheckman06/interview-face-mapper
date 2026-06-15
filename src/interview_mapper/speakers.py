from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import medfilt
from sklearn.cluster import AgglomerativeClustering

from .video_io import extract_audio


@dataclass
class SpeakerSegment:
    speaker_id: str
    start: float
    end: float


def _windowed_embeddings(wav: np.ndarray, sample_rate: int, window_s: float = 1.5, hop_s: float = 0.75):
    encoder = VoiceEncoder()
    window = int(window_s * sample_rate)
    hop = int(hop_s * sample_rate)
    embeddings = []
    timestamps = []
    for start in range(0, max(len(wav) - window, 1), hop):
        chunk = wav[start : start + window]
        if len(chunk) < window // 2:
            continue
        embed = encoder.embed_utterance(chunk)
        embeddings.append(embed)
        center = (start + window / 2) / sample_rate
        timestamps.append(center)
    return np.array(embeddings), np.array(timestamps)


def _segments_from_labels(labels: np.ndarray, timestamps: np.ndarray, min_segment_s: float = 0.8) -> List[SpeakerSegment]:
    if len(labels) == 0:
        return []

    segments: List[SpeakerSegment] = []
    current_label = labels[0]
    start_ts = timestamps[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current_label:
            end_ts = timestamps[idx]
            if end_ts - start_ts >= min_segment_s:
                segments.append(
                    SpeakerSegment(
                        speaker_id=f"speaker_{int(current_label)}",
                        start=max(start_ts - 0.75, 0.0),
                        end=end_ts + 0.75,
                    )
                )
            current_label = labels[idx]
            start_ts = timestamps[idx]

    end_ts = timestamps[-1]
    if end_ts - start_ts >= min_segment_s:
        segments.append(
            SpeakerSegment(
                speaker_id=f"speaker_{int(current_label)}",
                start=max(start_ts - 0.75, 0.0),
                end=end_ts + 0.75,
            )
        )
    return segments


def diarize_speakers(
    video_path: Path,
    output_dir: Path,
    max_speakers: int = 6,
    max_duration_s: Optional[float] = None,
) -> List[SpeakerSegment]:
    wav_path = output_dir / "audio.wav"
    extract_audio(video_path, wav_path, max_duration_s=max_duration_s)
    wav = preprocess_wav(wav_path)
    sample_rate = 16000
    embeddings, timestamps = _windowed_embeddings(wav, sample_rate)
    if len(embeddings) == 0:
        return []

    speaker_count = min(max_speakers, max(2, len(embeddings) // 4))
    clustering = AgglomerativeClustering(n_clusters=speaker_count)
    labels = clustering.fit_predict(embeddings)

    if len(labels) > 5:
        labels = medfilt(labels, kernel_size=5)

    segments = _segments_from_labels(labels, timestamps)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "speakers.json").write_text(
        json.dumps({"segments": [asdict(seg) for seg in segments]}, indent=2)
    )
    return segments
