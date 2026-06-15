from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from .correlate import LowerThirdSegment


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def export_lower_thirds(output_dir: Path, lower_thirds: List[LowerThirdSegment]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "lower_thirds.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "start": item.start,
                    "end": item.end,
                    "name": item.name,
                    "speaker_id": item.speaker_id,
                    "face_id": item.face_id,
                    "confidence": round(item.confidence, 3),
                }
                for item in lower_thirds
            ],
            indent=2,
        )
    )

    csv_path = output_dir / "lower_thirds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "end", "start_tc", "end_tc", "name", "speaker_id", "face_id", "confidence"],
        )
        writer.writeheader()
        for item in lower_thirds:
            writer.writerow(
                {
                    "start": round(item.start, 3),
                    "end": round(item.end, 3),
                    "start_tc": _format_timestamp(item.start),
                    "end_tc": _format_timestamp(item.end),
                    "name": item.name,
                    "speaker_id": item.speaker_id,
                    "face_id": item.face_id,
                    "confidence": round(item.confidence, 3),
                }
            )

    edl_path = output_dir / "lower_third_markers.edl"
    lines = ["TITLE: Lower Third Markers", "FCM: NON-DROP FRAME"]
    for idx, item in enumerate(lower_thirds, start=1):
        lines.extend(
            [
                f"{idx:03d}  AX       V     C        { _format_timestamp(item.start) } { _format_timestamp(item.end) } { _format_timestamp(item.start) } { _format_timestamp(item.end) }",
                f"* FROM CLIP NAME: {item.name}",
                f"* COMMENT: speaker={item.speaker_id} face={item.face_id} confidence={item.confidence:.2f}",
            ]
        )
    edl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
