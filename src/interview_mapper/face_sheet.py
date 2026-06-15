from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


def _speaker_for_face(correlation: dict) -> Dict[str, dict]:
    mapping = {}
    for item in correlation.get("speaker_face_mappings", []):
        mapping[item["face_id"]] = item
    return mapping


def _thumb_path(output_dir: Path, row: dict) -> Path:
    return output_dir / row["thumbnail_file"]


def _image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _all_detection_rows(output_dir: Path) -> List[dict]:
    detections_path = output_dir / "all_detected_names.json"
    if not detections_path.exists():
        return []
    rows: List[dict] = []
    for item in json.loads(detections_path.read_text(encoding="utf-8")):
        thumb_rel = item.get("thumbnail_file", "")
        rows.append(
            {
                "face_id": item.get("face_id", ""),
                "name": item.get("name", ""),
                "timestamp": round(float(item.get("timestamp", 0.0)), 1),
                "speaker_id": "",
                "match_confidence": round(float(item.get("confidence", 0.0)), 1),
                "evidence_segments": 0,
                "sample_count": 0,
                "thumbnail_file": thumb_rel,
                "transcript": item.get("transcript", ""),
                "clip_label": item.get("clip_label", ""),
            }
        )
    return rows


def _rows_from_payload(
    output_dir: Path,
    faces_payload: dict,
    correlation: Optional[dict] = None,
    labels: Optional[Dict[str, str]] = None,
) -> List[dict]:
    speaker_by_face = _speaker_for_face(correlation or {})
    rows: List[dict] = []

    for cluster in faces_payload.get("face_clusters", []):
        face_id = cluster["cluster_id"]
        speaker = speaker_by_face.get(face_id, {})
        name = (labels or {}).get(face_id, "")
        if not name:
            name = "Name not detected"
        thumb_file = cluster.get("thumbnail_path", "")
        if "/" in thumb_file:
            thumb_file = Path(thumb_file).name
        thumb_rel = f"face_thumbnails/{thumb_file}" if thumb_file else ""
        rows.append(
            {
                "face_id": face_id,
                "name": name,
                "speaker_id": speaker.get("speaker_id", ""),
                "match_confidence": round(float(speaker.get("confidence", 0.0)), 3),
                "evidence_segments": int(speaker.get("evidence_segments", 0)),
                "sample_count": int(cluster.get("sample_count", 0)),
                "thumbnail_file": thumb_rel,
            }
        )

    rows.sort(key=lambda item: item["face_id"])
    return rows


def _format_timecode(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _intro_roster_rows(output_dir: Path) -> List[dict]:
    rows = _all_detection_rows(output_dir)
    for row in rows:
        row["timecode"] = _format_timecode(row["timestamp"])
    rows.sort(key=lambda item: item["timestamp"])
    return rows


def _html_for_intro_roster(rows: List[dict], output_dir: Path) -> str:
    cards = []
    for row in rows:
        thumb_path = output_dir / row["thumbnail_file"]
        data_uri = _image_data_uri(thumb_path)
        img_tag = (
            f'<img src="{data_uri}" alt="{row["name"]}" />'
            if data_uri
            else '<div class="missing">No image</div>'
        )
        clip_line = (
            f'<div class="clip">{row.get("clip_label") or "main clip"}</div>'
            if row.get("clip_label")
            else ""
        )
        face_line = (
            f'<div class="face-id">{row["face_id"]}</div>'
            if row.get("face_id")
            else '<div class="face-id unmapped">Face not mapped</div>'
        )
        cards.append(
            f"""
            <article class="card">
              <div class="photo">{img_tag}</div>
              <div class="timecode">{row["timecode"]}</div>
              <div class="name">{row["name"]}</div>
              {clip_line}
              {face_line}
              <div class="quote">{row.get("transcript", "")}</div>
            </article>
            """
        )

    empty_state = (
        '<p class="empty">No on-camera introductions were detected yet.</p>'
        if not cards
        else f'<div class="roster">{"".join(cards)}</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>All Introductions</title>
  <style>
    body {{
      font-family: Inter, system-ui, sans-serif;
      margin: 24px;
      color: #111;
      background: #f7f7f7;
    }}
    h1 {{ margin-top: 0; font-size: 32px; }}
    .subtitle {{ color: #555; margin-bottom: 24px; max-width: 720px; line-height: 1.5; }}
    .roster {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 14px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      page-break-inside: avoid;
    }}
    .photo img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      border-radius: 10px;
      background: #eee;
      display: block;
    }}
    .missing {{
      width: 100%;
      aspect-ratio: 1;
      background: #eee;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #777;
    }}
    .timecode {{
      display: inline-block;
      align-self: flex-start;
      background: #111;
      color: #fff;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.03em;
      padding: 5px 10px;
      border-radius: 999px;
    }}
    .name {{
      font-size: 26px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .clip {{
      color: #666;
      font-size: 13px;
    }}
    .face-id {{
      color: #444;
      font-size: 13px;
      font-weight: 600;
    }}
    .face-id.unmapped {{
      color: #9a6700;
    }}
    .quote {{
      color: #666;
      font-size: 12px;
      line-height: 1.4;
      border-top: 1px solid #eee;
      padding-top: 8px;
      margin-top: 4px;
    }}
    @media print {{
      body {{ background: #fff; margin: 12px; }}
      .card {{ border-color: #999; }}
    }}
  </style>
</head>
<body>
  <h1>All On-Camera Introductions</h1>
  <p class="subtitle">Every detected name with the face on screen at the moment they introduced themselves, sorted by time.</p>
  {empty_state}
</body>
</html>
"""


def _generate_intro_roster_image(output_dir: Path, rows: List[dict]) -> Path:
    tile = 240
    label_height = 72
    padding = 20
    columns = min(max(len(rows), 1), 3)
    row_count = (len(rows) + columns - 1) // columns
    width = columns * tile + (columns + 1) * padding
    height = row_count * (tile + label_height) + (row_count + 1) * padding
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(rows):
        col = index % columns
        row_idx = index // columns
        x = padding + col * (tile + padding)
        y = padding + row_idx * (tile + label_height + padding)

        thumb_path = output_dir / row["thumbnail_file"]
        if thumb_path.exists():
            image = Image.open(thumb_path).convert("RGB")
            image = image.resize((tile, tile))
            canvas.paste(image, (x, y))
        else:
            draw.rectangle((x, y, x + tile, y + tile), fill=(220, 220, 220))

        draw.rectangle((x, y, x + tile, y + tile), outline=(60, 60, 60), width=2)
        timecode = row.get("timecode") or _format_timecode(row["timestamp"])
        draw.text((x, y + tile + 8), timecode, fill=(20, 20, 20))
        draw.text((x, y + tile + 30), row["name"], fill=(20, 20, 20))
        clip_label = row.get("clip_label") or row.get("face_id") or ""
        if clip_label:
            draw.text((x, y + tile + 52), clip_label, fill=(100, 100, 100))

    roster_path = output_dir / "intro_roster.jpg"
    canvas.save(roster_path, quality=92)
    return roster_path


def export_intro_roster(output_dir: Path) -> dict:
    rows = _intro_roster_rows(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "intro_roster.html"
    json_path = output_dir / "intro_roster.json"
    csv_path = output_dir / "intro_roster.csv"

    html_path.write_text(_html_for_intro_roster(rows, output_dir), encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "timestamp",
                "timecode",
                "clip_label",
                "face_id",
                "confidence",
                "transcript",
                "thumbnail_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "timestamp": row["timestamp"],
                    "timecode": row["timecode"],
                    "clip_label": row.get("clip_label", ""),
                    "face_id": row.get("face_id", ""),
                    "confidence": row.get("match_confidence", ""),
                    "transcript": row.get("transcript", ""),
                    "thumbnail_file": row.get("thumbnail_file", ""),
                }
            )

    image_path = _generate_intro_roster_image(output_dir, rows) if rows else None

    return {
        "intro_roster_html": str(html_path),
        "intro_roster_json": str(json_path),
        "intro_roster_csv": str(csv_path),
        "intro_roster_image": str(image_path) if image_path else "",
        "rows": rows,
    }


def _card_html(row: dict, output_dir: Path, extra_meta: Optional[List[str]] = None) -> str:
    thumb_path = output_dir / row["thumbnail_file"]
    data_uri = _image_data_uri(thumb_path)
    img_tag = (
        f'<img src="{data_uri}" alt="{row["name"]}" />'
        if data_uri
        else '<div class="missing">No image</div>'
    )
    meta_lines = extra_meta or [
        f"Face ID: {row['face_id']}",
        f"Speaker: {row['speaker_id'] or '—'}",
        f"Match confidence: {row['match_confidence']}",
        f"Samples seen: {row['sample_count']}",
    ]
    meta_html = "".join(f'<div class="meta">{line}</div>' for line in meta_lines)
    return f"""
            <div class="row">
              {img_tag}
              <div class="details">
                <div class="name">{row['name']}</div>
                {meta_html}
              </div>
            </div>
            """


def _html_for_rows(
    rows: List[dict],
    output_dir: Path,
    title: str,
    all_detections: Optional[List[dict]] = None,
) -> str:
    cards = [_card_html(row, output_dir) for row in rows]

    detection_section = ""
    if all_detections:
        detection_cards = []
        for row in all_detections:
            timecode = _format_timecode(row["timestamp"])
            detection_cards.append(
                _card_html(
                    row,
                    output_dir,
                    extra_meta=[
                        f"Intro at {timecode} ({row['timestamp']}s)",
                        f"Clip: {row.get('clip_label') or '—'}",
                        f"Face ID: {row['face_id'] or '—'}",
                        f"Confidence: {row['match_confidence']}",
                        f"Quote: {row.get('transcript', '')[:120]}",
                    ],
                )
            )
        detection_section = f"""
  <h2>All on-camera introductions</h2>
  <p>Every name heard in a self-introduction, with a frame from that moment.</p>
  <div class="sheet">
    {''.join(detection_cards)}
  </div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Inter, system-ui, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-top: 0; }}
    h2 {{ margin-top: 32px; }}
    .sheet {{ display: grid; gap: 16px; }}
    .row {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 16px;
      align-items: center;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 14px;
      page-break-inside: avoid;
    }}
    img {{ width: 160px; height: 160px; object-fit: cover; border-radius: 10px; background: #eee; }}
    .missing {{ width: 160px; height: 160px; background: #eee; border-radius: 10px; display:flex; align-items:center; justify-content:center; color:#777; }}
    .name {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .meta {{ color: #555; font-size: 14px; margin-top: 2px; }}
    @media print {{
      body {{ margin: 12px; }}
      .row {{ border-color: #999; }}
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Every detected face with its assigned name.</p>
  <div class="sheet">
    {''.join(cards)}
  </div>
  {detection_section}
</body>
</html>
"""


def _generate_cast_sheet_image(
    output_dir: Path,
    rows: List[dict],
    output_name: str = "cast_sheet.jpg",
) -> Path:
    tile = 220
    padding = 24
    columns = min(max(len(rows), 1), 4)
    rows_count = (len(rows) + columns - 1) // columns
    width = columns * tile + (columns + 1) * padding
    height = rows_count * (tile + 56) + (rows_count + 1) * padding
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(rows):
        col = index % columns
        row_idx = index // columns
        x = padding + col * (tile + padding)
        y = padding + row_idx * (tile + 56 + padding)

        thumb_path = output_dir / row["thumbnail_file"]
        if thumb_path.exists():
            image = Image.open(thumb_path).convert("RGB")
            image = image.resize((tile, tile))
            canvas.paste(image, (x, y))
        else:
            draw.rectangle((x, y, x + tile, y + tile), fill=(220, 220, 220))

        draw.rectangle((x, y, x + tile, y + tile), outline=(80, 80, 80), width=2)
        name = row["name"]
        draw.text((x, y + tile + 8), name, fill=(20, 20, 20))
        draw.text((x, y + tile + 30), row["face_id"], fill=(100, 100, 100))

    cast_path = output_dir / output_name
    canvas.save(cast_path, quality=92)
    return cast_path


def export_face_name_sheet(
    output_dir: Path,
    labels: Optional[Dict[str, str]] = None,
) -> dict:
    faces_path = output_dir / "faces.json"
    if not faces_path.exists():
        raise FileNotFoundError("Missing faces.json. Run analysis first.")

    faces_payload = json.loads(faces_path.read_text(encoding="utf-8"))
    correlation = {}
    correlation_path = output_dir / "correlation.json"
    if correlation_path.exists():
        correlation = json.loads(correlation_path.read_text(encoding="utf-8"))

    if labels is None and (output_dir / "labels.json").exists():
        labels = json.loads((output_dir / "labels.json").read_text(encoding="utf-8"))

    rows = _rows_from_payload(output_dir, faces_payload, correlation=correlation, labels=labels)
    all_detections = _all_detection_rows(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "face_name_sheet.json"
    csv_path = output_dir / "face_name_sheet.csv"
    html_path = output_dir / "face_name_sheet.html"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "face_id",
                "name",
                "speaker_id",
                "match_confidence",
                "evidence_segments",
                "sample_count",
                "thumbnail_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    html_path.write_text(
        _html_for_rows(rows, output_dir, "Interview Face Name Sheet", all_detections=all_detections),
        encoding="utf-8",
    )
    cast_path = _generate_cast_sheet_image(output_dir, rows)
    intro_roster = export_intro_roster(output_dir)
    intro_cast_path = intro_roster.get("intro_roster_image") or None
    if not intro_cast_path and all_detections:
        intro_cast_path = str(
            _generate_cast_sheet_image(
                output_dir,
                all_detections,
                output_name="intro_cast_sheet.jpg",
            )
        )

    return {
        "face_name_sheet_json": str(json_path),
        "face_name_sheet_csv": str(csv_path),
        "face_name_sheet_html": str(html_path),
        "cast_sheet": str(cast_path),
        "intro_cast_sheet": intro_cast_path or "",
        "intro_roster_html": intro_roster["intro_roster_html"],
        "intro_roster_csv": intro_roster["intro_roster_csv"],
        "intro_roster_image": intro_roster.get("intro_roster_image", ""),
        "all_detections": all_detections,
        "rows": rows,
    }
