from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .speech_names import refresh_detection_details, refresh_intro_thumbnails, resolve_display_thumbnail
from .intro_clips import export_intro_clips


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
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _thumbnail_data_uri(output_dir: Path, row: dict) -> str:
    return _image_data_uri(resolve_display_thumbnail(output_dir, row))


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
                "name_heard": item.get("name_heard", ""),
                "timestamp": round(float(item.get("timestamp", 0.0)), 1),
                "speaker_id": "",
                "match_confidence": round(float(item.get("confidence", 0.0)), 1),
                "evidence_segments": 0,
                "sample_count": 0,
                "thumbnail_file": thumb_rel,
                "transcript": item.get("transcript", ""),
                "clip_label": item.get("clip_label", ""),
                "location": item.get("location", ""),
                "name_spelling": item.get("name_spelling", ""),
                "spelling_note": item.get("spelling_note", ""),
                "pronunciation": item.get("pronunciation", ""),
                "intro_clip_file": item.get("intro_clip_file", ""),
                "intro_clip_faces_visible": int(item.get("intro_clip_faces_visible", 0) or 0),
                "intro_clip_includes_full_cast": bool(item.get("intro_clip_includes_full_cast", False)),
                "source_video": item.get("source_video", ""),
            }
        )
    return rows


def _intro_detail_lines(row: dict) -> List[str]:
    lines: List[str] = []
    if row.get("location"):
        lines.append(f"From: {row['location']}")
    if row.get("pronunciation"):
        lines.append(f"Pronunciation: {row['pronunciation']}")
    if row.get("name_spelling"):
        lines.append(f"Spelling: {row['name_spelling']}")
    if row.get("name_heard") and row.get("name_heard") != row.get("name"):
        lines.append(f"Heard as: {row['name_heard']}")
    if row.get("spelling_note"):
        lines.append(row["spelling_note"])
    return lines


def _intro_detail_html(row: dict) -> str:
    parts: List[str] = []
    if row.get("location"):
        parts.append(f'<div class="from"><span class="label">From</span> {row["location"]}</div>')
    if row.get("pronunciation"):
        parts.append(
            f'<div class="pronunciation"><span class="label">Pronunciation</span> {row["pronunciation"]}</div>'
        )
    if row.get("name_spelling"):
        parts.append(
            f'<div class="spelling"><span class="label">Spelling</span> {row["name_spelling"]}</div>'
        )
    if row.get("name_heard") and row.get("name_heard") != row.get("name"):
        parts.append(
            f'<div class="heard"><span class="label">Heard as</span> {row["name_heard"]}</div>'
        )
    if row.get("spelling_note"):
        parts.append(f'<div class="note">{row["spelling_note"]}</div>')
    faces_visible = int(row.get("intro_clip_faces_visible", 0) or 0)
    if faces_visible:
        parts.append(
            f'<div class="faces-visible"><span class="label">On camera</span> '
            f"Up to {faces_visible} people visible in clip</div>"
        )
    if row.get("intro_clip_includes_full_cast"):
        parts.append(
            '<div class="full-cast"><span class="label">Full cast</span> '
            "Clip ends with a wide shot showing everyone on camera</div>"
        )
    return "".join(parts)


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
        data_uri = _thumbnail_data_uri(output_dir, row)
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
        intro_clip = row.get("intro_clip_file", "")
        poster_attr = ""
        if data_uri:
            poster_attr = f' poster="{data_uri}"'
        video_tag = ""
        if intro_clip:
            video_tag = (
                f'<video class="intro-video" controls playsinline preload="metadata" '
                f'src="{intro_clip}"{poster_attr}></video>'
            )
        elif data_uri:
            video_tag = f'<div class="photo">{img_tag}</div>'
        thumb_tag = ""
        if intro_clip and data_uri:
            thumb_tag = f'<div class="photo-thumb">{img_tag}</div>'
        cards.append(
            f"""
            <article class="card">
              {video_tag}
              {thumb_tag}
              <div class="timecode">{row["timecode"]}</div>
              <div class="name">{row["name"]}</div>
              {_intro_detail_html(row)}
              {clip_line}
              {face_line}
              <div class="quote">"{row.get("transcript", "")}"</div>
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
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
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
    .intro-video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 10px;
      background: #000;
      display: block;
      object-fit: contain;
    }}
    .photo-thumb img {{
      width: 88px;
      height: 88px;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid #ddd;
    }}
    .photo-thumb {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .photo-thumb::before {{
      content: "Face at intro";
      color: #777;
      font-size: 12px;
      font-weight: 600;
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
    .label {{
      color: #777;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-right: 6px;
    }}
    .from, .pronunciation, .spelling {{
      color: #333;
      font-size: 14px;
      line-height: 1.35;
    }}
    .note {{
      color: #9a6700;
      font-size: 12px;
      line-height: 1.35;
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
  <p class="subtitle">Every detected on-camera introduction with face photo, playable intro clip, name, location, quote, and any spelling or pronunciation offered on camera. Play each clip to confirm the match.</p>
  {empty_state}
</body>
</html>
"""


def _generate_intro_roster_image(output_dir: Path, rows: List[dict]) -> Path:
    tile = 240
    label_height = 110
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

        display_path = resolve_display_thumbnail(output_dir, row)
        if display_path.exists():
            image = Image.open(display_path).convert("RGB")
            image = image.resize((tile, tile))
            canvas.paste(image, (x, y))
        else:
            draw.rectangle((x, y, x + tile, y + tile), fill=(220, 220, 220))

        draw.rectangle((x, y, x + tile, y + tile), outline=(60, 60, 60), width=2)
        timecode = row.get("timecode") or _format_timecode(row["timestamp"])
        draw.text((x, y + tile + 8), timecode, fill=(20, 20, 20))
        draw.text((x, y + tile + 28), row["name"], fill=(20, 20, 20))
        if row.get("location"):
            location = row["location"]
            if len(location) > 34:
                location = location[:31] + "..."
            draw.text((x, y + tile + 50), f"From: {location}", fill=(70, 70, 70))
        detail_y = y + tile + 68
        if row.get("pronunciation"):
            pronunciation = row["pronunciation"]
            if len(pronunciation) > 34:
                pronunciation = pronunciation[:31] + "..."
            draw.text((x, detail_y), f"Say: {pronunciation}", fill=(70, 70, 70))
            detail_y += 18
        if row.get("name_spelling"):
            spelling = row["name_spelling"]
            if len(spelling) > 34:
                spelling = spelling[:31] + "..."
            draw.text((x, detail_y), spelling, fill=(100, 100, 100))

    roster_path = output_dir / "intro_roster.jpg"
    canvas.save(roster_path, quality=92)
    return roster_path


def export_intro_roster(output_dir: Path) -> dict:
    export_intro_clips(output_dir)
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
                "location",
                "pronunciation",
                "name_spelling",
                "spelling_note",
                "clip_label",
                "face_id",
                "confidence",
                "transcript",
                "thumbnail_file",
                "intro_clip_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "timestamp": row["timestamp"],
                    "timecode": row["timecode"],
                    "location": row.get("location", ""),
                    "pronunciation": row.get("pronunciation", ""),
                    "name_spelling": row.get("name_spelling", ""),
                    "spelling_note": row.get("spelling_note", ""),
                    "clip_label": row.get("clip_label", ""),
                    "face_id": row.get("face_id", ""),
                    "confidence": row.get("match_confidence", ""),
                    "transcript": row.get("transcript", ""),
                    "thumbnail_file": row.get("thumbnail_file", ""),
                    "intro_clip_file": row.get("intro_clip_file", ""),
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
    data_uri = _thumbnail_data_uri(output_dir, row)
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
    cluster_cards = [_card_html(row, output_dir) for row in rows]

    intro_section = ""
    if all_detections:
        intro_cards = []
        for row in sorted(all_detections, key=lambda item: item["timestamp"]):
            row = {**row, "timecode": _format_timecode(row["timestamp"])}
            data_uri = _thumbnail_data_uri(output_dir, row)
            img_tag = (
                f'<img src="{data_uri}" alt="{row["name"]}" />'
                if data_uri
                else '<div class="missing">No image</div>'
            )
            intro_clip = row.get("intro_clip_file", "")
            poster_attr = f' poster="{data_uri}"' if data_uri else ""
            video_tag = ""
            if intro_clip:
                video_tag = (
                    f'<video class="intro-video" controls playsinline preload="metadata" '
                    f'src="{intro_clip}"{poster_attr}></video>'
                )
            thumb_tag = ""
            if intro_clip and data_uri:
                thumb_tag = f'<div class="photo-thumb">{img_tag}</div>'
            intro_cards.append(
                f"""
            <article class="intro-card">
              {video_tag or f'<div class="photo">{img_tag}</div>'}
              {thumb_tag}
              <div class="timecode">{row["timecode"]}</div>
              <div class="name">{row["name"]}</div>
              {_intro_detail_html(row)}
              <div class="meta">Face ID: {row.get("face_id") or "—"}</div>
              <div class="quote">"{row.get("transcript", "")}"</div>
            </article>
                """
            )
        intro_section = f"""
  <h1>All On-Camera Introductions</h1>
  <p class="lead">Every detected on-camera introduction with face photo, playable intro clip, name, location, quote, and spelling or pronunciation when offered. Play each clip to confirm.</p>
  <div class="intro-grid">
    {''.join(intro_cards)}
  </div>
  <h2>Face cluster reference</h2>
  <p class="lead">Grouped faces from the wide shot for lower-thirds and manual labeling.</p>
"""

    if intro_section:
        body_intro = intro_section
    else:
        body_intro = f'<h1>{title}</h1><p class="lead">Every detected face with its assigned name.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Inter, system-ui, sans-serif; margin: 24px; color: #111; background: #f7f7f7; }}
    h1 {{ margin-top: 0; font-size: 32px; }}
    h2 {{ margin-top: 36px; }}
    .lead {{ color: #555; max-width: 760px; line-height: 1.5; }}
    .sheet {{ display: grid; gap: 16px; }}
    .intro-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 18px;
      margin-bottom: 28px;
    }}
    .intro-card, .row {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 14px;
      padding: 14px;
      page-break-inside: avoid;
    }}
    .intro-card {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 16px;
      align-items: center;
    }}
    .intro-card img, .row img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      border-radius: 10px;
      background: #eee;
      display: block;
    }}
    .intro-video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 10px;
      background: #000;
      display: block;
      object-fit: contain;
      margin-bottom: 8px;
    }}
    .photo-thumb img {{
      width: 88px;
      height: 88px;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid #ddd;
    }}
    .row img {{ width: 160px; height: 160px; }}
    .missing {{
      width: 100%;
      aspect-ratio: 1;
      background: #eee;
      border-radius: 10px;
      display:flex;
      align-items:center;
      justify-content:center;
      color:#777;
    }}
    .timecode {{
      display: inline-block;
      align-self: flex-start;
      background: #111;
      color: #fff;
      font-size: 13px;
      font-weight: 700;
      padding: 5px 10px;
      border-radius: 999px;
    }}
    .name {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
    .label {{
      color: #777;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-right: 6px;
    }}
    .from, .pronunciation, .spelling {{ color: #333; font-size: 14px; line-height: 1.35; }}
    .note {{ color: #9a6700; font-size: 12px; line-height: 1.35; }}
    .meta {{ color: #555; font-size: 14px; margin-top: 2px; }}
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
      .intro-card, .row {{ border-color: #999; }}
    }}
  </style>
</head>
<body>
  {body_intro}
  <div class="sheet">
    {''.join(cluster_cards)}
  </div>
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
    refresh_intro_thumbnails(output_dir)
    refresh_detection_details(output_dir)
    export_intro_clips(output_dir)
    all_detections = _intro_roster_rows(output_dir)
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
