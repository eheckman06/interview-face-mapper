from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import click

from .faces import discover_videos
from .label_ui import generate_label_ui, load_labels
from .pipeline import rebuild_analysis, run_analysis, run_finalize
from .publish import publish_job
from .screengrab import generate_named_screengrab
from .web_app import serve


@click.group()
@click.version_option()
def main() -> None:
    """Map names to faces in multi-person interview videos."""


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (defaults to <video_stem>_mapper_output).",
)
@click.option("--max-speakers", default=6, show_default=True, help="Upper bound for speaker clusters.")
@click.option("--face-sample-fps", default=2.0, show_default=True, help="Face sampling rate.")
@click.option(
    "--extra-video",
    "extra_videos",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Additional clips to analyze for faces and on-camera names (lower-thirds still use the main video).",
)
@click.option(
    "--video-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder of videos; all clips are sampled for faces except the main video, which is already included.",
)
def analyze(
    video: Path,
    output: Optional[Path],
    max_speakers: int,
    face_sample_fps: float,
    extra_videos: Tuple[Path, ...],
    video_dir: Optional[Path],
) -> None:
    """Detect faces, diarize speakers, and build a labeling UI."""
    output_dir = output or video.with_name(f"{video.stem}_mapper_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    supplemental_videos: List[Path] = list(extra_videos)
    if video_dir:
        for item in discover_videos(video_dir):
            if item.resolve() != video.resolve() and item not in supplemental_videos:
                supplemental_videos.append(item)

    click.echo("Analyzing faces...")
    if supplemental_videos:
        click.echo(f"Sampling faces from {1 + len(supplemental_videos)} videos...")
    result = run_analysis(
        video,
        output_dir,
        extra_videos=supplemental_videos,
        max_speakers=max_speakers,
        face_sample_fps=face_sample_fps,
    )
    faces_payload = json.loads((output_dir / "faces.json").read_text(encoding="utf-8"))
    ui_path = output_dir / "label.html"

    click.echo("")
    click.echo(f"Found {result.face_cluster_count} face clusters")
    click.echo(f"Found {result.speaker_segment_count} speaker segments")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Open labeling UI: {ui_path}")
    click.echo(f"Intro roster (all names + faces + times): {output_dir / 'intro_roster.html'}")
    click.echo(f"Face-to-name sheet: {output_dir / 'face_name_sheet.html'}")
    click.echo(f"Face sheet CSV: {output_dir / 'face_name_sheet.csv'}")
    click.echo("Next: label faces in label.html, save labels.json, then run `interview-mapper finalize`.")
    click.echo("Optional preview before naming: `interview-mapper screengrab` (uses face_0, face_1, etc.).")


@main.command()
@click.argument(
    "output_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def finalize(output_dir: Path) -> None:
    """Apply manual labels and export lower-third timelines."""
    correlation_path = output_dir / "correlation.json"
    if not correlation_path.exists():
        raise click.ClickException("Missing correlation.json. Run `interview-mapper analyze` first.")

    labels = load_labels(output_dir)
    exports = run_finalize(output_dir, labels)

    click.echo(f"Exported lower thirds to {exports['lower_thirds_json']}")
    click.echo(f"CSV for spreadsheets: {exports['lower_thirds_csv']}")
    click.echo(f"EDL markers: {exports['lower_third_markers_edl']}")
    click.echo(f"Named screengrab: {exports['named_screengrab']}")
    click.echo(f"Face-to-name sheet: {exports['face_name_sheet_html']}")
    click.echo(f"Face sheet CSV: {exports['face_name_sheet_csv']}")
    click.echo(f"Cast sheet image: {exports['cast_sheet']}")


@main.command("rebuild")
@click.argument(
    "output_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def rebuild(output_dir: Path) -> None:
    """Re-cluster faces and regenerate the face-to-name sheet from an existing analysis."""
    result = rebuild_analysis(output_dir)
    click.echo(f"Detected {result['face_clusters']} people.")
    click.echo(f"Intro roster: {result.get('intro_roster_html', output_dir / 'intro_roster.html')}")
    click.echo(f"Intro roster: {result.get('intro_roster_html', output_dir / 'intro_roster.html')}")
    click.echo(f"Face-to-name sheet: {result['face_name_sheet_html']}")
    click.echo(f"Cast sheet image: {result['cast_sheet']}")


@main.command()
@click.argument(
    "output_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--timestamp",
    type=float,
    default=None,
    help="Exact frame time in seconds. Defaults to the frame with the most detected faces.",
)
@click.option(
    "--output-name",
    default="named_screengrab.jpg",
    show_default=True,
    help="Filename for the annotated screengrab.",
)
def screengrab(output_dir: Path, timestamp: Optional[float], output_name: str) -> None:
    """Generate an annotated screengrab with names over each face."""
    labels = None
    labels_path = output_dir / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        click.echo("No labels.json found yet; using face_0, face_1, etc. as placeholders.")

    image_path = generate_named_screengrab(
        output_dir,
        labels=labels,
        timestamp=timestamp,
        output_name=output_name,
    )
    click.echo(f"Saved annotated screengrab: {image_path}")


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host for the upload UI.")
@click.option("--port", default=4173, show_default=True, help="Port for the upload UI.")
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where uploaded MP4s and job outputs are stored.",
)
@click.option("--open/--no-open", default=True, show_default=True, help="Open the upload page in your browser.")
@click.option(
    "--browser",
    default=None,
    help="macOS app to open, e.g. Island, Safari, \"Google Chrome\".",
)
def serve_cmd(host: str, port: int, data_dir: Optional[Path], open: bool, browser: Optional[str]) -> None:
    """Start a local web UI for uploading MP4s."""
    try:
        serve(host=host, port=port, data_dir=data_dir, open_browser=open, browser=browser)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("publish")
@click.argument("job_id")
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Project data directory (defaults to ./data).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the static site (defaults to <data-dir>/publish/<job-id>).",
)
def publish_cmd(job_id: str, data_dir: Optional[Path], output: Optional[Path]) -> None:
    """Export a job as a self-contained static site for sharing or deployment."""
    root = data_dir or Path.cwd() / "data"
    publish_dir = publish_job(root, job_id, output_root=output)
    click.echo(f"Published static site: {publish_dir.resolve()}")
    click.echo("Open locally:  python3 -m http.server 8080  (from that folder)")
    click.echo("Public link:   upload the folder to https://app.netlify.com/drop")


@main.command("regenerate-ui")
@click.argument(
    "output_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def regenerate_ui(output_dir: Path) -> None:
    """Rebuild label.html from an existing faces.json."""
    faces_path = output_dir / "faces.json"
    if not faces_path.exists():
        raise click.ClickException("Missing faces.json. Run `interview-mapper analyze` first.")
    faces_payload = json.loads(faces_path.read_text(encoding="utf-8"))
    ui_path = generate_label_ui(output_dir, faces_payload)
    click.echo(f"Rebuilt labeling UI: {ui_path}")


if __name__ == "__main__":
    main()
