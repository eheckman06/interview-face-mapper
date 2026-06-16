from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional


PUBLISH_ENTRIES = (
    "intro_roster.html",
    "face_name_sheet.html",
    "intro_roster.csv",
    "intro_roster.json",
    "face_name_sheet.csv",
    "face_name_sheet.json",
    "intro_roster.jpg",
    "cast_sheet.jpg",
    "named_screengrab.jpg",
    "all_detected_names.json",
    "labels.json",
)

PUBLISH_DIRS = (
    "intro_clips",
    "name_intro_thumbnails",
    "face_thumbnails",
)


def publish_job(
    data_dir: Path,
    job_id: str,
    output_root: Optional[Path] = None,
) -> Path:
    source_dir = data_dir / "jobs" / job_id / "output"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing job output: {source_dir}")

    publish_dir = (output_root or data_dir / "publish") / job_id
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for name in PUBLISH_ENTRIES:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, publish_dir / name)
            copied.append(name)

    for dirname in PUBLISH_DIRS:
        source = source_dir / dirname
        if source.exists():
            shutil.copytree(source, publish_dir / dirname)
            copied.append(f"{dirname}/")

    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="refresh" content="0; url=intro_roster.html" />
  <title>Interview Face Mapper</title>
</head>
<body>
  <p><a href="intro_roster.html">Open intro roster</a></p>
  <p><a href="face_name_sheet.html">Open face-to-name sheet</a></p>
</body>
</html>
"""
    (publish_dir / "index.html").write_text(index_html, encoding="utf-8")

    readme = f"""# Interview Face Mapper — published job {job_id}

This folder is a self-contained static site. Open `index.html` locally, or deploy the whole folder to any static host.

## View locally

```bash
cd "{publish_dir}"
python3 -m http.server 8080
```

Then open: http://localhost:8080/

## Share with anyone (public URL)

Upload this entire folder to a static host, for example:

- **Netlify Drop**: https://app.netlify.com/drop — drag this folder in; you get a public `https://….netlify.app` link.
- **Cloudflare Pages**: connect a repo or upload the folder.
- **Google Drive / Dropbox**: zip this folder and share the link (viewers download and open `index.html`, or unzip and use a local server).

## Develop the tool itself

Clone and run the full app from the project repo:

```bash
git clone <your-repo-url>
cd interview-face-mapper
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
interview-mapper serve
```

Copied assets: {", ".join(copied)}
"""
    (publish_dir / "README.txt").write_text(readme, encoding="utf-8")
    return publish_dir
