from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

from .pipeline import create_job_root, finalize_upload_path, run_analysis, run_finalize
from .upload_stream import extract_boundary, parse_multipart_file, stream_body_to_file
from .face_sheet import _intro_roster_rows
from .intro_clips import export_intro_clips

JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _jobs_root(data_dir: Path) -> Path:
    root = data_dir / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_output_dir(data_dir: Path, job_id: str) -> Optional[Path]:
    job = _get_job(job_id)
    if job and job.get("output_dir"):
        return Path(job["output_dir"])
    output_dir = _jobs_root(data_dir) / job_id / "output"
    if (output_dir / "faces.json").exists():
        return output_dir
    return None


def _set_job(job_id: str, **updates) -> dict:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(updates)
        return job


def _get_job(job_id: str) -> Optional[dict]:
    with JOBS_LOCK:
        return JOBS.get(job_id)


FAST_MODE_LIMIT_S = 600.0
FAST_FACE_SAMPLE_FPS = 1.0


def _load_job_intros(output_dir: Path) -> List[dict]:
    if not output_dir.exists():
        return []
    export_intro_clips(output_dir)
    rows = _intro_roster_rows(output_dir)
    return rows


def _load_spoken_labels(output_dir: Path) -> dict:
    labels_path = output_dir / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))
    return {}


def _run_job(
    job_id: str,
    main_video: Path,
    extra_videos: List[Path],
    output_dir: Path,
    full_analysis: bool = False,
) -> None:
    def progress(message: str) -> None:
        _set_job(job_id, status="processing", message=message)

    try:
        progress(
            "Analyzing video (step 1 of 3). Face detection is CPU-heavy and can take several minutes..."
        )
        result = run_analysis(
            main_video,
            output_dir,
            extra_videos=extra_videos,
            face_sample_fps=2.0 if full_analysis else FAST_FACE_SAMPLE_FPS,
            max_duration_s=None if full_analysis else FAST_MODE_LIMIT_S,
            extra_max_duration_s=None if full_analysis else FAST_MODE_LIMIT_S,
            progress=progress,
        )
        faces_payload = json.loads((output_dir / "faces.json").read_text(encoding="utf-8"))
        spoken_labels = _load_spoken_labels(output_dir)
        intros = _load_job_intros(output_dir)
        _set_job(
            job_id,
            status="ready",
            message="Analysis complete. Review introductions and confirm names.",
            output_dir=str(output_dir),
            face_clusters=faces_payload.get("face_clusters", []),
            face_cluster_count=result.face_cluster_count,
            speaker_segment_count=result.speaker_segment_count,
            spoken_labels=spoken_labels,
            intros=intros,
            intro_count=len(intros),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - surface processing errors to the UI
        _set_job(job_id, status="error", message=str(exc), error=str(exc))


UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Interview Face Mapper</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; margin: 0; background: #101010; color: #f3f3f3; }
    main { max-width: 760px; margin: 0 auto; padding: 32px 20px 48px; }
    h1 { margin-top: 0; }
    .card { background: #1a1a1a; border: 1px solid #333; border-radius: 14px; padding: 20px; margin-top: 18px; }
    label { display: block; margin-bottom: 14px; font-size: 14px; color: #cfcfcf; }
    input[type=file] { display: block; margin-top: 8px; color: #fff; }
    button { background: #4f7cff; color: white; border: 0; border-radius: 10px; padding: 12px 18px; font-size: 15px; cursor: pointer; }
    .hint { color: #9a9a9a; font-size: 13px; line-height: 1.5; }
    .status { margin-top: 16px; padding: 12px; border-radius: 10px; background: #222; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; margin-top: 16px; }
    .intro-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; margin-top: 16px; }
    .intro-card { background: #141414; border: 1px solid #333; border-radius: 14px; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
    .intro-card video { width: 100%; aspect-ratio: 16 / 9; border-radius: 10px; background: #000; object-fit: contain; }
    .intro-card .face-thumb { width: 88px; height: 88px; object-fit: cover; border-radius: 8px; border: 1px solid #444; }
    .intro-card .thumb-row { display: flex; align-items: center; gap: 8px; color: #8f8f8f; font-size: 12px; font-weight: 600; }
    .intro-card .name { font-size: 24px; font-weight: 700; margin: 0; }
    .intro-card .timecode { display: inline-block; align-self: flex-start; background: #4f7cff; color: #fff; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 999px; }
    .intro-card .location, .intro-card .spelling, .intro-card .pronunciation { color: #bdbdbd; font-size: 14px; line-height: 1.4; }
    .intro-card .label { color: #7d7d7d; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-right: 6px; }
    .intro-card .quote { color: #d8d8d8; font-size: 14px; line-height: 1.45; border-top: 1px solid #2d2d2d; padding-top: 10px; margin-top: 4px; font-style: italic; }
    .intro-card .face-id { color: #8f8f8f; font-size: 12px; }
    .face-card img { width: 100%; border-radius: 8px; aspect-ratio: 1; object-fit: cover; background: #000; }
    .face-card input { width: 100%; margin-top: 8px; padding: 8px; border-radius: 8px; border: 1px solid #444; background: #111; color: #fff; box-sizing: border-box; }
    .downloads a { display: block; margin-top: 8px; color: #9eb7ff; }
    img.preview { width: 100%; border-radius: 12px; margin-top: 16px; border: 1px solid #333; }
    .hidden { display: none; }
    .progress { margin-top: 12px; height: 10px; background: #2b2b2b; border-radius: 999px; overflow: hidden; }
    .progress > div { height: 100%; width: 0%; background: #4f7cff; transition: width 0.2s ease; }
    .file-meta { color: #8f8f8f; font-size: 12px; margin-top: 6px; }
    .checkbox { display: flex; gap: 10px; align-items: flex-start; margin: 14px 0; }
    .checkbox input { margin-top: 3px; }
  </style>
</head>
<body>
  <main>
    <h1>Upload interview clips</h1>
    <p class="hint">Upload your wide-shot master plus any solo intro clips or extra angles. Whisper transcribes on-camera introductions for <strong>name, location, and quote</strong>, and the site generates a short <strong>intro clip</strong> for each person so you can confirm on camera. Faces are detected across all uploaded clips. Lower-thirds and speaker timing use the main clip only.</p>
    <p class="hint">Large files can take a while to upload before analysis starts. Check <strong>Analyze the full length</strong> if introductions happen after 10 minutes.</p>

    <form id="uploadForm" class="card" enctype="multipart/form-data" method="post" action="/upload">
      <label>Main clip — wide shot / master (required)
        <input id="mainVideo" type="file" name="main_video" accept="video/mp4,video/quicktime,video/x-m4v,video/*,.mp4,.mov,.m4v" required />
        <div id="mainVideoMeta" class="file-meta"></div>
      </label>
      <label>Additional clips (optional) — solo intros, other angles
        <input id="extraVideos" type="file" name="extra_videos" accept="video/mp4,video/quicktime,video/x-m4v,video/*,.mp4,.mov,.m4v" multiple />
        <div id="extraVideosMeta" class="file-meta"></div>
      </label>
      <label class="checkbox">
        <input type="checkbox" name="full_analysis" value="1" />
        <span>Analyze the full length of each clip (much slower for long interviews)</span>
      </label>
      <button type="submit">Upload and analyze</button>
    </form>

    <div id="statusCard" class="card hidden">
      <div id="statusText" class="status">Uploading...</div>
      <div id="progressBar" class="progress hidden"><div id="progressFill"></div></div>
      <div id="sheetSection" class="hidden">
        <h2>Review on-camera introductions</h2>
        <p class="hint">Whisper transcribes each intro for name, location, and quote. Play the clip to confirm each person on camera.</p>
        <div id="introGrid" class="intro-grid"></div>
        <div class="downloads" id="sheetDownloads" style="margin-top:18px;"></div>
      </div>
      <div id="labelSection" class="hidden">
        <h2>Name each person</h2>
        <div id="faceGrid" class="grid"></div>
        <button id="finalizeBtn" type="button" style="margin-top:16px;">Save names and update face sheet</button>
      </div>
      <div id="resultSection" class="hidden">
        <h2>Results</h2>
        <img id="screengrab" class="preview" alt="Named screengrab" />
        <div class="downloads" id="downloads"></div>
      </div>
    </div>
  </main>
  <script>
    const uploadForm = document.getElementById("uploadForm");
    const statusCard = document.getElementById("statusCard");
    const statusText = document.getElementById("statusText");
    const sheetSection = document.getElementById("sheetSection");
    const sheetDownloads = document.getElementById("sheetDownloads");
    const introGrid = document.getElementById("introGrid");
    const labelSection = document.getElementById("labelSection");
    const resultSection = document.getElementById("resultSection");
    const faceGrid = document.getElementById("faceGrid");
    const screengrab = document.getElementById("screengrab");
    const downloads = document.getElementById("downloads");
    const progressBar = document.getElementById("progressBar");
    const progressFill = document.getElementById("progressFill");
    let currentJobId = null;

    function formatSize(bytes) {
      if (!bytes) return "0 B";
      const units = ["B", "KB", "MB", "GB"];
      let size = bytes;
      let unit = 0;
      while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
      }
      return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
    }

    function updateSelectedFiles() {
      const main = document.getElementById("mainVideo").files[0];
      const extras = [...document.getElementById("extraVideos").files];
      document.getElementById("mainVideoMeta").textContent = main
        ? `${main.name} (${formatSize(main.size)})`
        : "";
      document.getElementById("extraVideosMeta").textContent = extras.length
        ? extras.map((file) => `${file.name} (${formatSize(file.size)})`).join(" · ")
        : "";
    }

    document.getElementById("mainVideo").addEventListener("change", updateSelectedFiles);
    document.getElementById("extraVideos").addEventListener("change", updateSelectedFiles);

    uploadForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusCard.classList.remove("hidden");
      sheetSection.classList.add("hidden");
      labelSection.classList.add("hidden");
      resultSection.classList.add("hidden");
      progressBar.classList.remove("hidden");
      progressFill.style.width = "0%";
      statusText.textContent = "Uploading MP4s to this computer...";

      const formData = new FormData(uploadForm);
      const payload = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/upload");
        xhr.upload.addEventListener("progress", (event) => {
          if (!event.lengthComputable) {
            statusText.textContent = "Uploading MP4s...";
            return;
          }
          const pct = Math.round((event.loaded / event.total) * 100);
          progressFill.style.width = `${pct}%`;
          statusText.textContent = `Uploading MP4s... ${pct}% (${formatSize(event.loaded)} of ${formatSize(event.total)})`;
        });
        xhr.addEventListener("load", () => {
          let body = {};
          try {
            body = JSON.parse(xhr.responseText || "{}");
          } catch (error) {
            reject(new Error("Upload failed. The server returned an invalid response."));
            return;
          }
          resolve({ ok: xhr.status >= 200 && xhr.status < 300, status: xhr.status, body });
        });
        xhr.addEventListener("error", () => reject(new Error("Upload failed. The server may have stopped, or the file may be too large for the connection.")));
        xhr.addEventListener("timeout", () => reject(new Error("Upload timed out. Try a smaller clip or restart the server.")));
        xhr.timeout = 0;
        xhr.send(formData);
      }).catch((error) => {
        statusText.textContent = error.message || "Upload failed.";
        return null;
      });

      if (!payload) return;
      if (!payload.ok) {
        statusText.textContent = payload.body?.error || `Upload failed (server status ${payload.status || "unknown"}).`;
        return;
      }

      progressBar.classList.add("hidden");
      currentJobId = payload.body.job_id;
      statusText.textContent = payload.body.message || "Upload complete. Analysis started.";
      pollJob();
    });

    async function pollJob() {
      const response = await fetch(`/api/jobs/${currentJobId}`);
      const job = await response.json();
      statusText.textContent = job.message || job.status;
      if (job.status === "processing" || job.status === "queued") {
        setTimeout(pollJob, 2000);
        return;
      }
      if (job.status === "error") {
        return;
      }
      renderIntros(job.intros || []);
      renderSheetDownloads();
      sheetSection.classList.remove("hidden");
      renderFaces(job.face_clusters || [], job.spoken_labels || {});
      labelSection.classList.remove("hidden");
    }

    function thumbUrl(filePath) {
      if (!filePath) return "";
      return `/jobs/${currentJobId}/file/${filePath}`;
    }

    function renderIntros(intros) {
      introGrid.innerHTML = "";
      if (!intros.length) {
        introGrid.innerHTML = '<p class="hint">No on-camera introductions were detected. Try full-length analysis if intros are after 10 minutes.</p>';
        return;
      }
      intros.forEach((intro) => {
        const card = document.createElement("article");
        card.className = "intro-card";
        const thumb = thumbUrl(intro.thumbnail_file);
        const clip = thumbUrl(intro.intro_clip_file);
        const details = [];
        if (intro.location) {
          details.push(`<div class="location"><span class="label">From</span> ${intro.location}</div>`);
        }
        if (intro.pronunciation) {
          details.push(`<div class="pronunciation"><span class="label">Pronunciation</span> ${intro.pronunciation}</div>`);
        }
        if (intro.name_spelling) {
          details.push(`<div class="spelling"><span class="label">Spelling</span> ${intro.name_spelling}</div>`);
        }
        if (intro.name_heard && intro.name_heard !== intro.name) {
          details.push(`<div class="heard"><span class="label">Heard as</span> ${intro.name_heard}</div>`);
        }
        if (intro.spelling_note) {
          details.push(`<div class="note">${intro.spelling_note}</div>`);
        }
        if (intro.intro_clip_faces_visible) {
          details.push(`<div class="faces-visible"><span class="label">On camera</span> Up to ${intro.intro_clip_faces_visible} people visible in clip</div>`);
        }
        if (intro.intro_clip_includes_full_cast) {
          details.push(`<div class="full-cast"><span class="label">Full cast</span> Clip ends with a wide shot showing everyone on camera</div>`);
        }
        const poster = thumb ? ` poster="${thumb}"` : "";
        card.innerHTML = `
          ${clip
            ? `<video controls playsinline preload="metadata" src="${clip}"${poster}></video>`
            : (thumb ? `<img src="${thumb}" alt="${intro.name}" style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;" />` : '<p class="hint">Intro clip unavailable</p>')}
          ${clip && thumb ? `<div class="thumb-row"><img class="face-thumb" src="${thumb}" alt="${intro.name}" /> Face at intro</div>` : ""}
          <div class="timecode">${intro.timecode || ""}</div>
          <div class="name">${intro.name}</div>
          ${details.join("")}
          <div class="face-id">${intro.face_id ? `Mapped to ${intro.face_id}` : "Face cluster not mapped"}</div>
          <div class="quote">"${intro.transcript || ""}"</div>
        `;
        introGrid.appendChild(card);
      });
    }

    function renderSheetDownloads() {
      sheetDownloads.innerHTML = `
        <a href="/jobs/${currentJobId}/file/intro_roster.html" target="_blank"><strong>Open full intro roster page</strong></a>
        <a href="/jobs/${currentJobId}/file/intro_roster.jpg" target="_blank">Open intro roster image</a>
        <a href="/jobs/${currentJobId}/file/intro_roster.csv" download>Download intro roster CSV</a>
        <a href="/jobs/${currentJobId}/file/cast_sheet.jpg" target="_blank">Open cast sheet (face clusters)</a>
        <a href="/jobs/${currentJobId}/file/face_name_sheet.html" target="_blank">Open full face-to-name sheet</a>
        <a href="/jobs/${currentJobId}/file/face_name_sheet.csv" download>Download face sheet CSV</a>
      `;
    }

    function renderFaces(clusters, spokenLabels) {
      faceGrid.innerHTML = "";
      clusters.forEach((cluster) => {
        const card = document.createElement("div");
        card.className = "face-card";
        const thumb = cluster.thumbnail_path.includes("/")
          ? cluster.thumbnail_path.split("/").pop()
          : cluster.thumbnail_path.replace("face_thumbnails/", "");
        const preset = spokenLabels[cluster.cluster_id] || "";
        card.innerHTML = `
          <img src="/jobs/${currentJobId}/file/face_thumbnails/${thumb}" alt="${cluster.cluster_id}" />
          <input data-id="${cluster.cluster_id}" placeholder="Full name" value="${preset.replace(/"/g, "&quot;")}" />
        `;
        faceGrid.appendChild(card);
      });
    }

    document.getElementById("finalizeBtn").addEventListener("click", async () => {
      const labels = {};
      faceGrid.querySelectorAll("input").forEach((input) => {
        if (input.value.trim()) labels[input.dataset.id] = input.value.trim();
      });
      statusText.textContent = "Generating screengrab and exports...";
      const response = await fetch(`/api/jobs/${currentJobId}/finalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labels }),
      });
      const payload = await response.json();
      if (!response.ok) {
        statusText.textContent = payload.error || "Finalize failed.";
        return;
      }
      statusText.textContent = "Done.";
      resultSection.classList.remove("hidden");
      screengrab.src = `/jobs/${currentJobId}/file/named_screengrab.jpg?ts=${Date.now()}`;
      renderIntros(payload.intros || []);
      renderSheetDownloads();
      downloads.innerHTML = `
        <a href="/jobs/${currentJobId}/file/intro_roster.html" target="_blank"><strong>Open full intro roster page</strong></a>
        <a href="/jobs/${currentJobId}/file/intro_roster.jpg" target="_blank">Open intro roster image</a>
        <a href="/jobs/${currentJobId}/file/intro_roster.csv" download>Download intro roster CSV</a>
        <a href="/jobs/${currentJobId}/file/cast_sheet.jpg" target="_blank">Open cast sheet (face clusters)</a>
        <a href="/jobs/${currentJobId}/file/face_name_sheet.html" target="_blank">Open full face-to-name sheet</a>
        <a href="/jobs/${currentJobId}/file/face_name_sheet.csv" download>Download face sheet CSV</a>
        <a href="/jobs/${currentJobId}/file/named_screengrab.jpg" download>Download named screengrab</a>
        <a href="/jobs/${currentJobId}/file/lower_thirds.csv" download>Download lower thirds CSV</a>
        <a href="/jobs/${currentJobId}/file/lower_thirds.json" download>Download lower thirds JSON</a>
        <a href="/jobs/${currentJobId}/file/lower_third_markers.edl" download>Download EDL markers</a>
      `;
    });
  </script>
</body>
</html>
"""


class InterviewMapperHandler(BaseHTTPRequestHandler):
    data_dir: Path = Path("data")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(str(path))
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/upload"):
            self._send_html(UPLOAD_PAGE)
            return

        if path == "/health":
            self._send_json({"status": "ok", "service": "interview-face-mapper"})
            return

        if path.startswith("/api/jobs/") and path.endswith("/intros"):
            job_id = path.split("/api/jobs/", 1)[1].rsplit("/intros", 1)[0].strip("/")
            job = _get_job(job_id)
            if not job or not job.get("output_dir"):
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            intros = _load_job_intros(Path(job["output_dir"]))
            self._send_json({"intros": intros, "spoken_labels": _load_spoken_labels(Path(job["output_dir"]))})
            return

        if path.startswith("/api/jobs/"):
            job_id = path.split("/api/jobs/", 1)[1].strip("/")
            job = _get_job(job_id)
            if not job:
                self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(job)
            return

        if path.startswith("/jobs/") and "/file/" in path:
            _, remainder = path.split("/jobs/", 1)
            job_id, _, rel_path = remainder.partition("/file/")
            output_dir = _resolve_output_dir(self.data_dir, job_id)
            if not output_dir:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            safe_rel = Path(unquote(rel_path))
            if safe_rel.is_absolute() or ".." in safe_rel.parts:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self._send_file(output_dir / safe_rel)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/upload":
            self._handle_upload()
            return

        if path.startswith("/api/jobs/") and path.endswith("/finalize"):
            job_id = path.split("/api/jobs/", 1)[1].rsplit("/finalize", 1)[0]
            self._handle_finalize(job_id)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_upload(self) -> None:
        try:
            self._process_upload()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self._send_json(
                {"error": f"Upload failed while receiving the file: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"error": f"Upload failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _process_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            self._send_json({"error": "Expected multipart upload"}, status=HTTPStatus.BAD_REQUEST)
            return

        boundary = extract_boundary(content_type)
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json({"error": "Empty upload."}, status=HTTPStatus.BAD_REQUEST)
            return

        job_dir = create_job_root(_jobs_root(self.data_dir))
        job_id = job_dir.name
        uploads_dir = job_dir / "uploads"
        output_dir = job_dir / "output"
        raw_upload = uploads_dir / "_raw_multipart.upload"

        stream_body_to_file(self.rfile, length, raw_upload)
        parsed = parse_multipart_file(raw_upload, boundary, uploads_dir)
        raw_upload.unlink(missing_ok=True)

        main_parts = parsed.parts.get("main_video", [])
        if not main_parts or not main_parts[0].filename or not main_parts[0].file_path:
            self._send_json({"error": "Main MP4 is required."}, status=HTTPStatus.BAD_REQUEST)
            return

        full_analysis = any(part.content == b"1" for part in parsed.parts.get("full_analysis", []))

        main_video = finalize_upload_path(
            main_parts[0].file_path,
            uploads_dir,
            prefix="main",
            upload_name=main_parts[0].filename,
        )
        extra_videos: List[Path] = []
        for idx, item in enumerate(parsed.parts.get("extra_videos", [])):
            if not item.filename or not item.file_path:
                continue
            extra_videos.append(
                finalize_upload_path(
                    item.file_path,
                    uploads_dir,
                    prefix=f"extra_{idx}",
                    upload_name=item.filename,
                )
            )

        total_bytes = main_video.stat().st_size + sum(path.stat().st_size for path in extra_videos)
        clip_count = 1 + len(extra_videos)
        mode_label = "full length of each clip" if full_analysis else "first 10 minutes of each clip"
        clip_note = f"{clip_count} clip{'s' if clip_count != 1 else ''}"
        _set_job(
            job_id,
            status="queued",
            message=f"Upload complete ({total_bytes / (1024 * 1024):.1f} MB, {clip_note}). Starting {mode_label} analysis...",
            output_dir=str(output_dir),
            main_video=str(main_video),
            full_analysis=full_analysis,
            clip_count=clip_count,
        )
        worker = threading.Thread(
            target=_run_job,
            args=(job_id, main_video, extra_videos, output_dir, full_analysis),
            daemon=True,
        )
        worker.start()
        self._send_json(
            {
                "job_id": job_id,
                "status": "queued",
                "message": f"Upload complete. Starting {mode_label} analysis...",
                "clip_count": clip_count,
            }
        )

    def _handle_finalize(self, job_id: str) -> None:
        job = _get_job(job_id)
        if not job or not job.get("output_dir"):
            self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        labels = payload.get("labels", {})
        if not labels:
            self._send_json({"error": "Add at least one name before finalizing."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            exports = run_finalize(Path(job["output_dir"]), labels)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        output_dir = Path(job["output_dir"])
        intros = _load_job_intros(output_dir)
        _set_job(
            job_id,
            status="complete",
            message="Exports ready.",
            exports=exports,
            intros=intros,
            spoken_labels=labels,
        )
        self._send_json({"status": "complete", "exports": exports, "intros": intros})


def create_server(host: str, port: int, data_dir: Path) -> ThreadingHTTPServer:
    handler = InterviewMapperHandler
    handler.data_dir = data_dir
    return ThreadingHTTPServer((host, port), handler)


def app_url(host: str, port: int) -> str:
    display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0", "::") else host
    return f"http://{display_host}:{port}/"


def open_in_browser(url: str, browser: Optional[str] = None) -> None:
    if sys.platform == "darwin":
        try:
            if browser:
                subprocess.run(["open", "-a", browser, url], check=True)
            else:
                subprocess.run(["open", url], check=True)
            return
        except (OSError, subprocess.CalledProcessError):
            pass
    if browser:
        webbrowser.get(browser).open(url)
    else:
        webbrowser.open(url)


def serve(
    host: str = "127.0.0.1",
    port: int = 4173,
    data_dir: Optional[Path] = None,
    open_browser: bool = True,
    browser: Optional[str] = None,
) -> None:
    root = data_dir or Path.cwd() / "data"
    root.mkdir(parents=True, exist_ok=True)
    url = app_url(host, port)

    try:
        server = create_server(host, port, root)
    except OSError as exc:
        if getattr(exc, "errno", None) == 48:
            raise RuntimeError(
                f"Port {port} is already in use. Restart with `interview-mapper serve --port {port + 1}`."
            ) from exc
        raise

    print("Interview Face Mapper is running.")
    print(f"Upload UI: {url}")
    print(f"Job data: {root.resolve()}")
    print("Keep this terminal open while you upload MP4s.")
    print("Do not use the chat link — run `interview-mapper serve` and let it open your browser.")

    if open_browser:
        open_in_browser(url, browser=browser)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
