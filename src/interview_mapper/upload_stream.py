from __future__ import annotations

import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

from .pipeline import _safe_stem

CHUNK_SIZE = 4 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


@dataclass
class UploadedPart:
    name: str
    filename: Optional[str]
    content: bytes = b""
    file_path: Optional[Path] = None


@dataclass
class MultipartUpload:
    parts: Dict[str, List[UploadedPart]] = field(default_factory=dict)

    def add_part(self, part: UploadedPart) -> None:
        self.parts.setdefault(part.name, []).append(part)


def extract_boundary(content_type: str) -> bytes:
    match = re.search(r"boundary=([^;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Missing multipart boundary.")
    return match.group(1).strip().strip('"').encode("utf-8")


def stream_body_to_file(source: BinaryIO, total_length: int, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    received = 0
    with destination.open("wb") as handle:
        while received < total_length:
            chunk = source.read(min(CHUNK_SIZE, total_length - received))
            if not chunk:
                raise OSError("Upload disconnected before the full file was received.")
            handle.write(chunk)
            received += len(chunk)
    return received


def _parse_headers(header_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    name = None
    filename = None
    for line in header_bytes.decode("utf-8", errors="ignore").split("\r\n"):
        if not line.lower().startswith("content-disposition:"):
            continue
        for token in line.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token.split("=", 1)[1].strip('"')
            elif token.startswith("filename="):
                filename = token.split("=", 1)[1].strip('"')
    return name, filename


def parse_multipart_file(path: Path, boundary: bytes, uploads_dir: Path) -> MultipartUpload:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    delimiter = b"--" + boundary
    result = MultipartUpload()

    with path.open("rb") as raw_file:
        with mmap.mmap(raw_file.fileno(), 0, access=mmap.ACCESS_READ) as content:
            position = 0
            while position < len(content):
                start = content.find(delimiter, position)
                if start == -1:
                    break
                start += len(delimiter)
                if content[start : start + 2] == b"--":
                    break
                if content[start : start + 2] == b"\r\n":
                    start += 2
                header_end = content.find(b"\r\n\r\n", start)
                if header_end == -1:
                    break
                name, filename = _parse_headers(content[start:header_end])
                body_start = header_end + 4
                next_delim = content.find(delimiter, body_start)
                if next_delim == -1:
                    break
                body_end = next_delim
                if content[body_end - 2 : body_end] == b"\r\n":
                    body_end -= 2
                if not name:
                    position = next_delim
                    continue

                if filename:
                    suffix = Path(filename).suffix.lower()
                    if suffix not in ALLOWED_VIDEO_SUFFIXES:
                        raise ValueError(
                            f"Unsupported file type: {suffix or '(none)'}. Upload MP4, MOV, or similar video files."
                        )
                    target = uploads_dir / f"incoming_{name}_{_safe_stem(filename)}{suffix}"
                    target.write_bytes(content[body_start:body_end])
                    result.add_part(UploadedPart(name=name, filename=filename, file_path=target))
                else:
                    result.add_part(
                        UploadedPart(
                            name=name,
                            filename=None,
                            content=content[body_start:body_end],
                        )
                    )
                position = next_delim

    return result
