# Interview Face Mapper

Map speaker names to faces in multi-person interview videos, then export a lower-third timeline for production.

Built for **single wide-shot interviews** where multiple people are visible at once. The tool:

1. Detects and clusters unique faces across the video
2. Diarizes speakers from the audio track
3. Correlates speakers to faces using lip activity during each speech segment
4. Lets you manually label each face cluster with a display name
5. Exports lower-third timing files for your edit

## Quick start

```bash
cd ~/Projects/interview-face-mapper
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Upload MP4s in the browser

Start the local upload UI:

```bash
interview-mapper serve
```

That command starts the server and opens the upload page in your default browser (Safari/Chrome/etc.). **Don't click the chat link** — it may open the wrong thing if the server isn't running.

The upload page is `http://localhost:4173/` by default.

**Why it can feel slow:** the browser first uploads the entire MP4 to your computer, then the tool analyzes it locally. Large files can take several minutes just to upload, and analysis after that is CPU-heavy. The web UI now shows upload progress and uses a fast default that analyzes only the first 10 minutes unless you opt into full-video analysis.

Then:

1. Upload your **main wide-shot clip** (used for lower-thirds timing)
2. Add **additional clips** — solo intros, other angles (faces and names are detected from every clip)
3. Wait for analysis to finish
4. Enter names for each detected face
5. Open the **face-to-name sheet** (all detected faces with thumbnails)
6. Add names, then download the updated sheet, screengrab, and lower-third files

Uploaded files are stored under `data/jobs/` in the project folder.

Analyze one main video:

```bash
interview-mapper analyze /path/to/interview.mp4
```

Or sample faces from every clip in a folder (useful when you have multiple angles or takes):

```bash
interview-mapper analyze /path/to/main_interview.mp4 --video-dir /path/to/all_clips
```

You can also add specific extra videos:

```bash
interview-mapper analyze /path/to/main.mp4 --extra-video /path/to/wide.mp4 --extra-video /path/to/reaction.mp4
```

Speaker diarization and lower-thirds always use the **main clip**. Additional clips improve face clustering and on-camera name detection (solo intros, etc.).

This creates an output folder (for example `interview_mapper_output/`) containing:

- `face_thumbnails/` — one image per detected person
- `label.html` — open in a browser to assign names
- `correlation.json` — speaker-to-face mapping draft
- `speakers.json`, `faces.json` — intermediate analysis data

Label faces:

1. Open `label.html` in your browser
2. Enter each person's display name
3. Click **Generate labels.json** and save it into the output folder

Finalize exports:

```bash
interview-mapper finalize /path/to/interview_mapper_output
```

Exports:

- `lower_thirds.json` — structured timeline
- `lower_thirds.csv` — spreadsheet-friendly with timecodes
- `lower_third_markers.edl` — marker-style EDL you can import or reference in an NLE
- `face_name_sheet.html` / `face_name_sheet.csv` — full face-to-name reference sheet after analysis
- `named_screengrab.jpg` — wide-shot frame with name labels drawn on each face

Generate a named screengrab anytime:

```bash
# After labeling
interview-mapper screengrab /path/to/interview_mapper_output

# Preview before naming (shows face_0, face_1, etc.)
interview-mapper screengrab /path/to/interview_mapper_output

# Pick a specific frame
interview-mapper screengrab /path/to/interview_mapper_output --timestamp 42.5
```

## How it works

| Step | What happens |
|------|----------------|
| Face detection | MediaPipe samples frames and clusters recurring faces |
| Speaker diarization | Resemblyzer groups the audio into speaker segments |
| Correlation | During each speech segment, the face with the most lip movement is treated as the active speaker |
| Manual labeling | You confirm names once per face cluster |
| Export | Named lower-thirds are written with start/end timestamps |

## Tips for best results

- Use footage where faces are clearly visible in the wide shot
- Clean dialogue audio helps speaker separation
- If two people talk over each other, review `correlation.json` and adjust names before export
- Re-run with `--max-speakers 4` if you know the exact participant count

## Example output

```json
[
  {
    "start": 12.5,
    "end": 18.2,
    "name": "Jane Doe",
    "speaker_id": "speaker_0",
    "face_id": "face_1",
    "confidence": 0.86
  }
]
```

## Requirements

- Python 3.9+
- No system ffmpeg install required (`imageio-ffmpeg` bundles one)

## Commands

```bash
interview-mapper serve [--host 127.0.0.1] [--port 4173] [--browser Island] [--open/--no-open]
interview-mapper analyze VIDEO [-o OUTPUT_DIR] [--video-dir DIR] [--extra-video PATH]...
interview-mapper finalize OUTPUT_DIR
interview-mapper screengrab OUTPUT_DIR [--timestamp SECONDS]
interview-mapper regenerate-ui OUTPUT_DIR
```

## Production workflow

1. Run `analyze` on your interview master
2. Label faces in `label.html`
3. Run `finalize`
4. Import `lower_thirds.csv` into your graphics template workflow, or use the JSON to drive a motion graphics script / Premiere extension

This is an MVP focused on wide-shot interviews and lower-thirds. Future improvements could include reference-photo enrollment, multicam support, and direct Premiere/DaVinci plugin export.
