from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Interview Face Labeler</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, system-ui, sans-serif;
    }
    body {
      margin: 0;
      padding: 24px;
      background: #111;
      color: #f3f3f3;
    }
    h1 { margin-top: 0; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 16px;
      margin: 24px 0;
    }
    .card {
      background: #1d1d1d;
      border: 1px solid #333;
      border-radius: 12px;
      padding: 12px;
    }
    img {
      width: 100%;
      border-radius: 8px;
      aspect-ratio: 1;
      object-fit: cover;
      background: #000;
    }
    label {
      display: block;
      margin-top: 10px;
      font-size: 12px;
      color: #aaa;
    }
    input {
      width: 100%;
      margin-top: 6px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid #444;
      background: #101010;
      color: #fff;
      box-sizing: border-box;
    }
    button {
      background: #4f7cff;
      color: white;
      border: 0;
      border-radius: 10px;
      padding: 12px 18px;
      font-size: 15px;
      cursor: pointer;
    }
    .meta { color: #888; font-size: 13px; }
    pre {
      background: #101010;
      border: 1px solid #333;
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
    }
  </style>
</head>
<body>
  <h1>Label detected faces</h1>
  <p class="meta">Assign a display name to each detected face cluster. Save the JSON below as <code>labels.json</code> in your output folder, then run <code>interview-mapper finalize</code>.</p>
  <div class="grid" id="grid"></div>
  <button id="exportBtn">Generate labels.json</button>
  <h2>labels.json</h2>
  <pre id="output">{}</pre>
  <script>
    const clusters = __CLUSTERS__;
    const grid = document.getElementById("grid");
    const output = document.getElementById("output");
    const inputs = {};

    clusters.forEach((cluster) => {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `
        <img src="${cluster.thumbnail_path}" alt="${cluster.cluster_id}" />
        <label>${cluster.cluster_id} (${cluster.sample_count} samples)
          <input type="text" placeholder="Full name" data-id="${cluster.cluster_id}" />
        </label>
      `;
      grid.appendChild(card);
      const input = card.querySelector("input");
      inputs[cluster.cluster_id] = input;
      input.addEventListener("input", render);
    });

    function render() {
      const labels = {};
      Object.entries(inputs).forEach(([id, input]) => {
        if (input.value.trim()) {
          labels[id] = input.value.trim();
        }
      });
      output.textContent = JSON.stringify(labels, null, 2);
    }

    document.getElementById("exportBtn").addEventListener("click", () => {
      render();
      const blob = new Blob([output.textContent], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "labels.json";
      link.click();
    });
  </script>
</body>
</html>
"""


def generate_label_ui(output_dir: Path, faces_payload: dict) -> Path:
    clusters = faces_payload.get("face_clusters", [])
    html = HTML_TEMPLATE.replace("__CLUSTERS__", json.dumps(clusters))
    ui_path = output_dir / "label.html"
    ui_path.write_text(html, encoding="utf-8")
    return ui_path


def load_labels(output_dir: Path) -> Dict[str, str]:
    labels_path = output_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path}. Open label.html, assign names, and save labels.json first."
        )
    return json.loads(labels_path.read_text(encoding="utf-8"))
