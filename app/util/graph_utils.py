import os
import random
import shutil
import time
import glob
import html

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import app.util.model_utils as model_utils
from app.config import config
from app.config.logger import fed_logger
from app.entity.node import Node


def report_results(node: Node, training_times: list[float], client_bandwidths: list[float],
                   accuracy: list[float], neighbor_bandwidths: Optional[list[float]] = None, accuracy_duration: bool = True):
    current_time = time.strftime("%Y-%m-%d %H:%M")
    runtime_config = f'{current_time} {config.SCENARIO_DESCRIPTION}'
    save_path = f"Results/{runtime_config}"
    rounds_count = config.R
    draw_graph(10, 5, range(1, rounds_count + 1), training_times, str(node), "FL Rounds", "Training Time (s)",
               save_path, f"training-time-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), client_bandwidths, str(node), "FL Rounds", "Bandwidths (bytes/s)",
               save_path, f"bandwidth-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), accuracy, str(node), "FL Rounds", "Accuracy (%)",
               save_path, f"accuracy-{str(node)}")
    if neighbor_bandwidths:
        draw_graph(10, 5, range(1, rounds_count + 1), neighbor_bandwidths, str(node), "FL Rounds",
                   "Neighbors Bandwidths (bytes/s)",
                   save_path, f"neighbor-bandwidths-{str(node)}")
    if accuracy_duration:
        timeline = [0]
        for duration in training_times:
            timeline.append(timeline[-1] + duration)
        draw_graph(10, 5, timeline[1:], accuracy, str(node), "Time (s)", "Accuracy (%)",
                   save_path, f"accuracy-duration-{str(node)}")
    copy_compose_file_if_exists(save_path)
    fed_logger.info(f"Results created successfully at {save_path}")
    #این خط رو اضافه کردم
    _write_results_index(results_root="Results", limit=5)


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()


def copy_compose_file_if_exists(dest):
    src = 'evaluation/docker-compose.yml'
    dest += '/docker-compose.yml'
    if os.path.isfile(src):
        try:
            shutil.copy(src, dest)
            fed_logger.info(f"File '{src}' copied to '{dest}' successfully.")
        except Exception as e:
            print(f"Failed to copy file: {e}")
    else:
        print(f"File '{src}' does not exist.")

def _scan_last_runs(results_root: str, limit: int = 5):
    runs = []
    if not os.path.isdir(results_root):
        return runs
    for d in os.listdir(results_root):
        path = os.path.join(results_root, d)
        if not os.path.isdir(path):
            continue
        # فقط پوشه‌هایی که داخلشان png یا yml دارند را به عنوان "run" می‌شناسیم
        pngs = glob.glob(os.path.join(path, "*.png"))
        order = ["accuracy-duration", "accuracy", "training-time", "bandwidth", "neighbor-bandwidths"]

        def sort_key(p):
            name = os.path.basename(p)
            for i, k in enumerate(order):
                if k in name: return (i, name)
            return (len(order), name)

        imgs = sorted(pngs, key=sort_key)

        compose = glob.glob(os.path.join(path, "docker-compose.yml"))
        if not imgs and not compose:
            continue
        runs.append({
            "name": d,
            "path": path,
            "mtime": os.path.getmtime(path),
            "images": [os.path.relpath(p, results_root) for p in imgs]
        })
    runs.sort(key=lambda x: x["mtime"], reverse=True)
    return runs[:limit]

def _write_results_index(results_root: str = "Results", limit: int = 5):
    os.makedirs(results_root, exist_ok=True)
    last = _scan_last_runs(results_root, limit)
    # HTML ساده و بدون وابستگی
    cards = []
    for r in last:
        title = html.escape(r["name"])
        imgs_html = "\n".join(
            f'<a href="{html.escape(img)}" target="_blank"><img loading="lazy" src="{html.escape(img)}" style="max-width: 360px; height: auto; margin: 6px; border: 1px solid #ddd; border-radius: 8px;" /></a>'
            for img in r["images"]
        ) or "<em>No images found for this run.</em>"
        cards.append(f"""
        <section style="background:#fff; border:1px solid #eee; border-radius:16px; padding:16px; margin:16px 0; box-shadow:0 2px 6px rgba(0,0,0,0.06);">
          <h2 style="margin:0 0 8px 0; font-size:18px;">{title}</h2>
          <div style="display:flex; flex-wrap:wrap; gap:8px;">{imgs_html}</div>
          <div style="margin-top:8px;">
            <a href="{html.escape(title)}/docker-compose.yml" target="_blank">docker-compose.yml</a>
          </div>
        </section>
        """)
    body = "\n".join(cards) or "<p>No runs found yet.</p>"


    latest = (last[0]["name"] if last else "#")
    html_doc = f"""<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta content="width=device-width, initial-scale=1" name="viewport" />
      <title>fed-flow – Last 5 runs</title>
      <style>
        :root {{ --fg:#111; --sub:#666; --bg:#f6f7fb; --card:#fff; --bd:#e9e9ef; }}
        * {{ box-sizing:border-box; }}
        body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial; background:var(--bg); color:var(--fg); }}
        header {{ position:sticky; top:0; background:#ffffffcc; backdrop-filter: blur(8px); border-bottom:1px solid var(--bd); padding:16px 20px; font-weight:800; font-size:22px; }}
        main {{ max-width: 1200px; margin: 28px auto; padding: 0 16px; }}
        a {{ color:#0b6bcb; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
        .card {{ background:var(--card); border:1px solid var(--bd); border-radius:18px; padding:16px; margin:16px 0; box-shadow:0 2px 10px rgba(0,0,0,.05); }}
        .head {{ display:flex; gap:12px; align-items:baseline; justify-content:space-between; flex-wrap:wrap; }}
        h2 {{ margin:0; font-size:22px; line-height:1.2; }}
        .mtime {{ color:var(--sub); font-size:14px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(300px,1fr)); gap:12px; margin-top:12px; }}
        .grid img {{ width:100%; height:auto; display:block; border:1px solid var(--bd); border-radius:12px; }}
        .links {{ margin-top:10px; }}
        .yml::before {{ content:"↗ "; }}
        .empty {{ padding:32px; text-align:center; color:var(--sub); }}
        .toolbar {{ display:flex; gap:10px; align-items:center; margin: 14px 0; }}
        .btn {{ display:inline-block; padding:8px 12px; border-radius:10px; border:1px solid var(--bd); background:#fff; font-weight:600; }}
        .btn:hover {{ background:#f0f3f8; }}
      </style>
    </head>
    <body>
      <header>fed-flow · آخرین ۵ ران</header>
      <main>
        <div class="toolbar">
          <a class="btn" href="{html.escape(latest)}">Open latest run</a>
        </div>
        {body}
      </main>
    </body>
    </html>"""

    with open(os.path.join(results_root, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_doc)
