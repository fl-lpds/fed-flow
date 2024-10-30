import os
import random
import shutil
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import app.util.model_utils as model_utils
from app.config import config
from app.config.logger import fed_logger
from app.entity.node import Node


def report_results(node: Node, training_times: list[float], client_bandwidths: list[float],
                   accuracy: list[float], neighbor_bandwidths: Optional[list[float]] = None):
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
    copy_compose_file_if_exists(save_path)
    fed_logger.info(f"Results created successfully at {save_path}")


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
