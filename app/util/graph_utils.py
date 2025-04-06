import os
import shutil
import csv
from typing import Optional

import matplotlib.pyplot as plt

from app.config import config
from app.config.logger import fed_logger
from app.entity.node import Node

_runtime_config = None


def set_runtime_config(config_value: str):
    global _runtime_config
    _runtime_config = config_value


def get_runtime_config():
    return _runtime_config


def init_round_csv(node_name: str):
    save_path = f"Results/{_runtime_config}"
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f"round_log-{node_name}.csv")
    if not os.path.isfile(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Training Time (s)", "Bandwidth (bytes/s)", "Accuracy (%)"])
    return filepath


def log_round_csv(node_name: str, round_id: int, training_time: float, bandwidth: float,
                  accuracy: float):
    filepath = init_round_csv(node_name)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round_id + 1, training_time, bandwidth, accuracy])


def report_results(node: Node, training_times: list[float], client_bandwidths: list[float],
                   accuracy: list[float], neighbor_bandwidths: Optional[list[float]] = None,
                   accuracy_duration: bool = True):
    save_path = f"Results/{_runtime_config}"
    rounds_count = config.R

    draw_graph(10, 5, range(1, rounds_count + 1), training_times, str(node), "FL Rounds", "Training Time (s)",
               save_path, f"training-time-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), client_bandwidths, str(node), "FL Rounds", "Bandwidths (bytes/s)",
               save_path, f"bandwidth-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), accuracy, str(node), "FL Rounds", "Accuracy (%)",
               save_path, f"accuracy-{str(node)}")

    if neighbor_bandwidths:
        draw_graph(10, 5, range(1, rounds_count + 1), neighbor_bandwidths, str(node), "FL Rounds",
                   "Neighbors Bandwidths (bytes/s)", save_path, f"neighbor-bandwidths-{str(node)}")

    if accuracy_duration:
        timeline = [0]
        for duration in training_times:
            timeline.append(timeline[-1] + duration)
        draw_graph(10, 5, timeline[1:], accuracy, str(node), "Time (s)", "Accuracy (%)",
                   save_path, f"accuracy-duration-{str(node)}")

    log_file_path = os.path.join(save_path, f"round_log-{str(node)}.csv")
    if os.path.isfile(log_file_path):
        shutil.copy(log_file_path, os.path.join(save_path, f"results-{str(node)}.csv"))

    copy_compose_file_if_exists(save_path)
    fed_logger.info(f"Results created successfully at {save_path}")


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig:
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
