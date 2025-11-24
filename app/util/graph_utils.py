import csv
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


def save_results_to_csv(node: Node, training_times: list[float], client_bandwidths: list[float],
                        accuracy: list[float], save_path: str, neighbor_bandwidths: Optional[list[float]] = None):
    """Save numerical results to a CSV file."""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    csv_file = os.path.join(save_path, f"results-{str(node)}.csv")
    rounds_count = len(training_times)
    
    # Prepare data for CSV
    headers = ["Round", "Training Time (s)", "Bandwidth (bytes/s)", "Accuracy (%)"]
    if neighbor_bandwidths:
        headers.append("Neighbor Bandwidth (bytes/s)")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(rounds_count):
            row = [
                i + 1,  # Round number
                training_times[i],
                client_bandwidths[i],
                accuracy[i]
            ]
            if neighbor_bandwidths:
                row.append(neighbor_bandwidths[i])
            writer.writerow(row)
    
    fed_logger.info(f"CSV results saved to {csv_file}")


def report_results(node: Node, training_times: list[float], client_bandwidths: list[float],
                   accuracy: list[float], neighbor_bandwidths: Optional[list[float]] = None, accuracy_duration: bool = True):
    current_time = time.strftime("%Y-%m-%d %H:%M")
    runtime_config = f'{current_time} {config.SCENARIO_DESCRIPTION}'
    save_path = f"Results/{runtime_config}"
    rounds_count = config.R
    
    # Save numerical data to CSV
    save_results_to_csv(node, training_times, client_bandwidths, accuracy, save_path, neighbor_bandwidths)
    
    # Generate plots
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
