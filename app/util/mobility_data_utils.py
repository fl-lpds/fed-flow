import os
import threading
import time
import logging

import pandas as pd

from app.entity.node import Node

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_user_data():
    csv_file = '/fed-flow/app/dataset/mobility_data/data.csv'
    user_data = pd.read_csv(csv_file)

    return user_data


def simulate_real_time_update(node: Node, user_data):
    max_seconds = user_data['Seconds_Since_Start'].max()

    for current_second in range(0, int(max_seconds) + 1):
        current_data = user_data[user_data['Seconds_Since_Start'] <= current_second].iloc[-1]

        node.update_coordinates(new_latitude=current_data['Latitude'],
                                new_longitude=current_data['Longitude'],
                                new_altitude=current_data['Altitude'],
                                new_seconds_since_start=current_second)

        # Add logging line here to trace mobility
        logger.info(f"[Mobility] Time={current_second}s | Lat={current_data['Latitude']} Lon={current_data['Longitude']}")

        # Optional: Use print instead if logger not visible
        # print(f"[Mobility] Time={current_second}s | Lat={current_data['Latitude']} Lon={current_data['Longitude']}")

        time.sleep(1)


def start_mobility_simulation_thread(node: Node):
    user_data = load_user_data()
    simulation_thread = threading.Thread(target=simulate_real_time_update, args=(node, user_data))
    simulation_thread.start()
