import os
import threading
import time

import pandas as pd

from app.entity.node import Node

CLIENT_COORDINATE_OFFSETS = {
    8080: (0.0, 0.0),          # client1
    8082: (0.00030, 0.00020),  # client2
    8084: (-0.00025, 0.00010),  # client3

    8086: (0.0, 0.0),          # client4
    8088: (0.00030, -0.00020),  # client5
    8090: (-0.00025, -0.00010),  # client6

    8092: (0.0, 0.0),          # client7
    8094: (0.00040, 0.00025),  # client8
    8096: (-0.00035, 0.00015),  # client9

    8098: (0.0, 0.0),          # client10
    8100: (0.00040, -0.00025),  # client11
    8102: (-0.00035, -0.00015)  # client12
}


def get_client_offset(node: Node):
    """
    براساس پورت کلاینت، offset مختصات رو برمی‌گردونه
    اگر پورتی توی دیکشنری نبود، offset = 0 در نظر می‌گیریم
    """
    port = getattr(node, "port", None)
    return CLIENT_COORDINATE_OFFSETS.get(port, (0.0, 0.0))


def load_user_data():
    csv_file = '/fed-flow/app/dataset/mobility_data/data.csv'
    user_data = pd.read_csv(csv_file)

    return user_data


def simulate_real_time_update(node: Node, user_data):
    max_seconds = user_data['Seconds_Since_Start'].max()

    # اینجا یک‌بار offset اختصاصی همین کلاینت رو می‌گیریم
    lat_offset, lon_offset = get_client_offset(node)

    for current_second in range(0, int(max_seconds) + 1):
        current_data = user_data[user_data['Seconds_Since_Start'] <= current_second].iloc[-1]

        base_lat = current_data['Latitude']
        base_lon = current_data['Longitude']

        node.update_coordinates(
            new_latitude=base_lat + lat_offset,
            new_longitude=base_lon + lon_offset,
            new_altitude=current_data['Altitude'],
            new_seconds_since_start=current_second
        )

        time.sleep(1)


def start_mobility_simulation_thread(node: Node):
    user_data = load_user_data()
    simulation_thread = threading.Thread(target=simulate_real_time_update, args=(node, user_data))
    simulation_thread.start()
