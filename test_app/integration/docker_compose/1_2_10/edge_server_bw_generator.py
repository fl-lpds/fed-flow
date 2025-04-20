import numpy as np
import os
import random
import config

edge_number = len(config.EDGE_SERVER_LIST)
edge_server_bw = {edge: [] for edge in config.EDGE_SERVER_LIST}

base_dir = "."
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for r in range(100):
    if r < 15:
        for edgeIndex in range(edge_number):
            bw = random.randint(45_000_000, 50_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 15 <= r < 20:
        for edgeIndex in range(edge_number):
            bw = random.randint(50_000, 500_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 20 <= r < 45:
        for edgeIndex in range(edge_number):
            bw = random.randint(35_000_000, 40_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 45 <= r < 60:
        for edgeIndex in range(edge_number):
            bw = random.randint(10_000_000, 15_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 60 <= r < 80:
        for edgeIndex in range(edge_number):
            bw = random.randint(50_000_000, 55_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 80 <= r < 100:
        for edgeIndex in range(edge_number):
            bw = random.randint(45_000_000, 50_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
#np.savez(f"{base_dir}/edge_server_bw.npz", **edge_server_bw)
data = np.load(f"{base_dir}/edge_server_bw.npz")
print(data['edge1'])