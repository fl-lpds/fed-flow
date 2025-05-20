import os
import random

import matplotlib.pyplot as plt
import numpy as np

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
    elif 20 <= r < 35:
        for edgeIndex in range(edge_number):
            bw = random.randint(500_000, 2_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 35 <= r < 40:
        for edgeIndex in range(edge_number):
            bw = random.randint(2_000_000, 10_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 40 <= r < 60:
        for edgeIndex in range(edge_number):
            bw = random.randint(10_000_000, 15_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 60 <= r < 75:
        for edgeIndex in range(edge_number):
            bw = random.randint(15_000_000, 25_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
    elif 75 <= r < 100:
        for edgeIndex in range(edge_number):
            bw = random.randint(25_000_000, 50_000_000)
            edge_server_bw[config.EDGE_SERVER_CONFIG[edgeIndex]].append(bw)
# np.savez(f"{base_dir}/edge_server_bw.npz", **edge_server_bw)
data = np.load(f"{base_dir}/edge_server_bw.npz")
print(data['edge1'])

plt.figure(figsize=(int(25), int(5)))
i = 0
for k in data.keys():
    edgeDevice_K = data[k]
    plt.title(f"Bandwidth of  each edge server")
    plt.xlabel("Round")
    plt.ylabel("Bandwidth (bits/sec)")
    plt.plot(edgeDevice_K, color='Blue', linewidth='3', label=f"Edge Server: {k}")
    i = i + 1
plt.legend()
plt.grid()
plt.savefig(f"edge_serverBW")
plt.close()
