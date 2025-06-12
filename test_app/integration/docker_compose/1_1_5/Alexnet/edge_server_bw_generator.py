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

# Edge ↔ Server: 50–100 Mbps (80%), 100–150 Mbps (20%)
for edge in edge_server_bw:
    for _ in range(100):
        chance = random.random()
        if chance < 0.2:
          bw = round(random.uniform(75_000_000, 100_000_000), 2)
        elif 0.2 <= chance < 0.4:
            bw = round(random.uniform(100_000_000, 150_000_000), 2)
        elif chance > 0.90:
            bw = round(random.uniform(3_000_000, 5_000_000), 2)
        else:
            bw = round(random.uniform(50_000_000, 100_000_000), 2)
        edge_server_bw[edge].append(bw)

np.savez(f"{base_dir}/edge_server_bw.npz", **edge_server_bw)
data = np.load(f"{base_dir}/edge_server_bw.npz")
#print(data['edge1'])

plt.figure(figsize=(int(25), int(5)))
i = 0
for k in data.keys():
    edgeDevice_K = data[k]
    plt.title(f"Bandwidth of  each edge server test")
    plt.xlabel("Round")
    plt.ylabel("Bandwidth (bits/sec)")
    plt.plot(edgeDevice_K, color='Blue', linewidth='3', label=f"Edge Server: {k}")
    i = i + 1
plt.legend()
plt.grid()
plt.savefig(f"edge_serverBW")
plt.close()
