import os
import random

import matplotlib.pyplot as plt
import numpy as np

import config

client_number = len(config.CLIENTS_LIST)
client_edge_bw = {client: [] for client in config.CLIENTS_LIST}

base_dir = "."
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

low_range = (1_000_000, 10_000_000)     # Frequent range
high_range = (10_000_000, 15_000_000)   # Occasional range
high_probability = 0.2

for device in config.CLIENTS_LIST:
    for _ in range(100):
        if random.random() < high_probability:
            bw = round(random.uniform(*low_range), 2)
        else:
            bw = round(random.uniform(*high_range), 2)
        client_edge_bw[device].append(bw)

np.savez(f"{base_dir}/client_edge_bw.npz", **client_edge_bw)
data = np.load(f"{base_dir}/client_edge_bw.npz")

plt.figure(figsize=(int(25), int(5)))
i = 0
for k in data.keys():
    edgeDevice_K = data[k]
    plt.title(f"Bandwidth of each client")
    plt.xlabel("Round")
    plt.ylabel("Bandwidth (bits/sec)")
    plt.plot(edgeDevice_K, linewidth='3', label=f"Client: {k}")
    i = i + 1
plt.legend()
plt.grid()
plt.savefig(f"client_edge_BW")
plt.close()
