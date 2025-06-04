import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config import config


def bandwidth(edge_server_BW):
    client_server_bw = []
    edge_server_bw_list = []

    # Constant: Bandwidth between device and edge
    client_edge_BW = config.CLIENTS_BANDWIDTH

    for i in range(len(config.CLIENTS_LIST)):
        edge_server_bw_list.append(edge_server_BW[config.CLIENT_MAP[config.CLIENTS_INDEX[i]]])

        # bw between client -> server ===> B1B2 / B1 + B2
        client_server_bw.append((client_edge_BW[i] * edge_server_bw_list[i]) / (client_edge_BW[i] + edge_server_bw_list[i]))

    bandwidths = np.array(client_server_bw)
    X = bandwidths.reshape(-1, 1)

    labels = KMeans(n_clusters=config.G, random_state=42).fit_predict(X)
    return labels


def none():
    labels = []
    for c in config.CLIENTS_LIST:
        labels.append(0)
    return labels
