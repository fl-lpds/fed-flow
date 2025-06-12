import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config import config


def bandwidth(edge_server_BW, client_edge_bw):
    client_server_bw = []
    edge_server_bw_list = []
    client_edge_bw_list = []

    for i in range(len(config.CLIENTS_LIST)):
        edgeOfClient = config.CLIENT_MAP[config.CLIENTS_INDEX[i]]
        ClientEdge_bw = client_edge_bw[config.CLIENTS_INDEX[i]]
        client_edge_bw_list.append(ClientEdge_bw)
        edge_server_bw_list.append(edge_server_BW[edgeOfClient] / len(config.EDGE_MAP[edgeOfClient]))

    for i in range(len(client_edge_bw_list)):
        # bw between client -> server ===> B1B2 / B1 + B2
        client_server_bw.append((client_edge_bw_list[i] * edge_server_bw_list[i]) / (client_edge_bw_list[i] + edge_server_bw_list[i]))
    bandwidths = np.array(client_server_bw)
    X = bandwidths.reshape(-1, 1)

    labels = KMeans(n_clusters=config.G, random_state=42).fit_predict(X)
    return labels


def none():
    labels = []
    for c in config.CLIENTS_LIST:
        labels.append(0)
    return labels
