import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.config import config


def bandwidth(edge_server_BW):
    edge_server_bw_list = []

    # Constant: Bandwidth between device and edge
    client_edge_BW = config.CLIENTS_BANDWIDTH

    for client in config.CLIENTS_LIST:
        edge_server_bw_list.append(edge_server_BW[config.CLIENT_MAP[client]])

    # Stack features into a 2D array
    features = np.column_stack((client_edge_BW, edge_server_bw_list))

    # Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(features_scaled)

    labels = []

    # Print device clusters
    device_clusters = list(zip(client_edge_BW, edge_server_bw_list, clusters))
    print("Device Clusters (Device↔Edge BW, Edge↔Server BW, Cluster ID):")
    for dc in device_clusters:
        labels.append(dc[2])

    return labels


def none():
    labels = []
    for c in config.CLIENTS_LIST:
        labels.append(0)
    return labels
