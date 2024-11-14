from app.config import config
from app.entity.node import Node
from app.entity.node_type import NodeType


def bandwidth(node: Node):
    # sort bandwidth in test_config.CLIENTS_LIST order
    bandwidth = config.CLIENTS_BANDWIDTH
    bandwidth_order = []
    for c in config.CLIENTS_LIST:
        bandwidth_order.append(bandwidth[c])

    labels = [0, 0, 1, 0, 0]  # Previous clustering results in RL
    for i in range(len(bandwidth_order)):
        if bandwidth_order[i] < 5:
            labels[i] = 2  # If network speed is limited under 5Mbps, we assign the device into group 2

    return labels


def none(node: Node):
    labels = []
    for client in node.get_neighbors([NodeType.CLIENT]):
        labels.append(0)
    return labels
