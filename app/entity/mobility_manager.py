import threading
import time
from geopy.distance import geodesic

from app.config.logger import fed_logger
from app.entity.node import NodeIdentifier
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType


class MobilityManager:
    THRESHOLD_DISTANCE = 10

    def __init__(self, client):
        self.client = client

    def discover_edges(self):
        fed_logger.info("[Mobility] discover_edges: seed neighbors=%s", list(self.client.neighbors))
        queue = list(self.client.neighbors)
        visited = set(self.client.discovered_edges)
        fed_logger.info("[Mobility] discover_edges: discovered_edges=%s", list(self.client.neighbors))
        while queue:
            current_neighbor = queue.pop(0)
            if current_neighbor in visited:
                continue

            visited.add(current_neighbor)

            if HTTPCommunicator.get_node_type(current_neighbor) == NodeType.EDGE:
                self.client.discovered_edges.add(current_neighbor)

            neighbor_info = self.client.fetch_neighbors_from_neighbor(current_neighbor)
            for info in neighbor_info:
                new_edge = NodeIdentifier(ip=info['ip'], port=info['port'])
                if new_edge not in self.client.discovered_edges and new_edge not in visited:
                    queue.append(new_edge)

    def find_closest_edge(self) -> NodeIdentifier:
        if not self.client.node_coordinate:
            raise ValueError("Node's coordinates are not set.")

        min_distance = float('inf')
        closest_edge = None

        for edge in self.client.discovered_edges:
            edge_info = HTTPCommunicator.get_node_coordinate(edge)
            edge_coords = (edge_info['latitude'], edge_info['longitude'])
            node_coords = (self.client.node_coordinate.latitude, self.client.node_coordinate.longitude)

            distance = geodesic(node_coords, edge_coords).meters
            fed_logger.info("[Mobility] distance to %s = %.1f m", edge, distance)

            if distance < min_distance:
                min_distance = distance
                closest_edge = edge
        fed_logger.info("[Mobility] closest_edge=%s (%.1f m)", closest_edge, min_distance if min_distance < float("inf") else -1)
        return closest_edge

    def initialize_neighbors(self):
        closest_edge = self.find_closest_edge()

        if closest_edge:
            fed_logger.info("[Mobility] initialize_neighbors: add %s as primary neighbor", closest_edge)
            self.client.add_neighbor(closest_edge)
            HTTPCommunicator.add_neighbor(closest_edge, self.client.ip, self.client.port)

    def get_current_edge(self) -> NodeIdentifier:
        for neighbor in self.client.neighbors:
            if HTTPCommunicator.get_node_type(neighbor) == NodeType.EDGE:
                return neighbor
        return None

    def migrate_to_edge(self, new_edge: NodeIdentifier):
        fed_logger.info("[Mobility] migrating to %s …", new_edge)
        current_edge = self.get_current_edge()
        if current_edge:
            self.client.remove_neighbor(current_edge)

        self.client.add_neighbor(new_edge)

        HTTPCommunicator.add_neighbor(new_edge, self.client.ip, self.client.port)

        if current_edge:
            HTTPCommunicator.remove_neighbor(current_edge, self.client.ip, self.client.port)

    def monitor_and_migrate(self):
        def monitor():
            fed_logger.info("[Mobility] monitor loop started (THRESHOLD=%sm)", self.THRESHOLD_DISTANCE)
            while True:
                time.sleep(1)

                closest_edge = self.find_closest_edge()
                current_edge = self.get_current_edge()

                if current_edge:
                    current_edge_coords = HTTPCommunicator.get_node_coordinate(current_edge)
                    current_coords = (self.client.node_coordinate.latitude, self.client.node_coordinate.longitude)
                    edge_coords = (current_edge_coords['latitude'], current_edge_coords['longitude'])

                    distance_to_current_edge = geodesic(current_coords, edge_coords).meters

                    if distance_to_current_edge > self.THRESHOLD_DISTANCE and closest_edge != current_edge:
                        self.migrate_to_edge(closest_edge)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
