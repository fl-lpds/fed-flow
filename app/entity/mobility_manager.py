import threading
import time
from geopy.distance import geodesic

from app.config.logger import fed_logger
from app.entity.node_identifier import NodeIdentifier
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType

fed_logger.info("[BOOT] mobility_manager LOADED at %s", __file__)


class MobilityManager:
    THRESHOLD_DISTANCE = 150

    def __init__(self, client):
        self.client = client
        self._migration_enabled = True
        self._lock = threading.Lock()

    def enable_migration(self):
        with self._lock:
            self._migration_enabled = True
        fed_logger.info("[Mobility] migration ENABLED (between rounds)")

    def disable_migration(self):
        with self._lock:
            self._migration_enabled = False
        fed_logger.info("[Mobility] migration DISABLED (during training)")


    def discover_edges(self):
        fed_logger.info("[Mobility] discover_edges: seed neighbors=%s", list(self.client.neighbors))
        queue = list(self.client.neighbors)
        visited = set(self.client.discovered_edges)
        fed_logger.info("[Mobility] discover_edges: discovered_edges=%s", list(self.client.discovered_edges))
        while queue:
            current_neighbor = queue.pop(0)
            if current_neighbor in visited:
                continue

            visited.add(current_neighbor)

            if HTTPCommunicator.get_node_type(current_neighbor) == NodeType.EDGE:
                self.client.discovered_edges.add(current_neighbor)

            neighbor_info = self.client.fetch_neighbors_from_neighbor(current_neighbor) or []
            for info in neighbor_info:
                if isinstance(info, NodeIdentifier):
                    new_edge = info
                elif isinstance(info, dict):
                    new_edge = NodeIdentifier(ip=info['ip'], port=info['port'])
                else:
                    continue
                if new_edge not in self.client.discovered_edges and new_edge not in visited:
                    queue.append(new_edge)

    def find_closest_edge(self) -> NodeIdentifier:
        if not self.client.node_coordinate:
            raise ValueError("Node's coordinates are not set.")

        min_distance = float('inf')
        closest_edge = None

        for edge in self.client.discovered_edges:
            edge_info = HTTPCommunicator.get_node_coordinate(edge)
            if not edge_info:
            # این edge هنوز مختصات نداده (404) یا خطای قابل‌هندل بوده؛ ردش کن
                continue
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

        # اگر edge پیدا شد، همه‌ی edgeهای قبلی را پاک کن و فقط همین را نگه داریم
        if closest_edge:
            # اختیاری: پاک کردن edgeهای قبلی از لیست همسایه‌ها
            edges_to_remove = []
            for n in list(self.client.neighbors):
                if HTTPCommunicator.get_node_type(n) == NodeType.EDGE:
                    edges_to_remove.append(n)
            for old_edge in edges_to_remove:
                self.client.remove_neighbor(old_edge)
                HTTPCommunicator.remove_neighbor(old_edge, self.client.ip, self.client.port)

            fed_logger.info("[Mobility] initialize_neighbors: add %s as primary neighbor", closest_edge)
            # هم در خود کلاینت، هم سمت اج ثبتش کن
            self.client.add_neighbor(closest_edge)
            fed_logger.info("[Mobility] add_neighbor() done; calling HTTPCommunicator.add_neighbor ...")
            HTTPCommunicator.add_neighbor(closest_edge, self.client.ip, self.client.port)
            fed_logger.info(
                "[Mobility] CONNECTED client=%s:%s -> edge=%s",
                self.client.ip, self.client.port, closest_edge
            )
        else:
            fed_logger.warning("[Mobility] No edge has coordinates yet; skipping initial neighbor setup for now.")
        return

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
            HTTPCommunicator.remove_neighbor(current_edge, self.client.ip, self.client.port)

        self.client.add_neighbor(new_edge)
        HTTPCommunicator.add_neighbor(new_edge, self.client.ip, self.client.port)

        if current_edge:
            HTTPCommunicator.remove_neighbor(current_edge, self.client.ip, self.client.port)
        fed_logger.info("[Mobility] MIGRATED. Now connected edge = %s", self.get_current_edge())


    def check_and_migrate_per_round(self):
        """
        این تابع فقط یک‌بار (مثلاً اول هر راند) صدا زده می‌شود:
        فاصله تا اج فعلی را حساب می‌کند و اگر از THRESHOLD بیشتر بود
        و یک اج نزدیک‌تر پیدا شد، مهاجرت انجام می‌دهد.
        """
        # اگر هنوز مختصات کلاینت ست نشده، هیچی نکن
        if not self.client.node_coordinate:
            fed_logger.warning("[Mobility] per-round check: client has no coordinates yet")
            return

        current_edge = self.get_current_edge()
        if current_edge is None:
            fed_logger.warning("[Mobility] per-round check: no current EDGE neighbor, skipping")
            return

        closest_edge = self.find_closest_edge()
        if closest_edge is None:
            fed_logger.warning("[Mobility] per-round check: no closest edge found")
            return

        current_edge_coords = HTTPCommunicator.get_node_coordinate(current_edge)
        if not current_edge_coords:
            fed_logger.warning("[Mobility] per-round check: current edge has no coordinates")
            return

        current_coords = (self.client.node_coordinate.latitude, self.client.node_coordinate.longitude)
        edge_coords = (current_edge_coords['latitude'], current_edge_coords['longitude'])

        distance_to_current_edge = geodesic(current_coords, edge_coords).meters

        fed_logger.info(
            "[Mobility] per-round check: current_edge=%s distance=%.1f m, closest_edge=%s",
            current_edge, distance_to_current_edge, closest_edge
        )

        if distance_to_current_edge > self.THRESHOLD_DISTANCE and closest_edge != current_edge:
            fed_logger.info(
                "[Mobility] per-round check: migrating from %s to %s (distance=%.1f m > %s m)",
                current_edge, closest_edge, distance_to_current_edge, self.THRESHOLD_DISTANCE
            )
            self.migrate_to_edge(closest_edge)
        else:
            fed_logger.info(
                "[Mobility] per-round check: stay on %s (distance=%.1f m, closest_edge=%s)",
                current_edge, distance_to_current_edge, closest_edge
            )

    def monitor_and_migrate(self):
        def monitor():
            fed_logger.info("[Mobility] monitor loop started (THRESHOLD=%sm)", self.THRESHOLD_DISTANCE)
            while True:
                time.sleep(1)

                with self._lock:
                    can_migrate = self._migration_enabled
                if not can_migrate:
                    continue

                closest_edge = self.find_closest_edge()
                current_edge = self.get_current_edge()

                if current_edge:
                    current_edge_coords = HTTPCommunicator.get_node_coordinate(current_edge)
                    if not current_edge_coords:
                        continue
                    current_coords = (self.client.node_coordinate.latitude, self.client.node_coordinate.longitude)
                    edge_coords = (current_edge_coords['latitude'], current_edge_coords['longitude'])

                    distance_to_current_edge = geodesic(current_coords, edge_coords).meters

                    if distance_to_current_edge > self.THRESHOLD_DISTANCE and closest_edge != current_edge:
                        if closest_edge is None:
                            continue
                        self.migrate_to_edge(closest_edge)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
