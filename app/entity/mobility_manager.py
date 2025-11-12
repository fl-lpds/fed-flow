import threading
import time
from geopy.distance import geodesic
from typing import Optional
from app.config.logger import fed_logger
from app.entity.node_identifier import NodeIdentifier
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType

fed_logger.info("[BOOT] mobility_manager LOADED at %s", __file__)


class MobilityManager:
    """
    Manages client mobility between edge servers.
    - Discovers available edges
    - Initializes connection to the closest edge
    - Monitors position and migrates when a better edge is available
    """
    # آستانه‌ی فاصله برای فیل‌سیف
    THRESHOLD_DISTANCE = 150  # meters

    def __init__(self, client):
        self.client = client

    # ----------------------------
    # Edge discovery
    # ----------------------------
    def discover_edges(self):
        """
        Fill client.discovered_edges with reachable edge nodes.
        Strategy:
          - Start from current neighbors (seed)
          - Keep only edges; try to expand via HTTPCommunicator.get_neighbors if available
        """
        fed_logger.info("[Mobility] discover_edges: seed neighbors=%s", list(self.client.neighbors))
        queue = list(self.client.neighbors)
        visited = set(getattr(self.client, "discovered_edges", set()))
        discovered = set(getattr(self.client, "discovered_edges", set()))
        fed_logger.info("[Mobility] discover_edges: discovered_edges(initial)=%s", list(discovered))

        while queue:
            current_neighbor = queue.pop(0)
            if current_neighbor in visited:
                continue
            visited.add(current_neighbor)

            # نوع نود را بگیر
            ntype = None
            try:
                ntype = HTTPCommunicator.get_node_type(current_neighbor)
            except Exception as e:
                fed_logger.warning("[Mobility] discover_edges: get_node_type(%s) failed: %s", current_neighbor, e)

            # اگر Edge است، به لیست کشف‌شده‌ها اضافه شود
            if ntype == NodeType.EDGE:
                discovered.add(current_neighbor)

            # سعی کن همسایه‌های این نود را هم کشف کنی (اگر API در دسترس است)
            try:
                # بعضی نسخه‌ها ممکن است فیلتر نوع بپذیرند؛ این‌جا محافظه‌کارانه صدا می‌زنیم:
                next_neighbors = []
                try:
                    next_neighbors = HTTPCommunicator.get_neighbors(current_neighbor, [NodeType.EDGE])
                except TypeError:
                    # امضای تابع فیلتر نمی‌گیرد
                    next_neighbors = HTTPCommunicator.get_neighbors(current_neighbor)

                if next_neighbors:
                    for new_edge in next_neighbors:
                        try:
                            if HTTPCommunicator.get_node_type(new_edge) == NodeType.EDGE:
                                if new_edge not in discovered:
                                    discovered.add(new_edge)
                                if new_edge not in visited:
                                    queue.append(new_edge)
                        except Exception:
                            # اگر تشخیص نوع خطا داد، فقط در صف کشف اضافه‌اش نکن
                            continue
            except Exception as e:
                # اگر API وجود ندارد/خطا داد، ادامه بده
                fed_logger.debug("[Mobility] discover_edges: expand failed for %s: %s", current_neighbor, e)
                continue

        # ثبت نهایی
        self.client.discovered_edges = discovered
        fed_logger.info("[Mobility] discover_edges: discovered_edges(final)=%s", list(self.client.discovered_edges))

    # ----------------------------
    # Pick closest edge
    # ----------------------------
    def find_closest_edge(self) -> Optional[NodeIdentifier]:
        """
        Return the closest edge among client.discovered_edges based on geodesic distance.
        """
        if not self.client.node_coordinate:
            raise ValueError("Node's coordinates are not set.")

        node_lat = self.client.node_coordinate.latitude
        node_lon = self.client.node_coordinate.longitude
        node_coords = (node_lat, node_lon)

        min_distance = float("inf")
        closest_edge = None

        for edge in list(getattr(self.client, "discovered_edges", [])):
            try:
                info = HTTPCommunicator.get_node_coordinate(edge)
                if not info:
                    # Edge without coordinates yet (e.g., 404) -> skip
                    continue
                edge_coords = (info["latitude"], info["longitude"])
                distance = geodesic(node_coords, edge_coords).meters
                fed_logger.info("[Mobility] distance to %s = %.1f m", edge, distance)

                if distance < min_distance:
                    min_distance = distance
                    closest_edge = edge
            except Exception as e:
                fed_logger.debug("[Mobility] find_closest_edge: skip %s due to %s", edge, e)
                continue

        if closest_edge is not None:
            fed_logger.info("[Mobility] closest_edge=%s (%.1f m)", closest_edge, min_distance)
        else:
            fed_logger.warning("[Mobility] closest_edge=NULL (no edge coordinates available)")
        return closest_edge

    # ----------------------------
    # Initialize neighbors (single active edge)
    # ----------------------------
    def initialize_neighbors(self):
        """
        Ensure the client is connected to exactly one edge at start.
        - Remove any existing edge neighbors locally and notify old edges via HTTP
        - Connect to the closest edge
        """
        closest_edge = self.find_closest_edge()

        # پاکسازی همه‌ی Edgeهای قبلی از همسایه‌ها
        edges_to_remove = []
        for n in list(self.client.neighbors):
            try:
                if HTTPCommunicator.get_node_type(n) == NodeType.EDGE:
                    edges_to_remove.append(n)
            except Exception:
                # اگر تشخیص نوع خطا داد، ریسک حذف نکن
                continue

        for old_edge in edges_to_remove:
            try:
                self.client.remove_neighbor(old_edge)
            except Exception as e:
                fed_logger.warning("[Mobility] initialize_neighbors: local remove %s failed: %s", old_edge, e)
            try:
                HTTPCommunicator.remove_neighbor(old_edge, self.client.ip, self.client.port)
            except Exception as e:
                fed_logger.warning("[Mobility] initialize_neighbors: HTTP remove to %s failed: %s", old_edge, e)

        if closest_edge:
            fed_logger.info("[Mobility] initialize_neighbors: add %s as primary neighbor", closest_edge)
            try:
                self.client.add_neighbor(closest_edge)
            except Exception as e:
                fed_logger.warning("[Mobility] initialize_neighbors: local add %s failed: %s", closest_edge, e)

            fed_logger.info("[Mobility] add_neighbor() done; calling HTTPCommunicator.add_neighbor ...")
            try:
                HTTPCommunicator.add_neighbor(closest_edge, self.client.ip, self.client.port)
            except Exception as e:
                fed_logger.warning("[Mobility] initialize_neighbors: HTTP add to %s failed: %s", closest_edge, e)

            fed_logger.info("[Mobility] CONNECTED client=%s:%s -> edge=%s",
                            self.client.ip, self.client.port, closest_edge)
        else:
            fed_logger.warning("[Mobility] No edge has coordinates yet; skipping initial neighbor setup for now.")

    # ----------------------------
    # Current edge helper
    # ----------------------------
    def get_current_edge(self) -> Optional[NodeIdentifier]:
        """
        Return the first neighbor of type EDGE from client.neighbors.
        """
        for neighbor in list(self.client.neighbors):
            try:
                if HTTPCommunicator.get_node_type(neighbor) == NodeType.EDGE:
                    return neighbor
            except Exception:
                continue
        return None

    # ----------------------------
    # Migration
    # ----------------------------
    def migrate_to_edge(self, new_edge: NodeIdentifier):
        """
        Switch connection from current edge to new_edge.
        Steps:
          - If already on new_edge, skip
          - Remove old edge locally and notify it
          - Add new edge locally and notify it
          - Ensure discovered_edges contains new_edge
        """
        current_edge = self.get_current_edge()

        # اگر در عمل برابرند، مهاجرت نکن
        if current_edge and (current_edge.ip, current_edge.port) == (new_edge.ip, new_edge.port):
            fed_logger.info("[Mobility] migrate_to_edge: already on %s, skip.", new_edge)
            return

        fed_logger.info("[Mobility] migrate_to_edge: old=%s -> new=%s", current_edge, new_edge)

        # 1) حذف اتصال قبلی
        if current_edge:
            try:
                self.client.remove_neighbor(current_edge)
            except Exception as e:
                fed_logger.warning("[Mobility] remove_neighbor(local) %s failed: %s", current_edge, e)
            try:
                HTTPCommunicator.remove_neighbor(current_edge, self.client.ip, self.client.port)
            except Exception as e:
                fed_logger.warning("[Mobility] remove_neighbor(HTTP) %s failed: %s", current_edge, e)

        # 2) افزودن اتصال جدید
        try:
            self.client.add_neighbor(new_edge)
        except Exception as e:
            fed_logger.warning("[Mobility] add_neighbor(local) %s failed: %s", new_edge, e)

        fed_logger.info("[Mobility] add_neighbor() done; calling HTTPCommunicator.add_neighbor ...")
        try:
            HTTPCommunicator.add_neighbor(new_edge, self.client.ip, self.client.port)
        except Exception as e:
            fed_logger.warning("[Mobility] add_neighbor(HTTP) %s failed: %s", new_edge, e)

        # 3) اطمینان از حضور در discovered_edges
        try:
            if new_edge not in getattr(self.client, "discovered_edges", set()):
                self.client.discovered_edges.add(new_edge)
        except Exception:
            pass

        fed_logger.info("[Mobility] CONNECTED client=%s:%s -> edge=%s",
                        self.client.ip, self.client.port, new_edge)

    # ----------------------------
    # Monitor & migrate (background thread)
    # ----------------------------
    def monitor_and_migrate(self):
        """
        Start a daemon thread that monitors distances and triggers migration.
        Switching rule:
          - If closest_edge != current_edge and (d_cur - d_closest) > SWITCH_MARGIN -> switch
          - Failsafe: if d_cur > 2 * THRESHOLD_DISTANCE and closest_edge != current_edge -> switch
        Also logs status every ~5s.
        """

        def monitor():
            fed_logger.info("[Mobility] monitor loop started (THRESHOLD=%sm)", self.THRESHOLD_DISTANCE)
            SWITCH_MARGIN = 100.0  # meters
            MAX_DISTANCE = self.THRESHOLD_DISTANCE * 2  # failsafe
            last_log = 0.0

            while True:
                time.sleep(1.0)

                try:
                    current_edge = self.get_current_edge()
                    closest_edge = self.find_closest_edge()
                except Exception as e:
                    fed_logger.debug("[Mobility] monitor: edge detection failed: %s", e)
                    continue

                if not current_edge:
                    # هنوز به اجی وصل نیست؛ تلاش کن وصل شوی
                    if closest_edge:
                        fed_logger.info("[Mobility] monitor: no current edge; connecting to %s", closest_edge)
                        try:
                            self.migrate_to_edge(closest_edge)
                        except Exception as e:
                            fed_logger.warning("[Mobility] monitor: initial connect failed: %s", e)
                    continue

                # مختصات کلاینت
                try:
                    node_lat = self.client.node_coordinate.latitude
                    node_lon = self.client.node_coordinate.longitude
                    node_coords = (node_lat, node_lon)
                except Exception:
                    # اگر هنوز مختصات ست نشده
                    continue

                # فاصله تا اج فعلی
                try:
                    cur_info = HTTPCommunicator.get_node_coordinate(current_edge)
                    if not cur_info:
                        continue
                    d_cur = geodesic(node_coords, (cur_info["latitude"], cur_info["longitude"])).meters
                except Exception:
                    continue

                # فاصله تا نزدیک‌ترین اج
                d_closest = float("inf")
                try:
                    if closest_edge:
                        cl_info = HTTPCommunicator.get_node_coordinate(closest_edge)
                        if cl_info:
                            d_closest = geodesic(
                                node_coords, (cl_info["latitude"], cl_info["longitude"])
                            ).meters
                except Exception:
                    pass

                # لاگ دوره‌ای برای مشاهده حرکت
                now = time.time()
                if now - last_log > 5.0:
                    fed_logger.info(
                        "[Mobility] monitor: cur=%s d_cur=%.1fm | closest=%s d_closest=%.1fm | pos=(%.5f, %.5f)",
                        current_edge, d_cur, closest_edge, d_closest,
                        node_lat, node_lon
                    )
                    last_log = now

                # قوانین سوییچ
                should_switch_by_margin = (
                    closest_edge
                    and (closest_edge.ip, closest_edge.port) != (current_edge.ip, current_edge.port)
                    and (d_cur - d_closest) > SWITCH_MARGIN
                )

                should_switch_by_failsafe = (
                    closest_edge
                    and (closest_edge.ip, closest_edge.port) != (current_edge.ip, current_edge.port)
                    and d_cur > MAX_DISTANCE
                )

                if should_switch_by_margin or should_switch_by_failsafe:
                    fed_logger.info(
                        "[Mobility] migrating: %s (%.1fm) -> %s (%.1fm) [margin=%s, failsafe=%s]",
                        current_edge, d_cur, closest_edge, d_closest,
                        should_switch_by_margin, should_switch_by_failsafe
                    )
                    try:
                        self.migrate_to_edge(closest_edge)
                        fed_logger.info("[Mobility] MIGRATED. Now edge=%s", self.get_current_edge())
                    except Exception as e:
                        fed_logger.exception("[Mobility] migrate_to_edge failed: %s", e)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
