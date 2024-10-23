import http
import threading

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from uvicorn import Server

from app.config import config
from app.config.logger import fed_logger
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_coordinate import NodeCoordinate
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType


class Node:
    _app: FastAPI
    neighbors: list[NodeIdentifier]
    _node_type: NodeType
    node_coordinate: NodeCoordinate
    discovered_edges: set[NodeIdentifier]
    cluster: str
    is_leader: False

    def __init__(self, ip: str, port: int, node_type: NodeType, cluster, neighbors: list[NodeIdentifier] = None):
        self._server_started = False
        self.ip = ip
        self.port = port
        self._node_type = node_type
        self.neighbors = []
        # self.node_coordinate = config.INITIAL_NODE_COORDINATE
        self.discovered_edges = set()
        self._app = FastAPI()
        self._server: Server
        if neighbors:
            self.add_neighbors(neighbors)
        self._setup_routes()
        self._start_server_in_thread(port)
        self.cluster = cluster
        self.is_leader = False

    def __str__(self):
        return f'{self.ip}:{self.port}'

    def _setup_routes(self):
        self._app.add_route("/get-node-type", self.get_node_type, methods=["GET"])
        self._app.add_route("/get-rabbitmq-url", self.get_rabbitmq_url, methods=["GET"])
        self._app.add_route("/get-node-coordinate", self.get_node_coordinate, methods=["GET"])
        self._app.add_route("/get-neighbors-info", self.get_neighbors_info, methods=["GET"])
        self._app.add_route("/add-neighbor", self.add_neighbor_api, methods=["POST"])
        self._app.add_route("/remove-neighbor", self.remove_neighbor_api, methods=["POST"])
        self._app.add_route("/get-cluster", self.get_cluster, methods=["GET"])
        self._app.add_route("/set-leader", self.set_leader_api, methods=["POST"])
        self._app.add_route("/get-is-leader", self.get_is_leader_api, methods=["GET"])

    async def get_node_type(self, _: Request):
        return JSONResponse({'node_type': self._node_type.name}, http.HTTPStatus.OK)

    @staticmethod
    async def get_rabbitmq_url(_: Request):
        return JSONResponse({'rabbitmq_url': config.current_node_mq_url}, http.HTTPStatus.OK)

    async def get_neighbors_info(self, _: Request):
        neighbors_info = [{'ip': neighbor.ip, 'port': neighbor.port} for neighbor in self.neighbors]
        return JSONResponse(neighbors_info, http.HTTPStatus.OK)

    async def get_node_coordinate(self, _: Request):
        if self.node_coordinate is None:
            return JSONResponse({'error': 'Node coordinate not set'}, http.HTTPStatus.NOT_FOUND)
        return JSONResponse({
            'latitude': self.node_coordinate.latitude,
            'longitude': self.node_coordinate.longitude,
            'altitude': self.node_coordinate.altitude,
            'seconds_since_start': self.node_coordinate.seconds_since_start
        }, http.HTTPStatus.OK)

    async def add_neighbor_api(self, request: Request):
        """
        API endpoint to add a neighbor. Expects a JSON body with 'ip' and 'port'.
        """
        data = await request.json()
        if 'ip' not in data or 'port' not in data:
            raise HTTPException(status_code=400, detail="Invalid data format. 'ip' and 'port' are required.")

        new_neighbor = NodeIdentifier(ip=data['ip'], port=data['port'])
        self.add_neighbor(new_neighbor)
        return JSONResponse({'message': 'Neighbor added successfully.'}, http.HTTPStatus.OK)

    async def remove_neighbor_api(self, request: Request):
        data = await request.json()
        if 'ip' not in data or 'port' not in data:
            raise HTTPException(status_code=400, detail="Invalid data format. 'ip' and 'port' are required.")

        neighbor_to_remove = NodeIdentifier(ip=data['ip'], port=data['port'])
        self.remove_neighbor(neighbor_to_remove)
        return JSONResponse({'message': 'Neighbor removed successfully.'}, http.HTTPStatus.OK)

    def remove_neighbor(self, node_id: NodeIdentifier):
        if node_id in self.neighbors:
            self.neighbors.remove(node_id)

    def update_coordinates(self, new_latitude, new_longitude, new_altitude, new_seconds_since_start):
        self.node_coordinate = NodeCoordinate(
            latitude=new_latitude,
            longitude=new_longitude,
            altitude=new_altitude,
            seconds_since_start=new_seconds_since_start
        )

    def add_neighbor(self, node_id: NodeIdentifier):
        if node_id not in self.neighbors:
            self.neighbors.append(node_id)

    def add_neighbors(self, node_ids: list[NodeIdentifier]):
        for node_id in node_ids:
            self.add_neighbor(node_id)

    def get_neighbors(self, node_types: list[NodeType] = None) -> list[NodeIdentifier]:
        if not node_types:
            return self.neighbors
        return [node for node in self.neighbors if HTTPCommunicator.get_node_type(node) in node_types]

    @staticmethod
    def fetch_neighbors_from_neighbor(neighbor: NodeIdentifier):
        response = HTTPCommunicator.get_neighbors_from_neighbor(neighbor)
        return response

    def get_exchange_name(self, extra_target: str = '') -> str:
        if extra_target:
            return f"{self.ip}:{self.port}_{extra_target}"
        return f"{self.ip}:{self.port}"

    def stop_server(self):
        self._server.should_exit = True

    def _run_server(self, port: int):
        self._server = uvicorn.Server(uvicorn.Config(self._app, host="0.0.0.0", port=port, log_level="warning"))
        self._server.run()

    def _start_server_in_thread(self, port: int):
        if self._server_started:
            return
        self._server_started = True
        server_thread = threading.Thread(target=self._run_server, args=(port,))
        server_thread.start()

    async def get_cluster(self, _: Request):
        return JSONResponse({'cluster': self.cluster}, http.HTTPStatus.OK)

    async def set_leader_api(self, request: Request):
        data = await request.json()
        if 'is_leader' not in data:
            raise HTTPException(status_code=400, detail="Missing 'is_leader' field.")
        self.is_leader = data['is_leader']
        return JSONResponse({'message': f"Node is_leader set to {self.is_leader}"}, http.HTTPStatus.OK)

    async def get_is_leader_api(self, _: Request):
        return JSONResponse({'is_leader': self.is_leader}, http.HTTPStatus.OK)