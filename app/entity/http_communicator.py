import time

import requests

from app.config.logger import fed_logger
from app.entity.node_type import NodeType
from app.entity.node_identifier import NodeIdentifier


class HTTPCommunicator:
    MAX_RETRIES = 5
    WAIT_DURATION_SECONDS = 5

    @staticmethod
    def _wait_for_neighbor_to_get_ready(node_identifier: NodeIdentifier):
        for i in range(HTTPCommunicator.MAX_RETRIES):
            try:
                requests.get(f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type")
                return
            except requests.exceptions.ConnectionError:
                pass
            fed_logger.info(
                f"Node {node_identifier} is not ready, waiting for {HTTPCommunicator.WAIT_DURATION_SECONDS} seconds")
            time.sleep(HTTPCommunicator.WAIT_DURATION_SECONDS)
        raise ConnectionError(f"Node {node_identifier} is not ready")

    @staticmethod
    def get_node_type(node_identifier: NodeIdentifier) -> NodeType:
        try:
            HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        except ConnectionError:
            return NodeType.UNKNOWN
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-node-type"
        response = requests.get(request_url)
        return NodeType.from_value(response.json()['node_type'])

    @staticmethod
    def get_rabbitmq_url(node_identifier: NodeIdentifier) -> str:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-rabbitmq-url"
        response = requests.get(request_url)
        return response.json()['rabbitmq_url']

    @staticmethod
    def get_node_coordinate(node_identifier: NodeIdentifier) -> dict:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-node-coordinate"
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to get coordinates for node {node_identifier}, status code: {response.status_code}")

    @staticmethod
    def get_neighbors_from_neighbor(neighbor: NodeIdentifier, node_types: list[NodeType] = None) -> list[
        NodeIdentifier]:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(neighbor)
        request_url = f"http://{neighbor.ip}:{neighbor.port}/get-neighbors-info"
        response = requests.get(request_url)
        if response.status_code == 200:
            result = [NodeIdentifier(info['ip'], info['port']) for info in response.json()]
            return [neighbor for neighbor in result if
                    node_types is None or HTTPCommunicator.get_node_type(neighbor) in node_types]
        else:
            raise Exception(
                f"Failed to fetch neighbors from {neighbor}, status code: {response.status_code}")

    @staticmethod
    def add_neighbor(node_identifier: NodeIdentifier, neighbor_ip: str, neighbor_port: int):
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/add-neighbor"
        payload = {"ip": neighbor_ip, "port": neighbor_port}
        response = requests.post(request_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to add neighbor to node {node_identifier}, status code: {response.status_code}, message: {response.text}")

    @staticmethod
    def remove_neighbor(node_identifier: NodeIdentifier, neighbor_ip: str, neighbor_port: int):
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/remove-neighbor"
        payload = {"ip": neighbor_ip, "port": neighbor_port}
        response = requests.post(request_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to remove neighbor from node {node_identifier}, status code: {response.status_code}, message: {response.text}")

    @staticmethod
    def get_cluster(node_identifier: NodeIdentifier) -> dict:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-cluster"
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to get cluster information for node {node_identifier}, status code: {response.status_code}")

    @staticmethod
    def set_leader(node_identifier: NodeIdentifier, leader_ip: str, leader_port: int, is_leader: bool) -> dict:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/set-leader"
        payload = {"ip": leader_ip, "port": leader_port, "is_leader": is_leader}  # Added is_leader to payload
        response = requests.post(request_url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to set leader for node {node_identifier}, status code: {response.status_code}, message: {response.text}")

    @staticmethod
    def get_is_leader(node_identifier: NodeIdentifier) -> bool:
        HTTPCommunicator._wait_for_neighbor_to_get_ready(node_identifier)
        request_url = f"http://{node_identifier.ip}:{node_identifier.port}/get-is-leader"
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()['is_leader']
        else:
            raise Exception(
                f"Failed to get is_leader status from node {node_identifier}, status code: {response.status_code}")