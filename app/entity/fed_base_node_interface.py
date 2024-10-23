import threading
import time
from abc import ABC

from app.config import config
from app.config.logger import fed_logger
from app.dto.bandwidth import BandWidth
from app.dto.message import BaseMessage, GlobalWeightMessage, SplitLayerConfigMessage, MessageType, NetworkTestMessage
from app.dto.received_message import ReceivedMessage
from app.entity.communicator import Communicator
from app.entity.node import Node
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.entity.http_communicator import HTTPCommunicator
from app.util import data_utils


# noinspection PyTypeChecker
class FedBaseNodeInterface(ABC, Node, Communicator):
    def __init__(self, ip: str, port: int, node_type: NodeType, cluster, neighbors: list[NodeIdentifier] = None):
        Node.__init__(self, ip, port, node_type, cluster, neighbors)
        Communicator.__init__(self)
        self.neighbor_bandwidth: dict[NodeIdentifier, BandWidth] = {}
        self.uninet = None
        self.split_layers = None
        self.device = None

    def scatter_msg(self, msg: BaseMessage, neighbors_types: list[NodeType] = None):
        for neighbor in self.get_neighbors(neighbors_types):
            rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
            self.send_msg(self.get_exchange_name(), rabbitmq_url, msg)

    def scatter_global_weights(self, neighbors_types: list[NodeType] = None):
        msg = GlobalWeightMessage([self.uninet.state_dict()])
        self.scatter_msg(msg, neighbors_types)

    def scatter_split_layers(self, neighbors_types: list[NodeType] = None):
        for neighbor in self.get_neighbors(neighbors_types):
            msg = SplitLayerConfigMessage(self.split_layers[neighbor])
            self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(neighbor), msg)

    def gather_msgs(self, msg_type: MessageType, neighbors_types: list[NodeType] = None) -> list[
        ReceivedMessage]:  # (ip, msg)
        messages = []
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_types is None or neighbor_type in neighbors_types:
                msg = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url, msg_type)
                messages.append(ReceivedMessage(msg, neighbor))
        return messages

    def gather_neighbors_network_bandwidth(self, neighbors_type: NodeType = None):
        net_threads = {}
        for neighbor in self.get_neighbors():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            if neighbors_type is None or neighbor_type == neighbors_type:
                net_threads[neighbor] = threading.Thread(target=self._thread_network_testing,
                                                              args=(neighbor,), name=str(neighbor))
                net_threads[neighbor].start()

        for _, thread in net_threads.items():
            thread.join()

    def _thread_network_testing(self, neighbor: NodeIdentifier):
        network_time_start = time.time()
        msg = NetworkTestMessage([self.uninet.to(self.device).state_dict()])
        neighbor_rabbitmq_url = HTTPCommunicator.get_rabbitmq_url(neighbor)
        self.send_msg(self.get_exchange_name(), neighbor_rabbitmq_url, msg)
        msg: NetworkTestMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                NetworkTestMessage.MESSAGE_TYPE)
        network_time_end = time.time()
        self.neighbor_bandwidth[neighbor] = BandWidth(data_utils.sizeofmessage(msg.weights),
                                                      network_time_end - network_time_start)
