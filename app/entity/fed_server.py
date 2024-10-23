import random
import threading
import time
from collections import defaultdict

import numpy as np
from colorama import Fore
from torch import nn, optim

from app.config import config
from app.config.logger import fed_logger
from app.dto.bandwidth import BandWidth
from app.dto.base_model import BaseModel
from app.dto.message import IterationFlagMessage, GlobalWeightMessage, NetworkTestMessage, JsonMessage, \
    SplitLayerConfigMessage, BaseMessage
from app.entity.aggregators.base_aggregator import BaseAggregator
from app.entity.fed_base_node_interface import FedBaseNodeInterface
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_identifier import NodeIdentifier
from app.entity.node_type import NodeType
from app.fl_method import fl_method_parser
from app.model.utils import get_available_torch_device
from app.util import model_utils, data_utils


# noinspection PyTypeChecker
class FedServer(FedBaseNodeInterface):
    def __init__(self, ip: str, port: int, model_name, dataset, aggregator: BaseAggregator,
                 neighbors: list[NodeIdentifier]):
        super().__init__(ip, port, NodeType.SERVER, None, neighbors)

        self.optimizers = {}
        self.nets = {}
        self.device = get_available_torch_device()
        self.model_name = model_name
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.neighbors_bandwidth: dict[str, BandWidth] = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.offloading = None
        self.split_layers: dict[NodeIdentifier, list] = {}
        self.aggregator = aggregator

        self.uninet = model_utils.get_model('Unit', None, self.device, True)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, 0)
        self.criterion = nn.CrossEntropyLoss()

    def initialize(self, LR):
        self.nets = {}
        self.optimizers = {}
        for edge in self.get_neighbors([NodeType.EDGE]):
            for client in HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT]):
                if client not in self.split_layers:
                    self.split_layers[client] = [len(self.uninet.cfg) - 4, len(self.uninet.cfg) - 2]
                split_point = self.split_layers[client][1]
                if split_point < len(self.uninet.cfg) - 1:
                    self.nets[client] = model_utils.get_model('Server', self.split_layers[client], self.device, True)
                    eweights = model_utils.get_model('Edge', self.split_layers[client], self.device, True).state_dict()
                    cweights = model_utils.get_model('Client', self.split_layers[client], self.device,
                                                     True).state_dict()

                    pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                                self.nets[client].state_dict(), eweights)
                    self.nets[client].load_state_dict(pweights)

                    if len(list(self.nets[client].parameters())) != 0:
                        self.optimizers[client] = optim.SGD(self.nets[client].parameters(), lr=LR, momentum=0.9)
                else:
                    self.nets[client] = model_utils.get_model('Server', self.split_layers[client], self.device, True)
        self.criterion = nn.CrossEntropyLoss()

    def scatter_split_layers(self, neighbors_types: list[NodeType] = None):
        for edge in self.get_neighbors([NodeType.EDGE]):
            clients = HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT])
            msg = SplitLayerConfigMessage(dict((k, self.split_layers[k]) for k in self.split_layers if k in clients))
            self.send_msg(self.get_exchange_name(), HTTPCommunicator.get_rabbitmq_url(edge), msg)

    def start_edge_training(self):
        self.threads = {}
        for edge in self.get_neighbors([NodeType.EDGE]):
            for client in HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT]):
                self.threads[client] = threading.Thread(target=self._thread_edge_training,
                                                        args=(edge, client))
                self.threads[client].start()

        for thread in self.threads:
            self.threads[thread].join()

    def _thread_edge_training(self, edge: NodeIdentifier, client: NodeIdentifier):
        edge_exchange_name = edge.get_exchange_name(client)
        flag: bool = self.recv_msg(edge_exchange_name, config.current_node_mq_url,
                                   IterationFlagMessage.MESSAGE_TYPE).flag
        fed_logger.info(Fore.RED + f"{flag}" + Fore.RESET)
        if not flag:
            fed_logger.info(str(client) + ' offloading training end')
            return 'Finish'
        while flag:
            flag: bool = self.recv_msg(edge_exchange_name, config.current_node_mq_url,
                                       IterationFlagMessage.MESSAGE_TYPE).flag
            fed_logger.info(Fore.RED + f"{flag}" + Fore.RESET)
            if not flag:
                break
            msg: GlobalWeightMessage = self.recv_msg(edge_exchange_name, config.current_node_mq_url,
                                                     GlobalWeightMessage.MESSAGE_TYPE)
            smashed_layers = msg.weights[0]
            labels = msg.weights[1]
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if client in self.optimizers:
                self.optimizers[client].zero_grad()
            outputs = self.nets[client](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if client in self.optimizers:
                self.optimizers[client].step()
            msg = GlobalWeightMessage([inputs.grad])
            self.send_msg(edge_exchange_name, HTTPCommunicator.get_rabbitmq_url(edge), msg)

        fed_logger.info(str(client) + ' offloading training end')
        return 'Finish'

    def aggregate(self, clients_local_weights):
        w_local_list = []
        for client, weight in clients_local_weights.items():
            split_point = self.split_layers[client][0]
            if split_point != (config.model_len - 1):
                w_local = (
                    model_utils.concat_weights(self.uninet.state_dict(), clients_local_weights[client],
                                               self.nets[client].state_dict()),
                    config.N / config.K)
                w_local_list.append(w_local)
            else:
                w_local = (clients_local_weights[client], config.N / config.K)
            w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, w_local_list)
        self.uninet.load_state_dict(aggregated_model)
        return aggregated_model

    def _concat_clients_local_weights(self, clients_local_weights) -> list:
        w_local_list = []
        for client, weight in clients_local_weights.items():
            split_point = self.split_layers[client][0]
            if split_point != (config.model_len - 1):
                w_local = (
                    model_utils.concat_weights(self.uninet.state_dict(), clients_local_weights[client],
                                               self.nets[client].state_dict()),
                    config.N / config.K)
                w_local_list.append(w_local)
            else:
                w_local = (clients_local_weights[client], config.N / config.K)
            w_local_list.append(w_local)
        return w_local_list

    def gather_clients_local_weights(self):
        clients_local_weights = {}
        for edge in self.get_neighbors([NodeType.EDGE]):
            for client in HTTPCommunicator.get_neighbors_from_neighbor(edge, [NodeType.CLIENT]):
                edge_exchange = edge.get_exchange_name(client)
                msg: GlobalWeightMessage = self.recv_msg(edge_exchange, config.current_node_mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                clients_local_weights[client] = msg.weights[0]
        return clients_local_weights

    def split(self, state, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels, self)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def get_neighbors_bandwidth(self) -> dict[str, BandWidth]:
        return self.neighbors_bandwidth

    def d2d_aggregate(self, client_local_weights: dict[str, BaseModel]) -> None:
        w_local_list = []
        client_neighbors = self.get_neighbors([NodeType.CLIENT])
        for neighbor in client_neighbors:
            if HTTPCommunicator.get_is_leader(neighbor):
                HTTPCommunicator.set_leader(neighbor, neighbor.ip, neighbor.port, False)
                w_local = (client_local_weights[str(neighbor)], config.N / len(client_neighbors))
                w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = self.aggregator.aggregate(zero_model, w_local_list)
        self.uninet.load_state_dict(aggregated_model)

    def receive_leaders_local_weights(self):
        client_local_weights = {}
        for neighbor in self.get_neighbors([NodeType.CLIENT]):
            is_leader = HTTPCommunicator.get_is_leader(neighbor)
            if is_leader:
                msg: GlobalWeightMessage = self.recv_msg(neighbor.get_exchange_name(), config.current_node_mq_url,
                                                         GlobalWeightMessage.MESSAGE_TYPE)
                client_local_weights[str(neighbor)] = msg.weights[0]
        return client_local_weights

    def choose_random_leader_per_cluster(self):
        clusters = defaultdict(list)
        neighbors = self.get_neighbors([NodeType.CLIENT])

        for neighbor in neighbors:
            cluster_info = HTTPCommunicator.get_cluster(neighbor)
            if cluster_info:
                cluster_id = cluster_info.get('cluster')
                if cluster_id:
                    clusters[cluster_id].append(neighbor)

        random_clients = {}
        for cluster, clients in clusters.items():
            if clients:
                random_client = random.choice(clients)
                random_clients[cluster] = random_client
                HTTPCommunicator.set_leader(random_client, random_client.ip, random_client.port, True)

        return random_clients