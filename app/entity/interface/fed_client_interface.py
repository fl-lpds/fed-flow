from abc import ABC, abstractmethod

import torch
from torch import nn, optim

from app.entity.communicator import Communicator
from app.util import model_utils


class FedClientInterface(ABC, Communicator):
    def __init__(self, server, datalen, model_name, dataset, train_loader, LR, edge_based, simnet):
        super(FedClientInterface, self).__init__()
        self.datalen = datalen
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.edge_based = edge_based
        self.server_id = server
        self.dataset = dataset
        self.train_loader = train_loader
        self.split_layers = None
        self.net = {}
        self.uninet = model_utils.get_model('Unit', None, self.device, edge_based)
        # self.uninet = model_utils.get_model('Unit', config.split_layer[config.index], self.device, edge_based)
        self.net = self.uninet
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)

        self.simnet: bool = simnet
        self.simnetbw = 10_000_000 if self.simnet else 0  # 10 Mbps

        self.computational_time = 0

    @abstractmethod
    def initialize(self, split_layer, LR, simnetbw: float = None):
        pass

    @abstractmethod
    def send_local_weights_to_edge(self):
        """
        send final weights for aggregation
        """
        pass

    @abstractmethod
    def send_local_weights_to_server(self):
        pass

    @abstractmethod
    def test_network(self):
        """
        send message to test_app network speed
        """
        pass

    @abstractmethod
    def get_server_global_weights(self):
        pass

    def no_offloading_train(self):
        pass

    def edge_test_network(self):
        pass

    @abstractmethod
    def get_split_layers_config_from_edge(self):
        pass

    @abstractmethod
    def get_split_layers_config(self):
        """
        receive splitting data
        """
        pass

    @abstractmethod
    def get_edge_global_weights(self):
        """
        receive global weights
        """
        pass

    @abstractmethod
    def get_server_global_weights(self):
        pass

    @abstractmethod
    def edge_offloading_train(self):
        pass

    @abstractmethod
    def no_offloading_train(self):
        pass

    @abstractmethod
    def offloading_train(self):
        pass

    @abstractmethod
    def energy_tt(self, remaining_energy, comp_energy, comm_energy, tt, utilization):
        pass

    @abstractmethod
    def next_round_attendance(self, remaining_energy):
        pass

    @abstractmethod
    def e_next_round_attendance(self, remaining_energy):
        pass

    @abstractmethod
    def send_simnet_bw_to_edge(self, simnetbw):
        pass

    @abstractmethod
    def send_power_to_edge(self):
        pass
