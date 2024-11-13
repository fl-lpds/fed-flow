from abc import ABC, abstractmethod

import torch
from torch import multiprocessing

from app.config import config
from app.entity.communicator import Communicator
from app.util import data_utils, model_utils


class FedEdgeServerInterface(ABC, Communicator):
    def __init__(self, model_name, dataset, offload, simnet: bool = None):
        super(FedEdgeServerInterface, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.nets = {}
        self.simnet = simnet
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        self.state = None
        self.client_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.central_server_communicator = Communicator()
        self.transmissionTimes = 0
        self.start_time_of_computation_each_client = {}
        self.computation_time_of_each_client = {}

        # if offload:
        model_len = model_utils.get_unit_model_len()
        self.uninet = model_utils.get_model('Unit', [model_len - 1, model_len - 1], self.device, True)

        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())

    @abstractmethod
    def test_client_network(self, client_ips):
        """
        send message to test_app network speed
        """
        pass

    @abstractmethod
    def test_server_network(self):
        pass

    @abstractmethod
    def client_network(self):
        """
        send client network speed to central server
        """
        pass

    @abstractmethod
    def get_split_layers_config(self, client_ips):
        """
        receive send splitting data to clients
        """
        pass

    @abstractmethod
    def local_weights(self, client_ip):
        """
        receive and send final weights for aggregation
        """
        pass

    @abstractmethod
    def global_weights(self, client_ips: []):
        """
        receive global weights
        """
        pass

    @abstractmethod
    def energy(self, client_ip):
        """
        Returns: client energy consumption of the given client_ip
        """
        pass

    @abstractmethod
    def initialize(self, split_layers, LR, client_ips, simnetbw: float = None):
        pass

    def scatter(self, msg, is_weight=False):
        for i in config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]:
            self.send_msg(i, msg, is_weight, url=config.EDGE_SERVER_CONFIG[config.index])

    @abstractmethod
    def forward_propagation(self, client_ip):
        pass

    @abstractmethod
    def backward_propagation(self, outputs, client_ip, inputs):
        pass

    @abstractmethod
    def thread_offload_training(self, client_ip):
        pass

    @abstractmethod
    def thread_no_offload_training(self, client_ip):
        pass

    @abstractmethod
    def no_offload_global_weights(self):
        pass

    @abstractmethod
    def client_attendance(self, client_ips):
        pass

    @abstractmethod
    def get_simnet_client_network(self):
        pass

    @abstractmethod
    def send_simnet_bw_to_server(self, simnetbw):
        pass

    @abstractmethod
    def get_power_and_send_to_server(self):
        pass