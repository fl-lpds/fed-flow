import multiprocessing
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from app.config import config
from app.config.logger import fed_logger
from app.entity.communicator import Communicator
from app.fl_method import fl_method_parser
from app.util import data_utils, model_utils


class FedServerInterface(ABC, Communicator):
    def __init__(self, model_name, dataset, offload, edge_based, simnet: bool = None):
        super(FedServerInterface, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.offload = offload
        self.edge_based = edge_based
        self.model_name = model_name
        self.group_labels = None
        self.criterion = None
        self.split_layers = None
        # self.state = None
        self.client_bandwidth = {}
        self.client_remaining_energy = {}
        self.client_energy = {}
        self.client_comp_energy = {}
        self.client_comp_time = {}
        for client, index in config.CLIENTS_CONFIG.items():
            self.client_comp_energy[client] = {}
            self.client_comp_time[client] = {}
        self.client_comm_energy = {}
        self.power_usage_of_client = {}
        self.client_utilization = {}
        self.edge_bandwidth = {}
        self.dataset = dataset
        self.threads = None
        self.net_threads = None
        self.offloading = None
        self.tt_start = {}
        self.tt_end = {}

        self.simnet = simnet
        self.client_training_transmissionTime = {}

        self.start_time_of_communication_each_client = {}
        self.real_communication_time_of_each_client = {}
        self.start_time_of_computation_each_client = {}
        self.computation_time_of_each_client = {}
        self.process_wall_time = {}

        self.computation_time_of_each_client_on_edges = {}
        self.total_computation_time_of_each_edge = {}
        self.total_computation_time = 0
        self.edge_flops = {}
        self.server_flops = {}
        self.aggregation_time = 0

        self.ttpiOfClients = {}  # Training time per iteration

        self.approximated_tt_of_actions = []
        self.approximated_energy_of_actions = []
        self.actions = []
        self.nice_value = {}

        self.uninet = model_utils.get_model('Unit', None, self.device, self.edge_based)
        self.testset = data_utils.get_testset()
        self.testloader = data_utils.get_testloader(self.testset, multiprocessing.cpu_count())
        self.criterion = nn.CrossEntropyLoss()

        self.total_model_size = 0
        self.activation_size = {}
        self.gradient_size = {}
        self.model_flops_per_layer = {}

        self.best_tt_splitting_found = {'splitting': [], 'time': 0}

    @abstractmethod
    def edge_offloading_train(self, client_ips, hasPriority=False):
        pass

    @abstractmethod
    def no_edge_offloading_train(self, client_ips):
        pass

    @abstractmethod
    def no_offloading_train(self, client_ips):
        pass

    @abstractmethod
    def test_network(self, edge_ips):
        """
        send message to test_app network speed
        """
        pass

    @abstractmethod
    def client_network(self, edge_ips):
        """
        receive client network speed
        """
        pass

    @abstractmethod
    def send_split_layers_config(self):
        pass

    @abstractmethod
    def split_layer(self):
        """
        send splitting data
        """
        pass

    @abstractmethod
    def e_local_weights(self, client_ips):
        """
        receive final weights for aggregation in offloading mode
        """
        pass

    @abstractmethod
    def c_local_weights(self, client_ips):
        """
        receive client local weights in no offloading mode
        """
        pass

    @abstractmethod
    def edge_offloading_global_weights(self):
        """
        send global weights
        """
        pass

    @abstractmethod
    def no_offloading_global_weights(self):
        pass

    @abstractmethod
    def initialize(self, split_layers, LR):
        pass

    @abstractmethod
    def aggregate(self, client_ips, aggregate_method, eweights):
        pass

    def call_aggregation(self, options: dict, eweights):
        method = fl_method_parser.fl_methods.get(options.get('aggregation'))
        if method is None:
            fed_logger.error("aggregate method is none")
        self.aggregate(config.CLIENTS_LIST, method, eweights)

    @abstractmethod
    def cluster(self, options: dict):
        pass

    @abstractmethod
    def split(self, state, options: dict):
        pass

    def scatter(self, msg, is_weight=False):
        list1 = config.CLIENTS_LIST
        if self.edge_based:
            list1 = config.EDGE_SERVER_LIST
            for i in list1:
                self.send_msg(exchange=i, msg=msg, is_weight=is_weight, url=i)
        else:
            for i in list1:
                self.send_msg(exchange=i, msg=msg, is_weight=is_weight)

    def concat_norm(self, ttpi, offloading):
        ttpi_order = []
        offloading_order = []
        for c in config.CLIENTS_LIST:
            ttpi_order.append(ttpi[c])
            offloading_order.append(offloading[c])

        group_max_index = [0 for i in range(config.G)]
        group_max_value = [0 for i in range(config.G)]
        for c in config.CLIENTS_LIST:
            label = self.group_labels[config.CLIENTS_CONFIG[c]]
            if ttpi_order[config.CLIENTS_CONFIG[c]] >= group_max_value[label]:
                group_max_value[label] = ttpi_order[config.CLIENTS_CONFIG[c]]
                group_max_index[label] = config.CLIENTS_CONFIG[c]

        ttpi_order = np.array(ttpi_order)[np.array(group_max_index)]
        offloading_order = np.array(offloading_order)[np.array(group_max_index)]
        state = np.append(ttpi_order, offloading_order)
        return state

    def get_offloading(self, split_layer):
        offloading = {}
        workload = 0
        # assert len(split_layer) == len(config.CLIENTS_LIST)
        for c in config.CLIENTS_LIST:
            for l in range(model_utils.get_unit_model_len()):
                split_point = split_layer[config.CLIENTS_CONFIG[c]]
                if self.edge_based:
                    split_point = split_layer[config.CLIENTS_CONFIG[c]][0]
                if l <= split_point:
                    workload += model_utils.get_class()().cfg[l][5]
            offloading[c] = workload / config.total_flops
            workload = 0

        return offloading

    def ttpi(self, client_ips, clients_TT):
        for client in client_ips:
            self.ttpiOfClients[client] = clients_TT[client] / ((config.N / config.K) / config.B)
        return self.ttpiOfClients

    def bandwith(self):
        return self.edge_bandwidth

    @abstractmethod
    def energy_tt(self, client_ips):
        pass

    @abstractmethod
    def e_energy_tt(self, client_ips):
        pass

    @abstractmethod
    def edge_based_state(self):
        pass

    @abstractmethod
    def client_attendance(self, client_ips):
        pass

    @abstractmethod
    def e_client_attendance(self, client_ips):
        pass

    @abstractmethod
    def get_simnet_edge_network(self):
        pass

    @abstractmethod
    def remove_non_pickleable(self):
        pass

    @abstractmethod
    def calculate_each_layer_activation_gradiant_size(self):
        pass

    @abstractmethod
    def calculate_each_layer_FLOP(self):
        pass

    @abstractmethod
    def get_power_of_client(self):
        pass

    @abstractmethod
    def getFlopsOnEdgeAndServer(self):
        pass

    def simnetTrainingTimeCalculation(self, aggregation_time, server_sequential_transmission_time, energy_tt_list,
                                      edgeBased=True):
        pass
