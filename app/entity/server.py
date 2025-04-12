import sys
import threading
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore
from torch.multiprocessing import Process, Manager

from app.entity.interface.fed_server_interface import FedServerInterface
from app.fl_method import fl_method_parser

sys.path.append('../../')

import config
from app.util import message_utils, model_utils, data_utils
from app.config.logger import fed_logger
from app.util.energy_estimation import *

np.random.seed(0)
torch.manual_seed(0)
lock = threading.Lock()


class FedServer(FedServerInterface):

    def initialize(self, split_layers, LR):
        self.split_layers = split_layers
        self.nets = {}
        self.optimizers = {}

        for i in range(len(split_layers)):
            client_ip = config.CLIENTS_INDEX[i]
            if client_ip in config.CLIENTS_LIST:
                client_index = config.CLIENTS_CONFIG[client_ip]
                split_point = split_layers[client_index]
                if self.edge_based:
                    split_point = split_layers[client_index][1]
                if split_point < len(
                        self.uninet.cfg) - 1:  # Only offloading client need initialize optimizer in server
                    if self.edge_based:
                        self.nets[client_ip] = model_utils.get_model('Server', split_layers[client_index], self.device,
                                                                     self.edge_based)

                        # offloading weight in server also need to be initialized from the same global weight
                        eweights = model_utils.get_model('Edge', split_layers[client_index], self.device,
                                                         self.edge_based).state_dict()
                        cweights = model_utils.get_model('Client', split_layers[client_index], self.device,
                                                         self.edge_based).state_dict()

                        pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                                    self.nets[client_ip].state_dict(), eweights)
                        self.nets[client_ip].load_state_dict(pweights)

                        if len(list(self.nets[client_ip].parameters())) != 0:
                            self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                                   momentum=0.9)
                    else:
                        self.nets[client_ip] = model_utils.get_model('Server', split_layers[client_index], self.device,
                                                                     self.edge_based)

                        # offloading weight in server also need to be initialized from the same global weight
                        cweights = model_utils.get_model('Client', split_layers[client_index], self.device,
                                                         self.edge_based).state_dict()
                        pweights = model_utils.split_weights_server(self.uninet.state_dict(), cweights,
                                                                    self.nets[client_ip].state_dict(), [])
                        self.nets[client_ip].load_state_dict(pweights)

                        # if len(list(self.nets[client_ip].parameters())) != 0:
                        self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                               momentum=0.9)
                else:
                    self.nets[client_ip] = model_utils.get_model('Server', split_layers[client_index], self.device,
                                                                 self.edge_based)
        self.criterion = nn.CrossEntropyLoss()

    def edge_offloading_train(self, client_ips, hasPriority=False):
        for clientips in config.CLIENTS_LIST:
            self.computation_time_of_each_client[clientips] = 0
            self.client_training_transmissionTime[clientips] = 0
            self.process_wall_time[clientips] = 0

        with Manager() as manager:
            shared_data = manager.dict()
            shared_data['computation_time_of_each_client'] = self.computation_time_of_each_client
            shared_data['client_training_transmissionTime'] = self.client_training_transmissionTime
            shared_data['current_round'] = config.current_round
            shared_data['process_wall_time'] = self.process_wall_time

            processes = {}
            for i in range(len(client_ips)):
                processes[client_ips[i]] = Process(target=self._thread_edge_training,
                                                   args=(client_ips[i], shared_data,),
                                                   name=client_ips[i])
            for i in range(len(client_ips)):
                fed_logger.info(str(client_ips[i]) + ' offloading training start')
                processes[client_ips[i]].start()
                if hasPriority:
                    os.system(f"renice -n {self.server_nice_value[client_ips[i]]} -p {processes[client_ips[i]].pid}")

                self.tt_start[client_ips[i]] = time.time()

            for i in range(len(client_ips)):
                processes[client_ips[i]].join()
            self.computation_time_of_each_client = shared_data['computation_time_of_each_client']
            self.client_training_transmissionTime = shared_data['client_training_transmissionTime']
            self.process_wall_time = shared_data['process_wall_time']

    def no_offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading,
                                                           args=(client_ips[i],), name=client_ips[i])
            fed_logger.info(str(client_ips[i]) + ' no offloading training start')
            self.threads[client_ips[i]].start()
        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def no_edge_offloading_train(self, client_ips):
        for clientips in config.CLIENTS_LIST:
            self.computation_time_of_each_client[clientips] = 0
            self.client_training_transmissionTime[clientips] = 0

        with Manager() as manager:
            shared_data = manager.dict()
            shared_data['computation_time_of_each_client'] = self.computation_time_of_each_client
            shared_data['client_training_transmissionTime'] = self.client_training_transmissionTime
            shared_data['current_round'] = config.current_round

            processes = {}
            for i in range(len(client_ips)):
                processes[client_ips[i]] = Process(target=self._thread_training_offloading,
                                                   args=(client_ips[i], shared_data,),
                                                   name=client_ips[i])
                fed_logger.info(str(client_ips[i]) + ' offloading training start')
                processes[client_ips[i]].start()
                self.tt_start[client_ips[i]] = time.time()

            for i in range(len(client_ips)):
                processes[client_ips[i]].join()
            self.computation_time_of_each_client = shared_data['computation_time_of_each_client']
            self.client_training_transmissionTime = shared_data['client_training_transmissionTime']

    def _thread_training_no_offloading(self, client_ip):
        pass

    def _thread_training_offloading(self, client_ip, sharedData):
        config.current_round = sharedData['current_round']
        comp_time = 0
        i = 0
        msg = self.recv_msg(client_ip, f"{message_utils.local_iteration_flag_client_to_server()}_{i}_{client_ip}")
        flag = msg[1]

        i += 1
        while flag:
            msg = self.recv_msg(client_ip, f"{message_utils.local_iteration_flag_client_to_server()}_{i}_{client_ip}")
            flag = msg[1]

            if not flag:
                continue

            msg = self.recv_msg(client_ip,
                                f"{message_utils.local_activations_client_to_server()}_{i}_{client_ip}", True)

            smashed_layers = msg[1]
            labels = msg[2]

            comp_start_time = time.process_time()

            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]] < config.model_len - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
                outputs = self.nets[client_ip](inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()

            comp_time += time.process_time() - comp_start_time
            msg = [f"{message_utils.server_gradients_server_to_client() + str(client_ip)}_{i}", inputs.grad]

            self.send_msg(client_ip, msg, True)
            i += 1

        fed_logger.info(str(client_ip) + 'no edge offloading training end')
        localData = sharedData['computation_time_of_each_client']
        localData[client_ip] = comp_time
        sharedData['computation_time_of_each_client'] = localData

        localData = sharedData['client_training_transmissionTime']
        localData[client_ip] = 0
        sharedData['client_training_transmissionTime'] = localData
        return 'Finish'

    def _thread_edge_training(self, client_ip, sharedData):
        time.sleep(1)
        config.current_round = sharedData['current_round']
        communication_time = 0
        comp_time = 0
        wall_time_comp_time = 0

        fed_logger.info(Fore.MAGENTA + f"Attribute of each process=> {client_ip}: {self.edge_bandwidth}, "
                                       f"{self.split_layers}")
        edge_of_client = config.CLIENT_MAP[client_ip]
        i = 0
        msg = self.recv_msg(config.CLIENT_MAP[client_ip],
                            f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}',
                            url=config.CLIENT_MAP[client_ip])

        if self.simnet:
            communication_time += (data_utils.sizeofmessage(msg) / self.edge_bandwidth[edge_of_client])

        flag = msg[1]
        i += 1
        fed_logger.debug(Fore.RED + f"{flag}")
        if not flag:
            fed_logger.info(str(client_ip) + ' offloading training end')
            localData = sharedData['computation_time_of_each_client']
            localData[client_ip] = comp_time
            sharedData['computation_time_of_each_client'] = localData

            localData = sharedData['process_wall_time']
            localData[client_ip] = wall_time_comp_time
            sharedData['process_wall_time'] = localData

            localData = sharedData['client_training_transmissionTime']
            localData[client_ip] = communication_time
            sharedData['client_training_transmissionTime'] = localData

            return 'Finish'
        while flag:

            if self.split_layers[config.CLIENTS_CONFIG[client_ip]][1] < len(self.uninet.cfg) - 1:

                msg = self.recv_msg(config.CLIENT_MAP[client_ip],
                                    f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}',
                                    url=config.CLIENT_MAP[client_ip])

                if self.simnet:
                    communication_time += (data_utils.sizeofmessage(msg) / self.edge_bandwidth[edge_of_client])

                flag = msg[1]
                fed_logger.debug(Fore.RED + f"{flag}")
                if not flag:
                    break

                msg = self.recv_msg(config.CLIENT_MAP[client_ip],
                                    f'{message_utils.local_activations_edge_to_server() + "_" + client_ip}_{i}', True,
                                    url=config.CLIENT_MAP[client_ip])

                if self.simnet:
                    communication_time += (data_utils.sizeofmessage(msg) / self.edge_bandwidth[edge_of_client])

                smashed_layers = msg[1]
                labels = msg[2]

                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)

                s_time = time.process_time()
                s_wall_time = time.time()

                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
                outputs = self.nets[client_ip](inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()

                comp_time += time.process_time() - s_time
                wall_time_comp_time += time.time() - s_wall_time

                # Send gradients to edge
                msg = [f'{message_utils.server_gradients_server_to_edge() + str(client_ip)}_{i}', inputs.grad]

                self.send_msg(config.CLIENT_MAP[client_ip], msg, True, url=config.CLIENT_MAP[client_ip])
                if self.simnet:
                    communication_time += (data_utils.sizeofmessage(msg) / self.edge_bandwidth[edge_of_client])

            i += 1

        localData = sharedData['computation_time_of_each_client']
        localData[client_ip] = comp_time
        sharedData['computation_time_of_each_client'] = localData

        localData = sharedData['process_wall_time']
        localData[client_ip] = wall_time_comp_time
        sharedData['process_wall_time'] = localData

        localData = sharedData['client_training_transmissionTime']
        localData[client_ip] = communication_time
        sharedData['client_training_transmissionTime'] = localData

        fed_logger.info(str(client_ip) + ' offloading training end')
        return 'Finish'

    def aggregate(self, client_ips, aggregate_method, eweights: dict):
        w_local_list = []
        # fed_logger.info("aggregation start")
        aggregation_time_start = time.time()
        for client in eweights.keys():
            if self.offload:
                i = config.CLIENTS_CONFIG[client]
                sp = self.split_layers[i]
                if self.edge_based:
                    sp = self.split_layers[i][0]
                if sp != (config.model_len - 1):
                    w_local = (
                        model_utils.concat_weights(self.uninet.state_dict(), eweights[client],
                                                   self.nets[client].state_dict()),
                        config.N / config.K)
                    # w_local_list.append(w_local)
                else:
                    w_local = (eweights[client], config.N / config.K)
            else:
                w_local = (eweights[client], config.N / config.K)
            # test_model = model_utils.get_model('Unit', None, self.device, self.edge_based)
            # acc = model_utils.test(test_model, self.testloader, self.device, self.criterion)
            # fed_logger.info(Fore.MAGENTA + f"mini accuracy: {acc}")

            w_local_list.append(w_local)
        zero_model = model_utils.zero_init(self.uninet).state_dict()
        aggregated_model = aggregate_method(zero_model, w_local_list, config.N)
        self.uninet.load_state_dict(aggregated_model)
        self.aggregation_time = time.time() - aggregation_time_start
        return aggregated_model

    def test_network(self, connection_ips):
        """
        send message to test network speed
        """
        # Network test_app
        self.net_threads = {}
        for i in range(len(connection_ips)):
            self.net_threads[connection_ips[i]] = threading.Thread(target=self._thread_network_testing,
                                                                   args=(connection_ips[i],), name=connection_ips[i])
            self.net_threads[connection_ips[i]].start()

        for i in range(len(connection_ips)):
            self.net_threads[connection_ips[i]].join()

    def _thread_network_testing(self, connection_ip):
        url = None
        if self.edge_based:
            url = connection_ip
        network_time_start = time.time()
        msg1 = [message_utils.test_server_network_from_server(), self.uninet.cpu().state_dict()]
        self.send_msg(exchange=connection_ip, msg=msg1, url=url, is_weight=True)
        fed_logger.info("server test network sent")
        network_time_end = time.time()

        msg2 = self.recv_msg(exchange=connection_ip,
                             expect_msg_type=message_utils.test_server_network_from_connection(),
                             url=url, is_weight=True)
        fed_logger.info("server test network received")
        self.edge_bandwidth[connection_ip] = data_utils.sizeofmessage(msg1) / (network_time_end - network_time_start)
        fed_logger.info(Fore.LIGHTBLUE_EX + f"Cloud-Edge BW: {self.edge_bandwidth}")

    def client_network(self, edge_ips):
        """
        receive client network speed
        """
        url = None
        if self.edge_based:
            for edge in edge_ips:
                start_transmission()
                if self.simnet:
                    set_simnet(self.edge_bandwidth[edge])
                msg = self.recv_msg(exchange=edge, expect_msg_type=message_utils.client_network(), url=edge)
                end_transmission(data_utils.sizeofmessage(msg))
                for k in msg[1].keys():
                    self.client_bandwidth[k] = msg[1][k]
        else:
            for client in config.CLIENTS_LIST:
                msg = self.recv_msg(exchange=client, expect_msg_type=message_utils.simnet_bw_client_to_edge(), url=url)
                end_transmission(data_utils.sizeofmessage(msg))
                self.client_bandwidth[client] = msg[1]

    def get_simnet_edge_network(self):
        """
        get all edge's BW
        """

        for edgeIP in config.EDGE_SERVER_LIST:
            start_transmission()
            set_simnet(self.edge_bandwidth[edgeIP])
            msg = self.recv_msg(exchange=edgeIP,
                                expect_msg_type=message_utils.simnet_bw_edge_to_server(),
                                is_weight=False,
                                url=edgeIP)
            self.edge_bandwidth[edgeIP] = msg[1]
            end_transmission(data_utils.sizeofmessage(msg))

    def split_layer(self):
        """
        send splitting data
        """
        msg = [message_utils.split_layers(), self.split_layers]
        if self.simnet:
            for client in config.CLIENTS_LIST:
                start_transmission()
                set_simnet(self.client_bandwidth[client])
                end_transmission(data_utils.sizeofmessage(msg))
        self.scatter(msg)

    def send_split_layers_config(self):
        """
        send splitting data
        """
        msg = [message_utils.split_layers_server_to_edge(), self.split_layers, self.edge_nice_value]

        if self.simnet:
            for edge in config.EDGE_SERVER_LIST:
                start_transmission()
                set_simnet(self.edge_bandwidth[edge])
                end_transmission(data_utils.sizeofmessage(msg))
        self.scatter(msg)

    def e_local_weights(self, client_ips):
        """
        gets final weights for aggregation
        """
        eweights = {}
        for i in range(len(client_ips)):
            start_transmission()
            set_simnet(self.edge_bandwidth[config.CLIENT_MAP[client_ips[i]]])
            msg = self.recv_msg(config.CLIENT_MAP[client_ips[i]],
                                message_utils.local_weights_edge_to_server() + "_" + client_ips[i], True,
                                config.CLIENT_MAP[client_ips[i]])
            self.tt_end[client_ips[i]] = time.time()
            end_transmission(data_utils.sizeofmessage(msg))
            eweights[client_ips[i]] = msg[1]
        return eweights

    def energy_tt(self, client_ips):
        """
        get energy consumption, training time, remaining energy and utilization of clients
        :param client_ips:
        :return:
        """
        energy_tt = {}
        for client in client_ips:
            energy_tt[client] = []

        for client in config.CLIENTS_LIST:
            start_transmission()
            set_simnet(self.client_bandwidth[client])
            msg = self.recv_msg(exchange=client, expect_msg_type=message_utils.energy_client_to_server() + '_' + client,
                                is_weight=False, url=None)
            end_transmission(data_utils.sizeofmessage(msg))
            energy_tt[client] = msg[1:]

        self.client_remaining_energy = {}
        self.client_utilization = {}
        self.client_energy = {}
        self.client_comm_energy = {}
        for client in energy_tt.keys():
            op = self.split_layers[config.CLIENTS_CONFIG[client]]
            self.client_energy[client] = (energy_tt[client][0] + energy_tt[client][1])
            self.client_comp_energy[client][op] = energy_tt[client][0]
            self.client_comm_energy[client] = energy_tt[client][1]
            self.client_remaining_energy[client] = energy_tt[client][3]
            self.client_utilization[client] = energy_tt[client][4]
            self.client_comp_time[client][op] = (energy_tt[client][0] /
                                                 (self.client_utilization[client] * self.power_usage_of_client[client][
                                                     0]))
        return energy_tt

    def e_energy_tt(self, client_ips) -> dict:
        """
        get energy consumption, training time, remaining energy and utilization of clients from edges
        :param client_ips:
        :return:
        """
        energy_tt_list = []

        energy_tt = {}
        for client in client_ips:
            energy_tt[client] = []

        for edge in config.EDGE_SERVER_LIST:
            start_transmission()
            set_simnet(self.edge_bandwidth[edge])
            msg = self.recv_msg(exchange=edge, expect_msg_type=message_utils.energy_tt_edge_to_server(), url=edge)
            end_transmission(data_utils.sizeofmessage(msg))
            for i in range(len(config.EDGE_MAP[edge])):
                # energy_tt_list.append(msg[1][i])
                if msg[1].__contains__(config.EDGE_MAP[edge][i]):
                    energy_tt[config.EDGE_MAP[edge][i]] = msg[1][config.EDGE_MAP[edge][i]]
                    self.computation_time_of_each_client_on_edges[config.EDGE_MAP[edge][i]] = \
                        msg[1][config.EDGE_MAP[edge][i]][5]
                    self.total_computation_time_of_each_edge[edge] = msg[1][config.EDGE_MAP[edge][i]][6]

                else:
                    energy_tt[config.EDGE_MAP[edge][i]] = [0, 0, 0, 0, 0, 0]
                    self.computation_time_of_each_client_on_edges[config.EDGE_MAP[edge][i]] = 0

        self.client_remaining_energy = {}
        self.client_utilization = {}
        self.client_energy = {}
        self.client_comm_energy = {}
        for client in energy_tt.keys():
            op1 = self.split_layers[config.CLIENTS_CONFIG[client]][0]
            self.client_energy[client] = (energy_tt[client][0] + energy_tt[client][1])
            self.client_comp_energy[client][op1] = energy_tt[client][0]
            self.client_comm_energy[client] = energy_tt[client][1]
            self.client_remaining_energy[client] = energy_tt[client][3]
            self.client_utilization[client] = energy_tt[client][4]
            self.client_comp_time[client][op1] = (energy_tt[client][0] /
                                                  (self.client_utilization[client] * self.power_usage_of_client[client][
                                                      0]))

        # self.client_remaining_energy = []
        # for i in range(len(energy_tt_list)):
        #     self.client_remaining_energy[](energy_tt_list[i][2])
        return energy_tt

    def c_local_weights(self, client_ips):
        cweights = {}
        for client_ip in client_ips:
            start_transmission()
            set_simnet(self.client_bandwidth[client_ip])
            msg = self.recv_msg(client_ip,
                                message_utils.local_weights_client_to_server(), True)
            self.tt_end[client_ip] = time.time()
            end_transmission(data_utils.sizeofmessage(msg))
            cweights[client_ip] = msg[1]

        return cweights

    def edge_offloading_global_weights(self):
        """
        send global weights to edges
        """
        msg = [message_utils.initial_global_weights_server_to_edge(), self.uninet.state_dict()]
        if self.simnet:
            for edgeIP in config.EDGE_SERVER_LIST:
                start_transmission()
                set_simnet(self.edge_bandwidth[edgeIP])
                end_transmission(data_utils.sizeofmessage(msg))
        self.scatter(msg, True)

    def no_offloading_global_weights(self):
        """
        send global weights to clients
        """
        msg = [message_utils.initial_global_weights_server_to_client(), self.uninet.state_dict()]
        if self.simnet:
            for client in config.CLIENTS_LIST:
                start_transmission()
                set_simnet(self.client_bandwidth[client])
                end_transmission(data_utils.sizeofmessage(msg))
        self.scatter(msg, True)

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    def split(self, state, options: dict):
        if options.get('splitting') == 'edge_based_heuristic':
            self.split_layers, approximated_energy, approximated_tt, edge_nice_value, server_nice_value = fl_method_parser.fl_methods.get(
                options.get('splitting'))(state, self.group_labels)
            self.actions.append(self.split_layers)
            self.approximated_energy_of_actions.append(approximated_energy)
            self.approximated_tt_of_actions.append(approximated_tt)
            self.server_nice_value = server_nice_value
            self.edge_nice_value = edge_nice_value
        else:
            self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def edge_based_state(self) -> dict:

        client_remaining_energy = {}
        if len(self.client_remaining_energy.keys()) == 0:
            for client, index in config.CLIENTS_CONFIG.items():
                client_remaining_energy[client] = 0
        else:
            for client, re in self.client_remaining_energy.items():
                client_remaining_energy[client] = re

        state = {'activation_size': self.activation_size,
                 'gradient_size': self.gradient_size,
                 'total_model_size': self.total_model_size,
                 'prev_action': self.split_layers,
                 'comp_time_of_each_client_on_edge': self.computation_time_of_each_client_on_edges,
                 'comp_time_of_each_client_on_server': self.process_wall_time,
                 'edge_server_comm_time': self.client_training_transmissionTime,
                 "client_comp_energy": self.client_comp_energy,
                 "client_comm_energy": self.client_comm_energy,
                 "client_comp_time": self.client_comp_time,
                 "client_power": self.power_usage_of_client,
                 "client_utilization": self.client_utilization,
                 "client_bw": self.client_bandwidth,
                 "edge_bw": self.edge_bandwidth,
                 "client_remaining_energy": client_remaining_energy,
                 "flops_of_each_layer": self.model_flops_per_layer,
                 "best_tt_splitting_found": self.best_tt_splitting_found,
                 "prev_edge_nice_value": self.edge_nice_value,
                 "prev_server_nice_value": self.server_nice_value,
                 "current_round": config.current_round
                 }

        return state

    def edge_based_reward_function_data(self, energy_tt_list, total_tt):
        energy = 0
        data = []
        tt = []
        for et in energy_tt_list:
            energy += et[0]
            tt.append(et[1])
        energy /= len(energy_tt_list)
        data.append(energy)
        data.append(total_tt)
        data.extend(tt)
        return data

    def e_client_attendance(self, client_ips):
        """
        Checks clients run ouf of charge or not
        Returns: Updates CLIENT_LIST
        """
        attend = {}
        for edge in list(config.EDGE_SERVER_LIST):
            msg = self.recv_msg(exchange=edge, expect_msg_type=message_utils.client_quit_edge_to_server(), url=edge)
            attend.update(msg[1])
            msg = [message_utils.client_quit_done(), True]
            self.send_msg(exchange=edge, msg=msg, url=edge)

        fed_logger.info(Fore.RED + f"{attend}")
        temp_list = []
        for client_ip in client_ips:
            if not attend[client_ip]:
                # config.CLIENTS_LIST.remove(client_ip)
                config.K -= 1
            else:
                temp_list.append(client_ip)
        config.CLIENTS_LIST = temp_list

    def client_attendance(self, client_ips):
        attend = {}
        for i in range(len(client_ips)):
            msg = self.recv_msg(client_ips[i], message_utils.client_quit_client_to_server() + '_' + client_ips[i])
            attend.update({client_ips[i]: msg[1]})
            msg = [message_utils.client_quit_done(), True]
            self.send_msg(client_ips[i], msg)

        temp_list = []
        for client_ip in client_ips:
            if attend[client_ip] == False:
                # config.CLIENTS_LIST.remove(client_ip)
                config.K -= 1
                config.S -= 1
            else:
                temp_list.append(client_ip)
        config.CLIENTS_LIST = temp_list

    def calculate_each_layer_activation_gradiant_size(self):
        split_layer = [[config.model_len - 1, config.model_len - 1]] * len(config.CLIENTS_CONFIG.keys())
        model = model_utils.get_model('Client', split_layer[0], self.device, self.edge_based)

        total_bits = 0
        for param_name, param in model.cpu().state_dict().items():
            num_elements = param.numel()
            bits = num_elements * 32
            total_bits += bits
        fed_logger.info(Fore.MAGENTA + f"Total Model Size (bit): {total_bits}")
        self.total_model_size = total_bits

        def calculate_size_in_megabits(tensor):
            num_elements = tensor.numel()
            size_in_bits = num_elements * 32  # assuming float32
            return size_in_bits

        activation_sizes = {}
        gradient_sizes = {}

        def forward_hook(module, input, output):
            activation_size = calculate_size_in_megabits(output)
            activation_sizes[module] = activation_size

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradient_size_mb = calculate_size_in_megabits(grad_output[0])
                gradient_sizes[module] = gradient_size_mb

        hooks = []
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Linear)):
                hooks.append(layer.register_forward_hook(forward_hook))
                hooks.append(layer.register_backward_hook(backward_hook))
                # layer.register_forward_hook(forward_hook)
                # layer.register_backward_hook(backward_hook)

        input_data = torch.randn(config.B, 3, 32, 32)

        output = model(input_data)
        output.mean().backward()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        layer_activation = {}
        layer_gradient = {}
        layer_num = 0
        current = None
        for num in list(activation_sizes.values()):
            if num != current:
                layer_activation[layer_num] = num
                layer_num += 1
                current = num
            elif num == 13107200 and current == 13107200 and 4 not in layer_activation:
                # Special case for the second occurrence of 12.5
                layer_activation[layer_num] = num
                layer_num += 1
        self.activation_size = layer_activation
        self.gradient_size = gradient_sizes
        fed_logger.info(Fore.MAGENTA + f"Each layer activation (bit): {self.activation_size}")
        fed_logger.info(Fore.MAGENTA + f"Each layer gradient (bit): {self.gradient_size}")

    def calculate_each_layer_FLOP(self):
        def count_relu_flops(input_shape) -> int:
            """Calculate FLOPS for ReLU layer."""
            batch_size, channels, height, width = input_shape
            # One comparison operation per element
            return batch_size * channels * height * width

        def count_linear_layer_flops(layer: nn.Linear) -> int:
            """Calculate FLOPS for a linear layer."""
            input_size, output_size = layer.weight.shape
            # Multiply + Add operations per neuron
            flops_per_neuron = 2 * input_size
            return flops_per_neuron * output_size

        def count_conv2d_layer_flops(layer: nn.Conv2d, input_shape) -> int:
            """Calculate FLOPS for a convolutional layer."""
            batch_size, in_channels, height, width = input_shape
            out_channels, kernel_height, kernel_width = layer.out_channels, layer.kernel_size[0], layer.kernel_size[1]

            # Convolution FLOPS calculation
            conv_flops = batch_size * out_channels * height * width * (in_channels * kernel_height * kernel_width)

            # Bias addition FLOPS
            bias_flops = batch_size * out_channels * height * width if layer.bias is not None else 0

            return conv_flops + bias_flops

        def count_batchnorm2d_flops(layer: nn.BatchNorm2d, input_shape) -> int:
            """Calculate FLOPS for a BatchNorm2d layer."""
            batch_size, channels, height, width = input_shape

            # Per-channel operations: scale, shift, normalization
            per_pixel_ops = 4  # multiply, divide, subtract mean, add epsilon
            return batch_size * channels * height * width * per_pixel_ops

        def count_maxpool_flops(layer: nn.MaxPool2d, input_shape) -> int:
            """Calculate FLOPS for a MaxPool layer."""
            batch_size, channels, height, width = input_shape
            kernel_height, kernel_width = layer.kernel_size, layer.kernel_size

            # Compare operations within each pooling window
            compare_ops_per_pixel = kernel_height * kernel_width - 1
            return batch_size * channels * height * width * compare_ops_per_pixel

        def analyze_model_flops(model: nn.Module, input_shape: tuple):
            """Analyze total FLOPS for a neural network model."""
            total_flops = 0
            layer_flops = {}
            current_shape = input_shape

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    flops = count_conv2d_layer_flops(module, current_shape)
                    layer_flops[module] = flops
                    total_flops += flops
                    # Update input shape for next layer
                    batch_size, _, height, width = current_shape
                    current_shape = (batch_size, module.out_channels,
                                     height // module.stride[0],
                                     width // module.stride[1])

                elif isinstance(module, nn.ReLU):
                    flops = count_relu_flops(current_shape)
                    layer_flops[module] = flops
                    total_flops += flops

                elif isinstance(module, nn.BatchNorm2d):
                    flops = count_batchnorm2d_flops(module, current_shape)
                    layer_flops[module] = flops
                    total_flops += flops

                elif isinstance(module, nn.MaxPool2d):
                    flops = count_maxpool_flops(module, current_shape)
                    layer_flops[module] = flops
                    total_flops += flops
                    # Update input shape for next layer
                    batch_size, channels, height, width = current_shape
                    current_shape = (batch_size, channels,
                                     height // module.kernel_size,
                                     width // module.kernel_size)

                elif isinstance(module, nn.Linear):
                    flops = count_linear_layer_flops(module)
                    layer_flops[module] = flops
                    total_flops += flops

            return layer_flops, total_flops

        split_layer = [[config.model_len - 1, config.model_len - 1]] * len(config.CLIENTS_CONFIG.keys())
        model: nn.Module = model_utils.get_model('Client', split_layer[0], self.device, self.edge_based)
        input_data = (config.B, 3, 32, 32)
        layer_flops, total_flops = analyze_model_flops(model, input_data)

        flops_per_layer = {}
        for i in range(config.model_len):
            flops_per_layer[i] = 0
        layer_num = -1
        convLayerDependency = 2
        for name, module in model.named_modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or
                    isinstance(module, nn.MaxPool2d) or isinstance(module, nn.BatchNorm2d) or
                    isinstance(module, nn.ReLU)):
                if isinstance(module, nn.Conv2d):
                    layer_num += 1
                    flops_per_layer[layer_num] += layer_flops[module]
                    convLayerDependency = 2
                elif convLayerDependency != 0:
                    flops_per_layer[layer_num] += layer_flops[module]
                    convLayerDependency -= 1
                elif convLayerDependency == 0:
                    layer_num += 1
                    flops_per_layer[layer_num] += layer_flops[module]
        fed_logger.info(Fore.MAGENTA + f"FLOPS PER LAYER: {flops_per_layer}")
        self.model_flops_per_layer = flops_per_layer

    def get_power_of_client(self):
        power_usage = {}
        if self.edge_based:
            for edgeIP in config.EDGE_SERVER_LIST:
                msg = self.recv_msg(exchange=edgeIP, url=edgeIP,
                                    expect_msg_type=message_utils.client_power_usage_edge_to_server())

                # each client has comp_power_usage and trans_power_usage
                for clientIP, power_usage_list in msg[1].items():
                    power_usage[clientIP] = power_usage_list
        else:
            for clientIP in config.CLIENTS_LIST:
                msg = self.recv_msg(exchange=clientIP, expect_msg_type=message_utils.client_power_usage_to_server())
                comp_power_usage = msg[1]
                trans_power_usage = msg[2]
                power_usage[clientIP] = [comp_power_usage, trans_power_usage]
        self.power_usage_of_client = power_usage

    def getFlopsOnEdgeAndServer(self):
        flop_on_server = 0
        flop_on_each_edge = {edge: 0 for edge in config.EDGE_SERVER_LIST}
        flop_of_each_edge_on_server = flop_on_each_edge
        flops_of_each_layer = self.model_flops_per_layer
        flops_of_each_layer = {key: flops_of_each_layer[key] for key in sorted(flops_of_each_layer)}
        flops_of_each_layer = list(flops_of_each_layer.values())

        flop_of_each_client_on_edge = {client: 0 for client in config.CLIENTS_LIST}
        flop_of_each_client_on_server = {client: 0 for client in config.CLIENTS_LIST}

        total_computation_time_on_server = self.total_computation_time
        if self.edge_based:
            for clientIP in config.CLIENTS_LIST:
                edge_flops = 0
                flop_of_edge_on_server = 0
                edgeIP = config.CLIENT_MAP[clientIP]
                clientAction = self.split_layers[config.CLIENTS_CONFIG[clientIP]]
                op1 = clientAction[0]
                op2 = clientAction[1]

                # offloading on client, edge and server
                if op1 < op2 < config.model_len - 1:
                    edge_flops = sum(flops_of_each_layer[op1 + 1:op2 + 1])
                    flop_of_edge_on_server = sum(flops_of_each_layer[op2 + 1:])
                    flop_on_server += sum(flops_of_each_layer[op2 + 1:])
                    flop_of_each_client_on_edge[clientIP] = edge_flops
                    flop_of_each_client_on_server[clientIP] = sum(flops_of_each_layer[op2 + 1:])

                # offloading on client and edge
                elif (op1 < op2) and op2 == config.model_len - 1:
                    edge_flops = sum(flops_of_each_layer[op1 + 1:op2 + 1])
                    flop_of_each_client_on_edge[clientIP] = edge_flops

                # offloading on client and server
                elif (op1 == op2) and op1 < config.model_len - 1:
                    flop_on_server += sum(flops_of_each_layer[op2 + 1:])
                    flop_of_edge_on_server = sum(flops_of_each_layer[op2 + 1:])
                    flop_of_each_client_on_server[clientIP] = sum(flops_of_each_layer[op2 + 1:])

                flop_on_each_edge[edgeIP] += edge_flops
                flop_of_each_edge_on_server[edgeIP] += flop_of_edge_on_server

            return flop_on_server, flop_on_each_edge, flop_of_each_edge_on_server, flop_of_each_client_on_edge, flop_of_each_client_on_server

        else:
            for clientIP in config.CLIENTS_LIST:
                clientOP = self.split_layers[config.CLIENTS_CONFIG[clientIP]]

                # offloading on client and server
                if clientOP < config.model_len - 1:
                    flop_on_server += sum(flops_of_each_layer[clientOP + 1:])

            return flop_on_server, None, None, None, None

    def remove_non_pickleable(self):
        self.calculate_each_layer_activation_gradiant_size = None

    def simnetTrainingTimeCalculation(self, aggregation_time, server_sequential_transmission_time, energy_tt_list,
                                      edgeBased=True):
        total_time_for_each_client = {}
        if edgeBased:
            if len(config.CLIENTS_LIST) > 0:
                for clientip in config.CLIENTS_LIST:
                    total_time_for_each_client[clientip] = \
                        self.computation_time_of_each_client_on_edges[clientip] + \
                        self.process_wall_time[clientip] + \
                        energy_tt_list[clientip][2] + \
                        self.client_training_transmissionTime[clientip]

                trainingTime_simnetBW = max(total_time_for_each_client.values())
                fed_logger.info(f"Training time of each client : {total_time_for_each_client}")
                fed_logger.info(f"Training time using Simnet bw : {trainingTime_simnetBW}")
                return trainingTime_simnetBW, total_time_for_each_client
            else:
                fed_logger.info("All Clients had been Turned off.")
                return 0
        else:
            fed_logger.info(
                Fore.MAGENTA + f"{self.computation_time_of_each_client}, {energy_tt_list}, "
                               f"{self.client_training_transmissionTime}, "
                               f"{aggregation_time}, {server_sequential_transmission_time}")
            if len(config.CLIENTS_LIST) > 0:
                for clientip in config.CLIENTS_LIST:
                    total_time_for_each_client[clientip] = self.computation_time_of_each_client[clientip] + \
                                                           energy_tt_list[clientip][2] + \
                                                           self.client_training_transmissionTime[clientip]

                trainingTime_simnetBW = (max(total_time_for_each_client.values())
                                         + aggregation_time + server_sequential_transmission_time)

                fed_logger.info(f"Training time using Simnet bw : {trainingTime_simnetBW}")
                return trainingTime_simnetBW, total_time_for_each_client
            else:
                fed_logger.info("All Clients had been Turned off.")
                return 0
