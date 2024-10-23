import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore

from app.entity.interface.fed_server_interface import FedServerInterface
from app.fl_method import fl_method_parser

sys.path.append('../../')
from app.util import message_utils, model_utils, data_utils
from app.config.logger import fed_logger
from app.util.energy_estimation import *

np.random.seed(0)
torch.manual_seed(0)


class FedServer(FedServerInterface):

    def initialize(self, split_layers, LR, simnetbw: dict = None):
        self.split_layers = split_layers
        self.nets = {}
        self.optimizers = {}
        if simnetbw is not None and self.simnet:
            set_simnet(simnetbw)
            self.simnetbw = simnetbw
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

                        if len(list(self.nets[client_ip].parameters())) != 0:
                            self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                                   momentum=0.9)
                else:
                    self.nets[client_ip] = model_utils.get_model('Server', split_layers[client_index], self.device,
                                                                 self.edge_based)
        self.criterion = nn.CrossEntropyLoss()

    def edge_offloading_train(self, client_ips):
        self.threads = {}
        for i in range(len(client_ips)):
            if self.simnet:
                self.computation_time_of_each_client[client_ips[i]] = 0
                self.client_training_transmissionTime[client_ips[i]] = 0
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_edge_training,
                                                           args=(client_ips[i],), name=client_ips[i])
            fed_logger.info(str(client_ips[i]) + ' offloading training start')
            self.threads[client_ips[i]].start()
            self.tt_start[client_ips[i]] = time.time()

        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

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
        self.threads = {}
        for i in range(len(client_ips)):
            self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading,
                                                           args=(client_ips[i],), name=client_ips[i])
            fed_logger.info(str(client_ips[i]) + ' no edge offloading training start')
            self.threads[client_ips[i]].start()
        for i in range(len(client_ips)):
            self.threads[client_ips[i]].join()

    def _thread_training_no_offloading(self, client_ip):
        pass

    def _thread_training_offloading(self, client_ip):
        # iteration = int((test_config.N / (test_config.K * test_config.B)))
        flag = self.recv_msg(client_ip,
                             message_utils.local_iteration_flag_client_to_server() + "_" + client_ip)[1]
        while flag:
            flag = self.recv_msg(client_ip,
                                 message_utils.local_iteration_flag_client_to_server() + "_" + client_ip)[1]
            if not flag:
                break
            # fed_logger.info(client_ip + " receiving local activations")
            msg = self.recv_msg(client_ip,
                                message_utils.local_activations_client_to_server() + "_" + client_ip, True)
            smashed_layers = msg[1]
            labels = msg[2]
            # fed_logger.info(client_ip + " training model")
            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
            outputs = self.nets[client_ip](inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]] < len(
                    self.uninet.cfg) - 1:
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()

            # Send gradients to edge
            # fed_logger.info(client_ip + " sending gradients")
            msg = [message_utils.server_gradients_server_to_client() + str(client_ip), inputs.grad]
            self.send_msg(client_ip, msg, True)

        fed_logger.info(str(client_ip) + 'no edge offloading training end')
        return 'Finish'

    def _thread_edge_training(self, client_ip):
        # iteration = int((test_config.N / (test_config.K * test_config.B)))
        i = 0
        msg = self.recv_msg(config.CLIENT_MAP[client_ip],
                            f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}',
                            url=config.CLIENT_MAP[client_ip])
        if self.simnet:
            self.client_training_transmissionTime[client_ip] += (data_utils.sizeofmessage(msg) / self.simnetbw)

        flag = msg[1]
        i += 1
        fed_logger.info(Fore.RED + f"{flag}")
        if not flag:
            fed_logger.info(str(client_ip) + ' offloading training end')
            return 'Finish'
        while flag:

            # fed_logger.info(client_ip + " receiving local activations")
            if self.split_layers[config.CLIENTS_CONFIG[client_ip]][1] < len(self.uninet.cfg) - 1:

                msg = self.recv_msg(config.CLIENT_MAP[client_ip],
                                    f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}',
                                    url=config.CLIENT_MAP[client_ip])
                if self.simnet:
                    self.client_training_transmissionTime[client_ip] += (data_utils.sizeofmessage(msg) / self.simnetbw)
                flag = msg[1]
                fed_logger.info(Fore.RED + f"{flag}")
                if not flag:
                    break
                msg = self.recv_msg(config.CLIENT_MAP[client_ip],
                                    f'{message_utils.local_activations_edge_to_server() + "_" + client_ip}_{i}', True,
                                    config.CLIENT_MAP[client_ip])
                if self.simnet:
                    self.client_training_transmissionTime[client_ip] += (data_utils.sizeofmessage(msg) / self.simnetbw)

                smashed_layers = msg[1]
                labels = msg[2]
                # fed_logger.info(client_ip + " training model")
                self.start_time_of_computation_each_client[client_ip] = time.time()
                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].zero_grad()
                outputs = self.nets[client_ip](inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.optimizers.keys().__contains__(client_ip):
                    self.optimizers[client_ip].step()
                if self.simnet:
                    self.computation_time_of_each_client[client_ip] += (
                            time.time() - self.start_time_of_computation_each_client[client_ip])
                # Send gradients to edge
                # fed_logger.info(client_ip + " sending gradients")
                msg = [f'{message_utils.server_gradients_server_to_edge() + str(client_ip)}_{i}', inputs.grad]
                if self.simnet:
                    self.client_training_transmissionTime[client_ip] += (data_utils.sizeofmessage(msg) / self.simnetbw)
                self.send_msg(config.CLIENT_MAP[client_ip], msg, True, config.CLIENT_MAP[client_ip])
            i += 1

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
                    w_local_list.append(w_local)
                else:
                    w_local = (eweights[client], config.N / config.K)
            else:
                w_local = (eweights[client], config.N / config.K)
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

        for i in edge_ips:
            start_transmission()
            msg = self.recv_msg(exchange=i, expect_msg_type=message_utils.client_network(), url=i)
            end_transmission(data_utils.sizeofmessage(msg))
            for k in msg[1].keys():
                self.client_bandwidth[k] = msg[1][k]

    def get_simnet_edge_network(self):
        """
        get all edge's BW
        """

        for edgeIP in config.EDGE_SERVER_LIST:
            start_transmission()
            msg = self.recv_msg(exchange=edgeIP, expect_msg_type=message_utils.simnet_bw_edge_to_server(),
                                is_weight=False)
            self.edge_bandwidth[edgeIP] = msg[1]
            end_transmission(data_utils.sizeofmessage(msg))

    def split_layer(self):
        """
        send splitting data
        """
        msg = [message_utils.split_layers(), self.split_layers]
        self.scatter(msg)

    def get_split_layers_config_from_edge(self):
        """
        send splitting data
        """
        start_transmission()
        msg = [message_utils.split_layers_server_to_edge(), self.split_layers]
        end_transmission(len(config.EDGE_SERVER_LIST) * data_utils.sizeofmessage(msg))
        self.scatter(msg)

    def e_local_weights(self, client_ips):
        """
        send final weights for aggregation
        """
        eweights = {}
        for i in range(len(client_ips)):
            start_transmission()
            msg = self.recv_msg(config.CLIENT_MAP[client_ips[i]],
                                message_utils.local_weights_edge_to_server() + "_" + client_ips[i], True,
                                config.CLIENT_MAP[client_ips[i]])
            self.tt_end[client_ips[i]] = time.time()
            end_transmission(data_utils.sizeofmessage(msg))
            eweights[client_ips[i]] = msg[1]
        return eweights

    def e_energy_tt(self, client_ips):
        """
        Returns: average energy consumption of clients
        """
        energy_tt_list = []
        for edge in list(config.EDGE_SERVER_LIST):
            start_transmission()
            msg = self.recv_msg(exchange=edge, expect_msg_type=message_utils.energy_tt_edge_to_server(), url=edge)
            end_transmission(data_utils.sizeofmessage(msg))
            for i in range(len(config.EDGE_MAP[edge])):
                energy_tt_list.append(msg[1][i])
                if self.simnet:
                    self.computation_time_of_each_client_on_edges[config.EDGE_MAP[edge][i]] = msg[1][i][3]
        self.client_remaining_energy = []
        for i in range(len(energy_tt_list)):
            self.client_remaining_energy.append(energy_tt_list[i][2])
        return energy_tt_list

    def c_local_weights(self, client_ips):
        cweights = []
        for i in range(len(client_ips)):
            msg = self.recv_msg(client_ips[i],
                                message_utils.local_weights_client_to_server(), True)
            self.tt_end[client_ips[i]] = time.time()
            cweights.append(msg[1])
        return cweights

    def edge_offloading_global_weights(self):
        """
        send global weights
        """
        msg = [message_utils.initial_global_weights_server_to_edge(), self.uninet.state_dict()]
        start_transmission()
        dataSize = len(config.EDGE_SERVER_LIST) * data_utils.sizeofmessage(msg)
        end_transmission(dataSize)
        self.scatter(msg, True)

    def no_offloading_global_weights(self):
        msg = [message_utils.initial_global_weights_server_to_client(), self.uninet.state_dict()]
        self.scatter(msg, True)

    def cluster(self, options: dict):
        self.group_labels = fl_method_parser.fl_methods.get(options.get('clustering'))()

    def split(self, state, options: dict):
        self.split_layers = fl_method_parser.fl_methods.get(options.get('splitting'))(state, self.group_labels)
        fed_logger.info('Next Round OPs: ' + str(self.split_layers))

    def edge_based_state(self):
        state = []
        for i in self.client_bandwidth:
            state.append(self.client_bandwidth[i])
        for i in self.edge_bandwidth:
            state.append(self.edge_bandwidth[i])

        if self.simnet:
            state.append(self.simnetbw)

        if len(self.client_remaining_energy) == 0:
            for i in range(config.K):
                state.append(0)
        else:
            for i in self.client_remaining_energy:
                state.append(i)
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
                config.CLIENTS_LIST.remove(client_ip)
                config.K -= 1
        #     else:
        #         temp_list.append(client_ip)
        # config.CLIENTS_LIST = temp_list

    def client_attendance(self, client_ips):
        attend = {}
        for i in range(len(client_ips)):
            msg = self.recv_msg(client_ips[i],
                                message_utils.client_quit_client_to_server())
            attend.update({client_ips[i], msg[1]})
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
