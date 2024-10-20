import socket
import threading
import time
import sys

from colorama import Fore
from torch import optim, nn
sys.path.append('../../')

import config
from app.config.logger import fed_logger
from app.entity.interface.fed_edgeserver_interface import FedEdgeServerInterface
from app.util import message_utils, model_utils, energy_estimation, data_utils
from app.util.energy_estimation import *


class FedEdgeServer(FedEdgeServerInterface):
    def initialize(self, split_layers, LR, client_ips, simnetbw: float = None):
        self.split_layers = split_layers
        self.optimizers = {}
        if simnetbw is not None and self.simnet:
            set_simnet(simnetbw)
        for i in range(len(split_layers)):
            if client_ips.__contains__(config.CLIENTS_INDEX[i]):
                client_ip = config.CLIENTS_INDEX[i]
                if client_ip in client_ips:
                    client_index = config.CLIENTS_CONFIG[client_ip]
                    if split_layers[client_index][0] < split_layers[client_index][
                        1]:  # Only offloading client need initialize optimizer in server
                        self.nets[client_ip] = model_utils.get_model('Edge', split_layers[client_index], self.device,
                                                                     True)

                        # offloading weight in server also need to be initialized from the same global weight
                        cweights = model_utils.get_model('Client', split_layers[client_index], self.device,
                                                         True).state_dict()

                        pweights = model_utils.split_weights_edgeserver(self.uninet.state_dict(), cweights,
                                                                        self.nets[client_ip].state_dict())
                        self.nets[client_ip].load_state_dict(pweights)
                        if len(list(self.nets[client_ip].parameters())) != 0:
                            self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
                                                                   momentum=0.9)
                    else:
                        self.nets[client_ip] = model_utils.get_model('Edge', split_layers[client_index], self.device,
                                                                     True)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, client_ips, aggregate_method):
        pass

    def forward_propagation(self, client_ip):
        i = 0
        msg = self.recv_msg(client_ip, f'{message_utils.local_iteration_flag_client_to_edge()}_{i}_{client_ip}')
        flag = msg[1]
        if self.split_layers[config.CLIENTS_CONFIG.get(client_ip)][1] < model_utils.get_unit_model_len() - 1:
            flmsg = [f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}', flag]
            self.send_msg(config.EDGE_SERVER_CONFIG[config.index], flmsg)
        else:
            flmsg = [f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}', False]
            self.send_msg(config.EDGE_SERVER_CONFIG[config.index], flmsg)
        i += 1

        while flag:
            if self.split_layers[config.CLIENTS_CONFIG.get(client_ip)][0] < model_utils.get_unit_model_len() - 1:
                msg = self.recv_msg(client_ip, f'{message_utils.local_iteration_flag_client_to_edge()}_{i}_{client_ip}')
                flag = msg[1]

                if not flag:
                    flmsg = [f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}', flag]
                    self.send_msg(config.EDGE_SERVER_CONFIG[config.index], flmsg)
                    break
                # fed_logger.info(client_ip + " receiving local activations")
                msg = self.recv_msg(exchange=client_ip,
                                    expect_msg_type=f'{message_utils.local_activations_client_to_edge()}_{i}_{client_ip}',
                                    is_weight=True)
                smashed_layers = msg[1]
                labels = msg[2]

                # fed_logger.info(client_ip + " training model forward")

                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                if self.split_layers[config.CLIENTS_CONFIG[client_ip]][0] < \
                        self.split_layers[config.CLIENTS_CONFIG[client_ip]][1]:
                    energy_estimation.computation_start()
                    self.start_time_of_computation_each_client[client_ip] = time.time()
                    if self.optimizers.keys().__contains__(client_ip):
                        self.optimizers[client_ip].zero_grad()
                    outputs = self.nets[client_ip](inputs)
                    self.computation_time_of_each_client[client_ip] += (time.time() - \
                                                                        self.start_time_of_computation_each_client[
                                                                            client_ip])
                    energy_estimation.computation_end()
                    # fed_logger.info(client_ip + " sending local activations")
                    if self.split_layers[config.CLIENTS_CONFIG[client_ip]][1] < model_utils.get_unit_model_len() - 1:
                        flmsg = [f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}', flag]
                        self.send_msg(config.EDGE_SERVER_CONFIG[config.index], flmsg)
                        msg = [f'{message_utils.local_activations_edge_to_server() + "_" + client_ip}_{i}',
                               outputs.cpu(),
                               targets.cpu()]
                        self.send_msg(exchange=config.EDGE_SERVER_CONFIG[config.index], msg=msg, is_weight=True)
                        msg = self.recv_msg(exchange=config.EDGE_SERVER_CONFIG[config.index],
                                            expect_msg_type=f'{message_utils.server_gradients_server_to_edge() + client_ip}_{i}',
                                            is_weight=True)
                        gradients = msg[1].to(self.device)
                        # fed_logger.info(client_ip + " training model backward")
                        energy_estimation.computation_start()
                        self.start_time_of_computation_each_client[client_ip] = time.time()
                        outputs.backward(gradients)
                        self.computation_time_of_each_client[client_ip] += (
                                time.time() - self.start_time_of_computation_each_client[client_ip])
                        energy_estimation.computation_end()
                        msg = [f'{message_utils.server_gradients_edge_to_client() + client_ip}_{i}', inputs.grad]
                        self.send_msg(exchange=client_ip, msg=msg, is_weight=True)
                    else:
                        energy_estimation.computation_start()
                        self.start_time_of_computation_each_client[client_ip] = time.time()
                        outputs = self.nets[client_ip](inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        if self.optimizers.keys().__contains__(client_ip):
                            self.optimizers[client_ip].step()
                        self.computation_time_of_each_client += (
                                time.time() - self.start_time_of_computation_each_client[client_ip])
                        energy_estimation.computation_end()
                        msg = [f'{message_utils.server_gradients_edge_to_client() + client_ip}_{i}', inputs.grad]
                        self.send_msg(exchange=client_ip, msg=msg, is_weight=True)
                else:
                    flmsg = [f'{message_utils.local_iteration_flag_edge_to_server()}_{i}_{client_ip}', flag]
                    self.send_msg(config.EDGE_SERVER_CONFIG[config.index], flmsg)
                    msg = [f'{message_utils.local_activations_edge_to_server() + "_" + client_ip}_{i}', inputs.cpu(),
                           targets.cpu()]
                    self.send_msg(exchange=config.EDGE_SERVER_CONFIG[config.index], msg=msg, is_weight=True)
                    # fed_logger.info(client_ip + " edge receiving gradients")
                    msg = self.recv_msg(exchange=config.EDGE_SERVER_CONFIG[config.index],
                                        expect_msg_type=f'{message_utils.server_gradients_server_to_edge() + client_ip}_{i}',
                                        is_weight=True)
                    # fed_logger.info(client_ip + " edge received gradients")
                    msg = [f'{message_utils.server_gradients_edge_to_client() + client_ip}_{i}', msg[1]]
                    self.send_msg(exchange=client_ip, msg=msg, is_weight=True)
            i += 1
        fed_logger.info(str(client_ip) + ' offloading training end')

    def backward_propagation(self, outputs, client_ip, inputs):
        pass

    def test_client_network(self, client_ips):
        """
        send message to test_app network speed
        """
        # Network test_app
        self.net_threads = {}
        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_client_network_testing,
                                                               args=(client_ips[i],))
            self.net_threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            self.net_threads[client_ips[i]].join()

    def _thread_client_network_testing(self, client_ip):
        network_time_start = time.time()
        msg = [message_utils.test_network_edge_to_client(), self.uninet.cpu().state_dict()]
        self.send_msg(exchange=client_ip, msg=msg, is_weight=True)
        msg = self.recv_msg(exchange=client_ip, expect_msg_type=message_utils.test_network_client_to_edge(),
                            is_weight=True)
        network_time_end = time.time()
        self.client_bandwidth[client_ip] = data_utils.sizeofmessage(msg) / (network_time_end - network_time_start)

    def test_server_network(self):
        msg = self.recv_msg(exchange=config.EDGE_SERVER_CONFIG[config.index],
                            expect_msg_type=message_utils.test_server_network_from_server(), is_weight=True)
        msg = [message_utils.test_server_network_from_connection(), self.uninet.cpu().state_dict()]
        self.send_msg(exchange=config.EDGE_SERVER_CONFIG[config.index], msg=msg, is_weight=True)

    def client_network(self):
        """
        send client network speed to central server
        """
        msg = [message_utils.client_network(), self.client_bandwidth]
        self.send_msg(config.EDGE_SERVER_CONFIG[config.index], msg)

    def send_simnet_bw_to_server(self, simnetbw):
        msg = [message_utils.simnet_bw_edge_to_server(), simnetbw]
        self.send_msg(exchange=config.EDGE_SERVER_CONFIG[config.index], msg=msg, is_weight=False)
        fed_logger.info("Simnet BW sent")

    def get_simnet_client_network(self):
        for clientip in config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]:
            msg = self.recv_msg(exchange=clientip, expect_msg_type=message_utils.simnet_bw_client_to_edge(),
                                is_weight=False)
            self.client_bandwidth[clientip] = msg[1]

    def get_split_layers_config(self, client_ips):
        """
        receive send splitting data to clients
        """
        msg = self.recv_msg(config.EDGE_SERVER_CONFIG[config.index],
                            message_utils.split_layers_server_to_edge())
        self.split_layers = msg[1]
        fed_logger.info(Fore.LIGHTYELLOW_EX + f"{msg[1]}")
        msg = [message_utils.split_layers_edge_to_client(), self.split_layers]
        self.scatter(msg)
        # for i in range(len(self.split_layers)):
        #     if client_ips.__contains__(config.CLIENTS_LIST[i]):
        #         client_ip = config.CLIENTS_LIST[i]
        #         msg = [message_utils.split_layers, self.split_layers[i]]
        #         self.send_msg(self.socks[socket.gethostbyname(client_ip)], msg)

    def local_weights(self, client_ip):
        """
        receive and send final weights for aggregation
        """
        cweights = self.recv_msg(client_ip,
                                 message_utils.local_weights_client_to_edge(), True)[1]
        sp = self.split_layers[config.CLIENTS_CONFIG[client_ip]][0]
        if sp != (config.model_len - 1):
            w_local = model_utils.concat_weights(self.uninet.state_dict(), cweights,
                                                 self.nets[client_ip].state_dict())
        else:
            w_local = cweights
            # print("------------------------"+str(eweights[i]))
        msg = [message_utils.local_weights_edge_to_server() + "_" + client_ip, w_local]
        self.send_msg(config.EDGE_SERVER_CONFIG[config.index], msg, True)

    def energy(self, client_ips):
        energy_tt_list = []
        edge_tt_list = {}
        for c in config.CLIENTS_CONFIG.keys():
            if client_ips.__contains__(c):
                ms = self.recv_msg(c, message_utils.energy_client_to_edge() + "_" + c)
                fed_logger.info(f"client message: {ms}")
                energy_tt_list.append([ms[1], ms[2], ms[3], self.computation_time_of_each_client[c]])
            else:
                energy_tt_list.append([0, 0, 0, 0])
        # fed_logger.info(f"sending enery tt {socket.gethostname()}")
        msg = [message_utils.energy_tt_edge_to_server(), energy_tt_list, ]
        self.send_msg(config.EDGE_SERVER_CONFIG[config.index], msg)

    def global_weights(self, client_ips: []):
        """
        receive and send global weights
        """
        # we do not use start and end transmission, because we calculated this time when cloud send global weights to edge
        # and because the broker is at edge, we assume no time need to download weight from broker into edge
        weights = self.recv_msg(config.EDGE_SERVER_CONFIG[config.index],
                                message_utils.initial_global_weights_server_to_edge(), True)
        weights = weights[1]
        for i in range(len(self.split_layers)):
            if client_ips.__contains__(config.CLIENTS_INDEX[i]):
                cweights = model_utils.get_model('Client', self.split_layers[i], self.device, True).state_dict()
                pweights = model_utils.split_weights_edgeserver(weights, cweights,
                                                                self.nets[config.CLIENTS_INDEX[i]].state_dict())
                self.nets[config.CLIENTS_INDEX[i]].load_state_dict(pweights)

        msg = [message_utils.initial_global_weights_edge_to_client(), weights]
        self.scatter(msg, True)

    def no_offload_global_weights(self):
        weights = \
            self.recv_msg(config.EDGE_SERVER_CONFIG[config.index],
                          message_utils.initial_global_weights_server_to_edge(), True)[1]
        msg = [message_utils.initial_global_weights_edge_to_client(), weights]
        self.scatter(msg, True)

    def thread_offload_training(self, client_ip):
        self.forward_propagation(client_ip)
        self.local_weights(client_ip)
        # self.energy(client_ip)

    def thread_no_offload_training(self, client_ip):
        self.local_weights(client_ip)

    def client_attendance(self, client_ips):
        attend = {}
        for client_ip in client_ips:
            ms = self.recv_msg(client_ip,
                               message_utils.client_quit_client_to_edge() + "_" + client_ip)
            msg = [message_utils.client_quit_done(), True]
            self.send_msg(client_ip, msg)
            attend[client_ip] = ms[1]
            if ms[1] == False:
                config.K -= 1
                for c in config.EDGE_MAP[socket.gethostname()]:
                    if c == client_ip:
                        tmp = list(config.EDGE_MAP[socket.gethostname()])
                        tmp.remove(c)
                        config.EDGE_MAP[socket.gethostname()] = tmp
                config.CLIENT_MAP.pop(client_ip)
                fed_logger.info(f"removing client {client_ip}")
                config.CLIENTS_LIST.remove(client_ip)
        # fed_logger.info(f"sending enery tt {socket.gethostname()}")
        msg = [message_utils.client_quit_edge_to_server(), attend]
        self.send_msg(config.EDGE_SERVER_CONFIG[config.index], msg)
        self.recv_msg(exchange=config.EDGE_SERVER_CONFIG[config.index],
                      expect_msg_type=message_utils.client_quit_done())
