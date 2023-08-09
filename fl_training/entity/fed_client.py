import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

sys.path.append('../../')
from config import config
from util import model_utils, message_utils
from fl_training.interface.fed_client_interface import FedClientInterface
from config.logger import fed_logger

np.random.seed(0)
torch.manual_seed(0)


class Client(FedClientInterface):

    def initialize(self, split_layer, offload, first, LR):
        if offload or first:
            self.split_layer = split_layer

            fed_logger.debug('Building Model.')
            self.net = model_utils.get_model('Client', self.split_layer, self.device)
            fed_logger.debug(self.net)
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
                                   momentum=0.9)
        fed_logger.debug('Receiving Global Weights..')
        weights = self.recv_msg(self.sock, message_utils.initial_global_weights_server_to_client)[1]
        if self.split_layer == (config.model_len - 1):
            self.net.load_state_dict(weights)
        else:
            pweights = model_utils.split_weights_client(weights, self.net.state_dict())
            self.net.load_state_dict(pweights)
        fed_logger.debug('Initialize Finished')

    def train(self, trainloader):
        # Network speed test
        network_time_start = time.time()
        msg = [message_utils.test_network, self.uninet.cpu().state_dict()]
        self.send_msg(self.sock, msg)
        msg = self.recv_msg(self.sock, message_utils.test_network)[1]
        network_time_end = time.time()
        network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start)  # Mbit/s

        fed_logger.info('Network speed is {:}'.format(network_speed))
        msg = [message_utils.test_network, self.ip, network_speed]
        self.send_msg(self.sock, msg)

        # Training start
        s_time_total = time.time()
        time_training_c = 0
        self.net.to(self.device)
        self.net.train()
        if self.split_layer == (config.model_len - 1):  # No offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

        else:  # Offloading training
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)

                msg = [message_utils.local_activations_client_to_server, outputs.cpu(), targets.cpu()]
                self.send_msg(self.sock, msg)

                # Wait receiving server gradients
                gradients = self.recv_msg(self.sock)[1].to(self.device)

                outputs.backward(gradients)
                self.optimizer.step()

        e_time_total = time.time()
        fed_logger.info('Total time: ' + str(e_time_total - s_time_total))

        training_time_pr = (e_time_total - s_time_total) / int((config.N / (config.K * config.B)))
        fed_logger.info('training_time_per_iteration: ' + str(training_time_pr))

        msg = [message_utils.training_time_per_iteration_client_to_server, self.ip, training_time_pr]
        self.send_msg(self.sock, msg)

        return e_time_total - s_time_total

    def upload(self):
        msg = [message_utils.local_weights_client_to_server, self.net.cpu().state_dict()]
        self.send_msg(self.sock, msg)