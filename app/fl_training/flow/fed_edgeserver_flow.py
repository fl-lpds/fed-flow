import os
import socket
import sys
import threading

from colorama import Fore

from app.util import message_utils, energy_estimation

sys.path.append('../../../')
from app.config import config
from app.config.logger import fed_logger
from app.entity.edge_server import FedEdgeServer
from app.entity.interface.fed_edgeserver_interface import FedEdgeServerInterface


def run_offload(server: FedEdgeServerInterface, LR):
    server.initialize(config.split_layer, LR, config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]])

    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(config.current_round))
        fed_logger.info("receiving global weights")
        server.global_weights(client_ips)
        fed_logger.info("test clients network")
        server.test_client_network(client_ips)
        fed_logger.info("sending clients network")
        server.client_network()
        fed_logger.info("test server network")
        server.test_server_network()
        fed_logger.info("receiving and sending splitting info")
        server.get_split_layers_config(client_ips)
        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR, client_ips)
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        server.energy(client_ips)
        if r > 49:
            LR = config.LR * 0.1
        energy = float(energy_estimation.energy())
        # energy /= batch_num
        fed_logger.info(Fore.LIGHTBLUE_EX + f"Energy : {energy}" + Fore.RESET)


def run_no_offload(server: FedEdgeServerInterface, LR):
    server.initialize(config.split_layer, LR, config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]])
    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))
        fed_logger.info("receiving global weights")
        server.no_offload_global_weights()
        # fed_logger.info("test clients network")
        # server.test_client_network(client_ips)
        # fed_logger.info("sending clients network")
        # server.client_network()
        # fed_logger.info("test server network")
        # server.test_server_network()
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_no_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        if r > 49:
            LR = config.LR * 0.1


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    offload = options_ins.get('offload')
    if offload:
        energy_estimation.init(os.getpid())
        edge_server_ins = FedEdgeServer(
            options_ins.get('model'),
            options_ins.get('dataset'), offload=offload)
        fed_logger.info("start mode: " + str(options_ins.values()))
        run_offload(edge_server_ins, LR)
    else:
        edge_server_ins = FedEdgeServer(options_ins.get('model'),
                                        options_ins.get('dataset'), offload=offload)
        fed_logger.info("start mode: " + str(options_ins.values()))
        run_no_offload(edge_server_ins, LR)
    # msg = edge_server_ins.recv_msg(config.SERVER_ADDR, message_utils.finish)
    # edge_server_ins.scatter(msg)
