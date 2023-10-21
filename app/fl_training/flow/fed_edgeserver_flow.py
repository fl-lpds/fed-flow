import socket
import sys
import threading

from app.util import message_utils

sys.path.append('../../../')
from app.config import config
from app.config.logger import fed_logger
from app.entity.edge_server import FedEdgeServer
from app.entity.interface.fed_edgeserver_interface import FedEdgeServerInterface


def run_offload(server: FedEdgeServerInterface, LR):
    server.initialize(config.split_layer, LR, config.EDGE_MAP[server.ip])

    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_MAP[server.ip]
    for r in range(config.R):
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r))
        fed_logger.info("receiving global weights")
        server.global_weights(client_ips)
        # fed_logger.info("test clients network")
        # server.test_client_network(client_ips)
        # fed_logger.info("sending clients network")
        # server.client_network()
        # fed_logger.info("test server network")
        # server.test_server_network()
        fed_logger.info("receiving and sending splitting info")
        server.split_layer(client_ips)
        fed_logger.info("initializing server")
        server.initialize(server.split_layers, LR, client_ips)
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=server.thread_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()

        if r > 49:
            LR = config.LR * 0.1


def run(options_ins):
    LR = config.LR
    ip_address = socket.gethostname()
    fed_logger.info('Preparing Sever.')
    edge_server_ins = FedEdgeServer(ip_address, config.EDGESERVER_PORT[ip_address], config.SERVER_ADDR,
                                    config.SERVER_PORT, options_ins.get('model'),
                                    options_ins.get('dataset'))
    fed_logger.info("start mode: " + str(options_ins.values()))
    run_offload(edge_server_ins, LR)
    msg = edge_server_ins.recv_msg(edge_server_ins.central_server_communicator.sock, message_utils.finish)
    edge_server_ins.scatter(msg)