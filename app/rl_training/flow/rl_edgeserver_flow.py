import socket
import sys
import threading

from app.util import message_utils, rl_utils

sys.path.append('../../../')
from app.config import config
from app.config.logger import fed_logger
from app.entity.edge_server import FedEdgeServer


def run(options_ins):
    LR = config.learning_rate
    ip_address = socket.gethostname()
    # fed_logger.info('Preparing Sever.')
    edge_server = FedEdgeServer(options_ins.get('model'),options_ins.get('dataset'),offload=options_ins.get('offload'))
    # fed_logger.info("start mode: " + str(options_ins.values()))

    edge_server.initialize(config.split_layer, LR, config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]])

    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_NAME_TO_CLIENTS_NAME[config.EDGE_SERVER_INDEX_TO_NAME[config.index]]

    preTrain(edge_server, options_ins, client_ips)

    for r in range(config.max_episodes):
        # fed_logger.info('====================================>')
        # fed_logger.info('==> Episode {:} Start'.format(r))
        #
        # fed_logger.info("receiving global weights")
        edge_server.get_global_weights(client_ips)
        # fed_logger.info("test clients network")
        # server.test_client_network(client_ips)
        # fed_logger.info("sending clients network")
        # server.client_network()
        # fed_logger.info("test server network")
        # server.test_server_network()

        # fed_logger.info("receiving and sending splitting info")
        edge_server.get_split_layers_config(client_ips)

        # fed_logger.info("initializing server")
        edge_server.initialize(edge_server.split_layers, LR, client_ips)
        threads = {}

        # fed_logger.info("start training")

        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=edge_server.thread_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()

        if r > 49:
            LR = config.learning_rate * 0.1
        edge_server.energy(client_ips)

        for i in range(config.max_timesteps):
            # fed_logger.info('====================================>')
            # fed_logger.info('==> Round {:} Start'.format(r))
            #
            # fed_logger.info("receiving global weights")
            edge_server.get_global_weights(client_ips)

            # fed_logger.info("test clients network")
            # server.test_client_network(client_ips)

            # fed_logger.info("sending clients network")
            # server.client_network()

            # fed_logger.info("test server network")
            # server.test_server_network()

            # fed_logger.info("receiving and sending splitting info")
            edge_server.get_split_layers_config(client_ips)

            # fed_logger.info("initializing server")
            edge_server.initialize(edge_server.split_layers, LR, client_ips)
            threads = {}

            # fed_logger.info("start training")
            for i in range(len(client_ips)):
                threads[client_ips[i]] = threading.Thread(target=edge_server.thread_offload_training,
                                                          args=(client_ips[i],), name=client_ips[i])
                threads[client_ips[i]].start()

            for i in range(len(client_ips)):
                threads[client_ips[i]].join()
            edge_server.energy(client_ips)
            if r > 49:
                LR = config.learning_rate * 0.1

    # msg = edge_server.recv_msg(edge_server.central_server_communicator.sock, message_utils.finish)
    # edge_server.scatter(msg)


def preTrain(edge_server, options, client_ips):
    # splittingArray = [6, 6]
    # edge_server.split_layers = [splittingArray * config.K]

    for i in range(10):

        fed_logger.info("receiving global weights")
        edge_server.get_global_weights(client_ips)
        # fed_logger.info("test clients network")
        # server.test_client_network(client_ips)
        # fed_logger.info("sending clients network")
        # server.client_network()
        # fed_logger.info("test server network")
        # server.test_server_network()
        fed_logger.info("receiving and sending splitting info")
        edge_server.get_split_layers_config(client_ips)
        fed_logger.info("initializing server")
        edge_server.initialize(edge_server.split_layers)
        threads = {}
        fed_logger.info("start training")
        for i in range(len(client_ips)):
            threads[client_ips[i]] = threading.Thread(target=edge_server.thread_offload_training,
                                                      args=(client_ips[i],), name=client_ips[i])
            threads[client_ips[i]].start()

        for i in range(len(client_ips)):
            threads[client_ips[i]].join()
        edge_server.energy(client_ips)