import time

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_server import FedServer
from app.entity.node_type import NodeType
from app.util import model_utils
from app.util import graph_utils


def run_centralized(server: FedServer, learning_rate: float, options):
    server.initialize(learning_rate)
    server.scatter_split_layers()
    training_time = []
    transferred_data = []
    rounds = []
    accuracy = []
    for r in range(config.R):
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("Scatter split config")
        server.scatter_split_layers()
        fed_logger.info("sending global weights")
        server.scatter_global_weights()

        s_time = time.time()

        fed_logger.info("test neighbors network")
        server.gather_neighbors_network_bandwidth()

        fed_logger.info("getting bandwidth")
        neighbors_bandwidth = server.get_neighbors_bandwidth()
        bw = [bw[1].bandwidth for bw in neighbors_bandwidth.items()]
        if len(neighbors_bandwidth) == 0:
            transferred_data.append(0)
        else:
            transferred_data.append(sum(bw) / len(bw))

        fed_logger.info("splitting")
        server.split(bw, options)

        fed_logger.info("start training")
        server.start_edge_training()

        fed_logger.info("receiving local weights")
        local_weights = server.gather_clients_local_weights()

        fed_logger.info("aggregating weights")
        server.aggregate(local_weights)

        e_time = time.time()

        elapsed_time = e_time - s_time
        training_time.append(elapsed_time)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        accuracy.append(test_acc)

        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(elapsed_time))

    graph_utils.report_results(server, training_time, transferred_data, accuracy)


def run_d2d(server: FedServer, options):
    training_time = []
    accuracy = []
    rounds = []
    transferred_data = []

    for r in range(config.R):
        config.current_round = r
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("sending global weights")
        server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()
        server.choose_random_leader_per_cluster()
        local_weights = server.receive_leaders_local_weights()
        server.d2d_aggregate(local_weights)

        e_time = time.time()
        elapsed_time = e_time - s_time
        training_time.append(elapsed_time)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        fed_logger.info(f"Test Accuracy : {test_acc}")
        accuracy.append(test_acc)
        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(elapsed_time))

    graph_utils.report_results(server, training_time, [0] * len(training_time), accuracy)


def run(options_ins):
    learning_rate = config.learning_rate
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options_ins.values()))
    aggregator = create_aggregator(options_ins.get('aggregation'))
    fed_server = FedServer(options_ins.get('ip'), options_ins.get('port'), options_ins.get('model'),
                           options_ins.get('dataset'), aggregator, config.CURRENT_NODE_NEIGHBORS)
    d2d = options_ins.get('d2d')
    if d2d:
        run_d2d(fed_server, options_ins)
    else:
        run_centralized(fed_server, learning_rate, options_ins)
    time.sleep(10)
    fed_server.stop_server()
