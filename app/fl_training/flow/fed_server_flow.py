import sys
import time

from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_server import FedServer
from app.entity.node_type import NodeType

sys.path.append('../../../')
from app.config import config
from app.util import model_utils
from app.config.logger import fed_logger
from app.util import rl_utils


def run_centralized(server: FedServer, learning_rate: float, options):
    server.initialize(learning_rate)
    training_time = []
    transferred_data = []
    rounds = []
    accuracy = []
    for r in range(config.R):
        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

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
        server.scatter_split_layers()

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

    current_time = time.strftime("%Y-%m-%d %H:%M")
    runtime_config = f'{current_time} offload decentralized'
    rl_utils.draw_graph(10, 5, rounds, training_time, f"Server {str(server)} Training time", "FL Rounds",
                        "Training Time (s)",
                        f"Graphs/{runtime_config}",
                        f"trainingTime-{str(server)}", True)
    rl_utils.draw_graph(10, 5, rounds, transferred_data, f"Server {str(server)} Average clients BW", "FL Rounds",
                        "clients BW (bytes / s)",
                        f"Graphs/{runtime_config}",
                        f"client_bw-{str(server)}", True)
    rl_utils.draw_graph(10, 5, rounds, accuracy, f"Server {str(server)} Accuracy", "FL Rounds", "accuracy",
                        f"Graphs/{runtime_config}",
                        f"accuracy-{str(server)}", True)


def run_d2d(server: FedServer, options):
    training_times = []
    accuracy = []
    rounds = []
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
        training_time = e_time - s_time
        training_times.append(training_time)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
        fed_logger.info(f"Test Accuracy : {test_acc}")
        accuracy.append(test_acc)
        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))


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
    fed_server.stop_server()

