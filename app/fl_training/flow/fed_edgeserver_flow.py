import time

from app.config import config
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_edge_server import FedEdgeServer
from app.entity.http_communicator import HTTPCommunicator
from app.entity.node_type import NodeType
from app.util import graph_utils, model_utils


def run_decentralized(edge_server: FedEdgeServer, learning_rate, options: dict):
    edge_server.initialize(learning_rate)
    fed_logger.info(f"Split Config : {edge_server.split_layers}")
    edge_server.scatter_split_layers([NodeType.CLIENT])
    client_bw, edge_bw = [], []
    training_times = []
    rounds = []
    accuracy = []
    for r in range(config.R):
        config.current_round = r

        client_neighbors = edge_server.get_neighbors([NodeType.CLIENT])
        fed_logger.info("[Edge] ROUND %d -> connected_clients=%s", r + 1, [str(n) for n in client_neighbors])

        rounds.append(r)
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))

        fed_logger.info("sending global weights")
        edge_server.scatter_global_weights([NodeType.CLIENT])

        s_time = time.time()

        fed_logger.info("gathering neighbors network speed")
        edge_server.gather_neighbors_network_bandwidth()

        fed_logger.info("clustering")
        edge_server.clustering(options)

        fed_logger.info("getting neighbors bandwidth")
        neighbors_bandwidth = edge_server.get_neighbors_bandwidth()
        neighbors_bandwidth_by_type: dict[NodeType, list[float]] = {}
        for neighbor, bw in neighbors_bandwidth.items():
            neighbor_type = HTTPCommunicator.get_node_type(neighbor)
            neighbors_bandwidth_by_type.setdefault(neighbor_type, [])
            neighbors_bandwidth_by_type[neighbor_type].append(bw.bandwidth)

        client_bandwidths = neighbors_bandwidth_by_type.get(NodeType.CLIENT, [])
        has_client_neighbors = len(client_bandwidths) > 0

        # ثبت میانگین پهنای باند کلاینت و اج
        if has_client_neighbors:
            client_bw.append(sum(client_bandwidths) / len(client_bandwidths))
        else:
            client_bw.append(0)
            fed_logger.warning(
                "[Edge] ROUND %d: no client neighbors; skipping split/train for this edge (but WILL gossip)", r + 1,)

        if NodeType.EDGE in neighbors_bandwidth_by_type:
            edge_values = neighbors_bandwidth_by_type[NodeType.EDGE]
            edge_bw.append(sum(edge_values) / len(edge_values))
        else:
            edge_bw.append(0)

        # --- فقط اگر کلاینت داریم، split + train + aggregate ---
        if has_client_neighbors:
            fed_logger.info("splitting")
            edge_server.split(client_bandwidths, options)
            fed_logger.info(f"Split Config : {edge_server.split_layers}")
            edge_server.scatter_split_layers([NodeType.CLIENT])

            fed_logger.info("start training")
            edge_server.start_decentralized_training()

            fed_logger.info("receiving local weights")
            local_weights = edge_server.gather_local_weights()

            fed_logger.info("aggregating weights")
            edge_server.aggregate(local_weights)
        else:
            fed_logger.info(
                "[Edge] ROUND %d: no clients -> skipping split/train/aggregate, keeping current model", r + 1,)

        # --- مهم: در هر صورت، gossip انجام می‌شود ---
        fed_logger.info("start gossiping with neighbors")
        # added
        if not edge_server.is_active_this_round:
            pass
        else:
            edge_server.gossip_with_neighbors()

        e_time = time.time()

        # Recording each round training time, bandwidth and test_app accuracy
        training_time = e_time - s_time
        training_times.append(training_time)

        fed_logger.info("testing accuracy")
        test_acc = model_utils.test(edge_server.uninet, edge_server.testloader, edge_server.device,
                                    edge_server.criterion)
        fed_logger.info(f"Test Accuracy : {test_acc}")
        accuracy.append(test_acc)
        fed_logger.info('Round Finish')
        fed_logger.info('==> Round {:} End'.format(r + 1))
        fed_logger.info('==> Round Training Time: {:}'.format(training_time))
    graph_utils.report_results(edge_server, training_times, client_bw, accuracy, edge_bw)


def run_centralized(edge_server: FedEdgeServer, learning_rate):
    edge_server.gather_and_scatter_split_config()
    edge_server.initialize(learning_rate)
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('==> Round {:} Start'.format(r + 1))
        fed_logger.info("receiving and sending splitting info")
        edge_server.gather_and_scatter_split_config()
        fed_logger.info("receiving and sending global weights")
        edge_server.gather_and_scatter_global_weight()
        fed_logger.info("test clients network")
        edge_server.gather_neighbors_network_bandwidth()
        fed_logger.info("start training")
        edge_server.start_centralized_training()
        fed_logger.info('==> Round {:} End'.format(r + 1))


def run(options_ins):
    LR = config.learning_rate
    fed_logger.info('Preparing Sever.')
    offload = options_ins.get('offload')
    decentralized = options_ins.get('decentralized')
    aggregator = create_aggregator(options_ins.get('aggregation'))
    edge_server = FedEdgeServer(options_ins.get('ip'), options_ins.get('port'),
                                options_ins.get('model'),
                                options_ins.get('dataset'), offload, aggregator,
                                config.CURRENT_NODE_NEIGHBORS)
    fed_logger.info("neighbors: " + str(config.CURRENT_NODE_NEIGHBORS))

    fed_logger.info("start mode: " + str(options_ins.values()))
    if decentralized:
        run_decentralized(edge_server, LR, options_ins)
    else:
        run_centralized(edge_server, LR)
    time.sleep(10)
    edge_server.stop_server()
