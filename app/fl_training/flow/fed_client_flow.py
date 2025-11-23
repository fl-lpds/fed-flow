import logging
import time
import warnings

from app.config import config
from app.config.config import *
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_client import FedClient
from app.entity.node_type import NodeType
from app.util import data_utils, energy_estimation, model_utils
from app.util.mobility_data_utils import start_mobility_simulation_thread

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_client(client: FedClient, learning_rate):
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))
        fed_logger.info("receiving splitting info")
        client.gather_split_config()
        fed_logger.info("receiving global weights")
        client.gather_global_weights(NodeType.EDGE)
        fed_logger.info("test network")
        client.scatter_network_speed_to_edges()
        fed_logger.info("start training")
        client.start_offloading_train()
        fed_logger.info("sending local weights")
        client.scatter_local_weights()
        fed_logger.info('ROUND: {} END'.format(r + 1))


def run_d2d(client: FedClient):
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))
        fed_logger.info("receiving global weights")
        client.gather_global_weights(NodeType.SERVER)
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("gossip with neighbors")
        client.gossip_with_neighbors()
        fed_logger.info("sending local weights")
        client.scatter_random_local_weights()
        fed_logger.info('ROUND: {} END'.format(r + 1))


def run(options_ins):
    fed_logger.info("start mode: " + str(options_ins.values()))
    index = config.index
    learning_rate = config.learning_rate

    fed_logger.info('Preparing Client')
    fed_logger.info('Preparing Data.')
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
    train_loader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, 0)
    fed_logger.info('Data preparation completed. Training samples: {}'.format(len(part_tr)))

    estimate_energy = options_ins.get("energy") == "True"
    mobility = options_ins.get('mobility')
    d2d = options_ins.get('d2d')
    fed_logger.info('Configuration parsed - Energy: {}, Mobility: {}, D2D: {}'.format(estimate_energy, mobility, d2d))

    if estimate_energy:
        energy_estimation.init(os.getpid())
        fed_logger.info('Energy estimation initialized')

    ip = options_ins.get('ip')
    port = options_ins.get('port')
    cluster = options_ins.get('cluster')
    fed_logger.info('Network configuration - IP: {}, Port: {}, Cluster: {}'.format(ip, port, cluster))

    aggregator = create_aggregator(options_ins.get('aggregation'))
    fed_logger.info('Aggregator created: {}'.format(options_ins.get('aggregation')))

    client = FedClient(ip=ip, port=port, model_name=options_ins.get('model'),
                       dataset=options_ins.get('dataset'), train_loader=train_loader, LR=learning_rate,
                       cluster=cluster, aggregator=aggregator, neighbors=config.CURRENT_NODE_NEIGHBORS)
    fed_logger.info('FedClient initialized - Model: {}, Dataset: {}'.format(options_ins.get('model'), options_ins.get('dataset')))
    
    if mobility:
        start_mobility_simulation_thread(client)
        fed_logger.info('Mobility simulation thread started')
        # client.mobility_manager.discover_edges()
        # client.mobility_manager.monitor_and_migrate()
    
    fed_logger.info('Starting federated learning training')
    if d2d:
        fed_logger.info('Running in D2D mode')
        run_d2d(client)
    else:
        fed_logger.info('Running in client mode')
        run_client(client, learning_rate)
    
    fed_logger.info('Training completed. Cleaning up...')
    time.sleep(10)
    client.stop_server()
    fed_logger.info('Client server stopped')
