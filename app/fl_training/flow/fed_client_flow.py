import logging
import time
import warnings

from app.config import config
from app.config.config import *
from app.config.logger import fed_logger
from app.entity.aggregators.factory import create_aggregator
from app.entity.fed_client import FedClient
from app.entity.node_type import NodeType
from app.util import data_utils, energy_estimation
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
    dataset = data_utils.get_trainset()
    if True:
        part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
        train_loader = data_utils.get_trainloader(dataset, part_tr, 0)
    else:
        # Non-IID (20% non-IID + 80% IID):
        total_samples = len(dataset)

        # 1. Compute the global non-IID pool (20% of the data).
        non_iid_total = int(total_samples * 0.8)
        # Sort all indices by label to create a skewed (non-IID) ordering.
        labels = [dataset[i][1] for i in range(total_samples)]
        sorted_indices = sorted(range(total_samples), key=lambda i: labels[i])
        # Partition the non-IID pool evenly among clients.
        non_iid_per_client = non_iid_total // config.K
        non_iid_start = non_iid_per_client * index
        non_iid_end = non_iid_start + non_iid_per_client
        client_non_iid_indices = sorted_indices[non_iid_start:non_iid_end]

        # 2. Compute the IID pool (remaining 80% of the data).
        # Use the original order of indices for IID data.
        global_non_iid_set = set(sorted_indices[:non_iid_total])
        iid_pool = [i for i in range(total_samples) if i not in global_non_iid_set]
        # Partition the IID pool evenly among clients.
        iid_per_client = len(iid_pool) // config.K
        iid_start = iid_per_client * index
        iid_end = iid_start + iid_per_client
        client_iid_indices = iid_pool[iid_start:iid_end]

        # 3. Combine the two parts for the client.
        client_indices = client_non_iid_indices + client_iid_indices
        train_loader = data_utils.get_trainloader(dataset, client_indices, 0)

    # client_labels = [dataset[i][1] for i in client_indices]
    # label_counts = Counter(client_labels)
    #
    # print(f"\nClient {index} - Total samples: {len(client_indices)}")
    # print("Label distribution:")
    # for label, count in sorted(label_counts.items()):
    #     print(f"  Label {label}: {count} samples")

    estimate_energy = options_ins.get("energy") == "True"
    mobility = options_ins.get('mobility')
    d2d = options_ins.get('d2d')

    if estimate_energy:
        energy_estimation.init(os.getpid())

    ip = options_ins.get('ip')
    port = options_ins.get('port')
    cluster = options_ins.get('cluster')

    aggregator = create_aggregator(options_ins.get('aggregation'))

    client = FedClient(ip=ip, port=port, model_name=options_ins.get('model'),
                       dataset=options_ins.get('dataset'), train_loader=train_loader, LR=learning_rate,
                       cluster=cluster, aggregator=aggregator, neighbors=config.CURRENT_NODE_NEIGHBORS)
    if mobility:
        start_mobility_simulation_thread(client)
        # client.mobility_manager.discover_edges()
        # client.mobility_manager.monitor_and_migrate()
    if d2d:
        run_d2d(client)

    else:
        run_client(client, learning_rate)
    time.sleep(10)
    # client.stop_server()
