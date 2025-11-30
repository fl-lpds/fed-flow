import logging
import time
import warnings
import os

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
    """
    اجرای سناریوی offloading برای کلاینت.
    در ابتدای هر راند (قبل از train) یک بار check_and_migrate_per_round صدا می‌زنیم
    تا بر اساس فاصله، در صورت نیاز مهاجرت انجام شود.
    """
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r + 1))

        # --- Mobility: per-round distance check & (optional) migration ---
        edge_before = None
        if hasattr(client, "mobility_manager") and client.mobility_manager is not None:
            fed_logger.info(
                "[Mobility] ROUND %s: per-round distance check before training",
                r + 1,
            )
            client.mobility_manager.check_and_migrate_per_round()
            edge_before = client.mobility_manager.get_current_edge()

        fed_logger.info(
            "[Mobility] ROUND %s START -> current_edge=%s",
            r + 1,
            edge_before,
        )

        # --- FL pipeline ---
        fed_logger.info("receiving splitting info")
        client.gather_split_config()

        fed_logger.info("receiving global weights")
        client.gather_global_weights(NodeType.EDGE)

        fed_logger.info("test network")
        client.scatter_network_speed_to_edges()

        fed_logger.info("start training")
        client.start_offloading_train()

        fed_logger.info("sending local weights")

        # لاگ وضعیت موبیلیتی در انتهای راند
        if hasattr(client, "mobility_manager") and client.mobility_manager is not None:
            edge_after = client.mobility_manager.get_current_edge()
            fed_logger.info(
                "[Mobility] ROUND %s END -> current_edge=%s",
                r + 1,
                edge_after,
            )
        else:
            fed_logger.info(
                "[Mobility] ROUND %s END (no mobility manager attached)",
                r + 1,
            )

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
    fed_logger.info("[BOOT] fed_client_flow.run() ENTER at %s", __file__)
    fed_logger.info("start mode: " + str(options_ins.values()))
    index = config.index
    learning_rate = config.learning_rate

    fed_logger.info('Preparing Client')
    fed_logger.info('Preparing Data.')
    indices = list(range(N))
    part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
    train_loader = data_utils.get_trainloader(data_utils.get_trainset(), part_tr, 0)

    estimate_energy = options_ins.get("energy") == "True"
    mobility = options_ins.get('mobility')
    fed_logger.info(f"[Mobility] flag = {mobility}")
    d2d = options_ins.get('d2d')

    if estimate_energy:
        energy_estimation.init(os.getpid())

    ip = options_ins.get('ip')
    port = options_ins.get('port')
    cluster = options_ins.get('cluster')

    aggregator = create_aggregator(options_ins.get('aggregation'))

    client = FedClient(
        ip=ip,
        port=port,
        model_name=options_ins.get('model'),
        dataset=options_ins.get('dataset'),
        train_loader=train_loader,
        LR=learning_rate,
        cluster=cluster,
        aggregator=aggregator,
        neighbors=config.CURRENT_NODE_NEIGHBORS,
    )

    if mobility:
        fed_logger.info("[Mobility] Starting mobility simulation thread")
        start_mobility_simulation_thread(client)

        fed_logger.info("[Mobility] Discover edges")
        client.mobility_manager.discover_edges()

        fed_logger.info("[Mobility] Initialize to closest edge (or keep config edge)")
        client.mobility_manager.initialize_neighbors()

        fed_logger.info(
            "[Mobility] Current edge before training = %s",
            client.mobility_manager.get_current_edge(),
        )
        assert (
            client.mobility_manager.get_current_edge() is not None
        ), "No EDGE neighbor set! Did initialize_neighbors() run?"

        # دیگه monitor_and_migrate لازم نیست، چون مهاجرت را per-round انجام می‌دهیم
        # fed_logger.info("[Mobility] Start monitor and migrate")
        # client.mobility_manager.monitor_and_migrate()

    if d2d:
        run_d2d(client)
    else:
        run_client(client, learning_rate)

    time.sleep(10)
    client.stop_server()
