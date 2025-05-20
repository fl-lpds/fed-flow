import logging
import multiprocessing
import os
import sys
import time
import warnings

sys.path.append('../../../')
from app.entity.client import Client
from app.config import config
from app.config.config import *
from app.util import data_utils, energy_estimation
from app.config.logger import fed_logger
from app.entity.interface.fed_client_interface import FedClientInterface
from colorama import Fore
import random

warnings.filterwarnings('ignore')
logging.getLogger("requests").setLevel(logging.WARNING)


def run_edge_based(client: FedClientInterface, LR):
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B

    fed_logger.info('Sending power usage to edge.')
    client.send_power_to_edge()
    fed_logger.info(Fore.LIGHTGREEN_EX + f"TOTAL ROUND: {config.R}")

    for r in range(config.R):

        simnet_BW = config.CLIENTS_BANDWIDTH[config.index]
        energy_estimation.set_simnet(simnet_BW)

        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))

        st = time.time()
        proces_time_start = time.process_time()

        if not client.simnet:
            fed_logger.info("test network")
            client.edge_test_network()
        else:
            fed_logger.info("Sending BW to edge")
            client.send_simnet_bw(simnet_BW)

        fed_logger.info("receiving splitting info")
        client.get_split_layers_config_from_edge()

        fed_logger.info("initializing client")
        energy_estimation.computation_start()
        client.initialize(client.split_layers, LR, simnetbw=simnet_BW)
        energy_estimation.computation_end()

        fed_logger.info("receiving global weights")
        client.get_edge_global_weights()

        fed_logger.info("start training")
        client.edge_offloading_train()

        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.send_local_weights_to_edge()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))

        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregation')

        et = time.time()
        process_time_end = time.process_time()

        tt = 0
        if client.simnet:
            transmission_time = float(energy_estimation.get_transmission_time())
            computation_time = client.computational_time
            tt = transmission_time + computation_time
            fed_logger.info(Fore.MAGENTA + f"Client SIMNET Total time: {tt}")
        else:
            computation_time = process_time_end - proces_time_start
            tt = computation_time
        fed_logger.info(Fore.MAGENTA + f"Client Wall-Time: {et - st}")
        fed_logger.info(Fore.MAGENTA + f"Client Process-Time: {process_time_end - proces_time_start}")

        utilization = float(energy_estimation.get_utilization())
        comp_energy, comm_energy = energy_estimation.energy(computation_time)
        remaining_energy = float(energy_estimation.remaining_energy())

        fed_logger.info(
            Fore.MAGENTA + f"Comp Energy, Comm Energy TT, Remaining-energy, utilization: {comp_energy}, {comm_energy}, {tt},"
                           f" {remaining_energy}, {utilization}")
        fed_logger.info("Sending Comp Energy, Comm Energy, TT, Remaining-energy, utilization to edge.")
        client.energy_tt(remaining_energy, comp_energy, comm_energy, tt, utilization)
        client.e_next_round_attendance(remaining_energy)

        if r > 49:
            LR = config.LR * 0.1


def run_no_offload_edge(client: FedClientInterface, LR):
    mx: int = int((N / K) * (index + 1))
    mn: int = int((N / K) * index)
    data_size = mx - mn
    batch_num = data_size / config.B
    client.initialize(split_layer=config.split_layer, LR=LR, simnetbw=None)
    for r in range(config.R):
        config.current_round = r
        fed_logger.info("receiving global weights")
        client.get_edge_global_weights()
        st = time.time()
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("sending local weights")
        energy_estimation.start_transmission()
        msg = client.send_local_weights_to_edge()
        energy_estimation.end_transmission(data_utils.sizeofmessage(msg))
        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1
        et = time.time()
        tt = et - st
        comp_energy, comm_energy = energy_estimation.energy()
        # energy /= batch_num
        fed_logger.info(Fore.CYAN + f"Comp energy, Comm energy, tt : {comp_energy}, {comm_energy}, {tt}")
        remaining_energy = float(energy_estimation.remaining_energy())
        fed_logger.info(Fore.MAGENTA + f"remaining energy: {remaining_energy}")
        client.e_next_round_attendance(remaining_energy)


def run_no_edge_offload(client: FedClientInterface, LR):
    fed_logger.info('Sending power usage to server.')
    client.send_power_to_edge()

    client.split_layers = config.split_layer

    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))

        st = time.time()
        proces_time_start = time.process_time()

        base_bw = 10_000_000
        random_number = random.uniform(1, 100)
        if random_number < 80:
            simnet_BW = base_bw
        else:
            simnet_BW = base_bw * random.randint(3, 10)

        if not client.simnet:
            fed_logger.info("test network")
            client.edge_test_network()
        else:
            fed_logger.info("Sending BW to server")
            client.send_simnet_bw(simnet_BW)

        fed_logger.info("receiving splitting info")
        client.get_split_layers_config()

        fed_logger.info("receiving global weights")
        client.get_server_global_weights()

        fed_logger.info("initializing client")
        client.initialize(client.split_layers, LR, simnet_BW)

        fed_logger.info("start training")
        client.offloading_train()

        fed_logger.info("sending local weights")
        msg = client.send_local_weights_to_server()

        et = time.time()
        process_time_end = time.process_time()

        if client.simnet:
            transmission_time = float(energy_estimation.get_transmission_time())
            computation_time = client.computational_time
            tt = transmission_time + computation_time
            fed_logger.info(Fore.MAGENTA + f"Client SIMNET Total time: {tt}")
        else:
            computation_time = process_time_end - proces_time_start
            tt = computation_time

        fed_logger.info('ROUND: {} END'.format(r))
        fed_logger.info(Fore.MAGENTA + f"Client Wall-Time: {et - st}")
        fed_logger.info(Fore.MAGENTA + f"Client Process-Time: {process_time_end - proces_time_start}")

        utilization = float(energy_estimation.get_utilization())
        comp_energy, comm_energy = energy_estimation.energy(computation_time)
        remaining_energy = float(energy_estimation.remaining_energy())

        fed_logger.info(
            Fore.MAGENTA + f"Comp Energy, Comm Energy TT, Remaining-energy, utilization: {comp_energy}, {comm_energy}, {tt},"
                           f" {remaining_energy}, {utilization}")
        fed_logger.info("Sending Comp Energy, Comm Energy, TT, Remaining-energy, utilization to edge.")
        client.energy_tt(remaining_energy, comp_energy, comm_energy, tt, utilization)

        client.next_round_attendance(remaining_energy)


def run_no_edge(client: FedClientInterface, LR):
    client.initialize(split_layer=config.split_layer, LR=LR, simnetbw=None)
    for r in range(config.R):
        config.current_round = r
        fed_logger.info('====================================>')
        fed_logger.info('ROUND: {} START'.format(r))
        fed_logger.info("receiving global weights")

        st = time.time()

        client.get_server_global_weights()
        fed_logger.info("start training")
        client.no_offloading_train()
        fed_logger.info("sending local weights")
        client.send_local_weights_to_server()

        tt = time.time()
        fed_logger.info('ROUND: {} END'.format(r))

        fed_logger.info('==> Waiting for aggregration')
        if r > 49:
            LR = config.LR * 0.1

        energy = float(energy_estimation.energy())
        # energy /= batch_num
        fed_logger.info(Fore.CYAN + f"Energy_tt : {energy}, {tt}")
        remaining_energy = float(energy_estimation.remaining_energy())
        fed_logger.info(Fore.MAGENTA + f"remaining energy: {remaining_energy}")
        client.next_round_attendance(remaining_energy)


def run(options_ins):
    fed_logger.info("start mode: " + str(options_ins.values()))
    index = config.index
    datalen = config.N / config.K
    LR = config.LR

    config.R = 10
    if (options_ins.get('splitting') == 'edge_based_heuristic' or
            options_ins.get('splitting') == 'edge_rl_splitting' or
            options_ins.get('splitting') == 'random_splitting'):
        config.R = 100

    fed_logger.info('Preparing Client')
    fed_logger.info('Preparing Data.')
    cpu_count = multiprocessing.cpu_count()

    dataset = data_utils.get_trainset()

    indices = list(range(N))
    if options_ins.get("iid"):
        part_tr = indices[int((N / K) * index): int((N / K) * (index + 1))]
        trainloader = data_utils.get_trainloader(dataset, part_tr, cpu_count)
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        subset_size = len(dataset) // config.K
        start_idx = subset_size * index
        end_idx = start_idx + subset_size
        client_indices = sorted_indices[start_idx:end_idx]
        trainloader = data_utils.get_trainloader(dataset, client_indices, cpu_count)

    estimate_energy = options_ins.get("energy") == "True"
    simnet = options_ins.get("simulatebandwidth") == "True"
    if estimate_energy:
        energy_estimation.init(os.getpid())

    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    if edge_based and offload:
        client_ins = Client(server=config.CLIENT_MAP[config.CLIENTS_INDEX[index]], datalen=datalen,
                            model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            simnet=simnet
                            )
        run_edge_based(client_ins, LR)
    elif edge_based and not offload:
        client_ins = Client(server=config.CLIENT_MAP[config.CLIENTS_INDEX[index]],
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            simnet=simnet
                            )
        run_no_offload_edge(client_ins, LR)
    elif offload:
        client_ins = Client(server=config.SERVER_ADDR,
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            simnet=simnet
                            )
        run_no_edge_offload(client_ins, LR)
    else:
        client_ins = Client(server=config.SERVER_ADDR,
                            datalen=datalen, model_name=options_ins.get('model'),
                            dataset=options_ins.get('dataset'), train_loader=trainloader, LR=LR, edge_based=edge_based,
                            simnet=simnet
                            )
        run_no_edge(client_ins, LR)
