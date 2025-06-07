import os
import socket
import sys
import threading
import time

from colorama import Fore

from app.util import energy_estimation

sys.path.append('../../../')
from app.config import config
from app.config.logger import fed_logger
from app.entity.edge_server import FedEdgeServer
from app.entity.interface.fed_edgeserver_interface import FedEdgeServerInterface
from torch.multiprocessing import Process, Manager


def run_offload(server: FedEdgeServerInterface, LR, options):
    server.initialize(config.split_layer, LR, config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]], simnetbw=10)

    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]

    fed_logger.info('Getting power usage from clients and sending to server.')
    server.get_power_and_send_to_server()
    fed_logger.info(Fore.LIGHTGREEN_EX + f"TOTAL ROUND: {config.R}")

    for r in range(config.R):
        fed_logger.info(Fore.LIGHTRED_EX + f" left clients {client_ips}")
        if len(config.CLIENTS_LIST) > 0:
            simnetbw = 150_000_000  # 150 Mbps

            config.current_round = r
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(config.current_round))
            if not server.simnet:
                fed_logger.info("test clients network")
                server.test_client_network(client_ips)
                fed_logger.info("sending clients network")
                server.client_network()
                fed_logger.info("test server network")
                server.test_server_network()
            else:
                fed_logger.info("receiving clients simnet bw")
                server.get_simnet_client_network()
                fed_logger.info("sending clients simnet bw to server")
                server.client_network()

            fed_logger.info("receiving and sending splitting info")
            server.get_split_layers_config(client_ips)
            fed_logger.info(Fore.MAGENTA + f"Nice Values: {server.nice_value}")

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR, client_ips, simnetbw=simnetbw)

            fed_logger.info("receiving global weights")
            server.global_weights(client_ips)

            fed_logger.info("start training")
            start_training = time.time()
            for clientips in config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]:
                server.computation_time_of_each_client[clientips] = 0
                server.process_wall_time[clientips] = 0
            with Manager() as manager:
                shared_data = manager.dict()
                shared_data['computation_time_of_each_client'] = server.communication_time_of_each_client
                shared_data['current_round'] = config.current_round
                shared_data['process_wall_time'] = server.process_wall_time
                shared_data['edge_index'] = config.index
                fed_logger.info(Fore.RED + f"shared data: {shared_data}")
                fed_logger.info(Fore.RED + f"Edge Index: {config.index}")

                processes = {}
                for i in range(len(client_ips)):
                    processes[client_ips[i]] = Process(target=server.thread_offload_training,
                                                       args=(client_ips[i], shared_data,),
                                                       name=client_ips[i])
                for i in range(len(client_ips)):
                    processes[client_ips[i]].start()
                    if options.get('splitting') == 'edge_based_heuristic':
                        os.system(f"renice -n {server.nice_value[client_ips[i]]} -p {processes[client_ips[i]].pid}")

                for process in processes.values():
                    process.join()
                server.computation_time_of_each_client = shared_data['computation_time_of_each_client']
                server.process_wall_time = shared_data['process_wall_time']

            total_training_time = time.time() - start_training
            fed_logger.info(Fore.RED + f"Total training time(Wall-time): {total_training_time}")
            fed_logger.info(Fore.RED + f"Each Process computation Wall Time: {server.process_wall_time}")
            server.total_computation_time_on_edge = max(server.process_wall_time.values())
            fed_logger.info(
                Fore.RED + f"each client communication time: {server.communication_time_of_each_client}")
            fed_logger.info(Fore.RED + f"computation time of each client: {server.computation_time_of_each_client}")
            fed_logger.info(Fore.RED + f"Total computation time: {server.total_computation_time_on_edge}")

            fed_logger.info(
                "receiving Comp Energy, Comm Energy, TT, Remaining-energy, utilization from clients and sending to server")
            server.energy(client_ips)
            fed_logger.info(Fore.GREEN + f"Clients' attributes received")
            if r > 49:
                LR = config.LR * 0.1

            server.client_attendance(client_ips)
            client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def run_no_offload(server: FedEdgeServerInterface, LR):
    server.initialize(config.split_layer, LR, config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]])
    res = {}
    res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]
    for r in range(config.R):
        if len(config.CLIENTS_LIST) > 0:
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
            server.client_attendance(client_ips)
            client_ips = config.EDGE_MAP[config.EDGE_SERVER_CONFIG[config.index]]
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    offload = options_ins.get('offload')
    simnet = options_ins.get("simulatebandwidth") == "True"
    estimate_energy = options_ins.get("energy") == "True"

    config.R = 5
    if (options_ins.get('splitting') == 'edge_based_heuristic' or
            options_ins.get('splitting') == 'edge_rl_splitting' or
            options_ins.get('splitting') == 'random_splitting'):
        config.R = 100

    if estimate_energy:
        energy_estimation.init(os.getpid())
    if offload:
        edge_server_ins = FedEdgeServer(
            options_ins.get('model'),
            options_ins.get('dataset'), offload=offload, simnet=simnet)
        fed_logger.info("start mode: " + str(options_ins.values()))
        run_offload(edge_server_ins, LR, options_ins)
    else:
        edge_server_ins = FedEdgeServer(options_ins.get('model'),
                                        options_ins.get('dataset'), offload=offload)
        fed_logger.info("start mode: " + str(options_ins.values()))
        run_no_offload(edge_server_ins, LR)
    # msg = edge_server_ins.recv_msg(config.SERVER_ADDR, message_utils.finish)
    # edge_server_ins.scatter(msg)
