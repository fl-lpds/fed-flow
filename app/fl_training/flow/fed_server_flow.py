import pickle
import socket
import sys
import time

from colorama import Fore

sys.path.append('../../../')
from app.config import config
from app.util import model_utils
from app.entity.server import FedServer
from app.config.logger import fed_logger
from app.entity.interface.fed_server_interface import FedServerInterface
from app.util import rl_utils
from app.util import energy_estimation

import matplotlib.pyplot as plt
import random
import os
import csv
import json
import numpy as np


def run_edge_based_no_offload(server: FedServerInterface, LR, options):
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        fed_logger.info(Fore.LIGHTBLUE_EX + f"left clients in server{config.K}")
        if config.K > 0:
            config.current_round = r
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            fed_logger.info("sending global weights")
            server.edge_offloading_global_weights()
            s_time = time.time()
            fed_logger.info("clustering")
            server.cluster(options)
            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            e_time = time.time()

            # Recording each round training time, bandwidth and test_app accuracy
            training_time = e_time - s_time
            fed_logger.info("testing accuracy")
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
            server.e_client_attendance(config.CLIENTS_LIST)
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def run_edge_based_offload(server: FedServerInterface, LR, options):
    avgEnergy, tt, simnet_tt, splitting_list = [], [], [], []
    clientBW, edge_server_BW = {}, {}
    clientRemainingEnergy = {}
    clientConsumedEnergy = {}
    comp_time_of_each_client_on_edge = {}
    comp_time_of_each_client_on_server = {}
    clientCompEnergy = {}
    clientCommEnergy = {}
    clientCompTime = {}
    clientCommTime = {}
    clientUtilization = {}
    clientTT = {}
    comp_time_of_each_layer_on_client = {client: [] for client in config.CLIENTS_LIST}

    edge_server_comm_time_list = {}

    flop_of_each_client_on_edge_list = {}
    flop_of_each_client_on_server_list = {}
    flop_on_each_edge = {}
    time_on_each_edge = {}
    flop_of_each_edge_on_server = {}
    flop_on_server = []
    time_on_server = []

    for client in config.CLIENTS_LIST:
        clientBW[client] = []
        clientRemainingEnergy[client] = []
        clientConsumedEnergy[client] = []
        clientUtilization[client] = []
        clientCompEnergy[client] = []
        clientCommEnergy[client] = []
        clientCompTime[client] = []
        clientCommTime[client] = []
        clientTT[client] = []
        comp_time_of_each_client_on_edge[client] = []
        comp_time_of_each_client_on_server[client] = []
        flop_of_each_client_on_edge_list[client] = []
        flop_of_each_client_on_server_list[client] = []
        edge_server_comm_time_list[client] = []

    for edge in config.EDGE_SERVER_LIST:
        edge_server_BW[edge] = []
        flop_on_each_edge[edge] = []
        time_on_each_edge[edge] = []
        flop_of_each_edge_on_server[edge] = []

    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    fed_logger.info(f"OPTION: {options}")

    fed_logger.info(Fore.RED + f"PID of process: {os.getpid()}")
    if os.getpid() > 0:
        fed_logger.info(Fore.MAGENTA + f"Calculation of Each layer's activation and gradient size started on server")
        server.calculate_each_layer_activation_gradiant_size()
        server.remove_non_pickleable()
    fed_logger.info(Fore.MAGENTA + f"Calculation of Each layer's FLOP started on server")
    server.calculate_each_layer_FLOP()

    flops_of_each_layer = server.model_flops_per_layer
    flops_of_each_layer = {key: flops_of_each_layer[key] for key in sorted(flops_of_each_layer)}
    flops_of_each_layer = list(flops_of_each_layer.values())

    test_load_on_edges_and_server = [[[config.model_len - 1, config.model_len - 1] for _ in range(config.K)]]

    # # low load on edge 90% of each model on client
    # op1, op2 = rl_utils.actionToLayer([0.9, 1.0], flops_of_each_layer)
    # test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])
    #
    # # medium load on edge 50% of each model on client
    # op1, op2 = rl_utils.actionToLayer([0.5, 1.0], flops_of_each_layer)
    # test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])
    #
    # # high load on edge 100% of each model on edge
    # op1, op2 = rl_utils.actionToLayer([0.0, 1.0], flops_of_each_layer)
    # test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])
    #
    # # low load on server 90% of each model on client
    # op1, op2 = rl_utils.actionToLayer([0.9, 0.0], flops_of_each_layer)
    # test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])
    #
    # # medium load on server 50% of each model on client
    # op1, op2 = rl_utils.actionToLayer([0.5, 0.0], flops_of_each_layer)
    # test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])
    #
    # # high load on server 100% of each model on edge
    # op1, op2 = rl_utils.actionToLayer([0.0, 0.0], flops_of_each_layer)
    # test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    for layer in range(config.model_len - 1):
        test_load_on_edges_and_server.append(
            [[layer, config.model_len - 1] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    fed_logger.info(Fore.RED + f"Load testing: {test_load_on_edges_and_server}")

    fed_logger.info('Getting power usage from edge servers')
    server.get_power_of_client()

    simulated_edge_server_bw = {}
    if server.simnet:
        simulated_edge_server_bw = np.load(f"/fed-flow/app/config/edge_server_bw.npz")

    for r in range(config.R):
        fed_logger.debug(Fore.LIGHTBLUE_EX + f"number of final K: {config.K}")
        if config.K > 0:

            config.current_round = r
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            process_time_start = time.process_time()

            if not server.simnet:
                fed_logger.info("receiving client network info")
                server.client_network(config.EDGE_SERVER_LIST)
                fed_logger.info("test edge servers network")
                server.test_network(config.EDGE_SERVER_LIST)
            else:
                # setting BW between each edge and sever
                for edge in config.EDGE_SERVER_LIST:
                    server.edge_bandwidth[edge] = simulated_edge_server_bw[edge][r]
                fed_logger.info("receiving client simnet network info")
                server.client_network(config.EDGE_SERVER_LIST)

            for client in server.client_bandwidth.keys():
                clientBW[client].append(server.client_bandwidth[client])
            for edge_server in server.edge_bandwidth.keys():
                edge_server_BW[edge_server].append(server.edge_bandwidth[edge_server])

            # fed_logger.info("preparing state...")
            # server.offloading = server.get_offloading(server.split_layers)

            fed_logger.info("clustering")
            server.cluster(options)

            fed_logger.info("getting state")
            offloading = server.split_layers

            fed_logger.info("creating state")
            for client, index in config.CLIENTS_CONFIG.items():
                if client not in config.CLIENTS_LIST:
                    server.split_layers[index] = [config.model_len, config.model_len]
            state = server.edge_based_state()

            if r < len(test_load_on_edges_and_server) and options.get('splitting') == 'edge_based_heuristic':
                server.split_layers = test_load_on_edges_and_server[r]
                server.server_nice_value = {client: 0 for client in config.CLIENTS_CONFIG.keys()}
                server.edge_nice_value = {client: 0 for client in config.CLIENTS_CONFIG.keys()}
                splitting_list.append(server.split_layers)
            else:
                fed_logger.info("splitting")
                splitTime_start = time.time()
                server.split(state, options)
                splittingTime = time.time() - splitTime_start
                fed_logger.info(Fore.MAGENTA + f"Splitting Time : {splittingTime}")
                fed_logger.info(Fore.MAGENTA + f"Action : {server.split_layers}")
                splitting_list.append(server.split_layers)
                fed_logger.info(Fore.MAGENTA + f"SERVER Nice Value : {server.server_nice_value}")
                fed_logger.info(Fore.MAGENTA + f"EDGE Nice Value : {server.edge_nice_value}")

            fed_logger.info("Scattering splitting info to edges.")
            server.send_split_layers_config()

            if r > 49:
                LR = config.LR * 0.1

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            fed_logger.info("sending global weights")
            server.edge_offloading_global_weights()
            # fed_logger.info('==> Reinitialization Finish')

            fed_logger.info("start training")
            start_training_time = time.time()
            if options.get('splitting') == 'edge_based_heuristic':
                server.edge_offloading_train(config.CLIENTS_LIST, hasPriority=True)
            else:
                server.edge_offloading_train(config.CLIENTS_LIST, hasPriority=False)
            total_training_time = time.time() - start_training_time

            e_time = time.time()
            process_time_end = time.process_time()
            training_time = e_time - s_time
            total_process_time = process_time_end - process_time_start

            server.total_computation_time = max(server.process_wall_time.values())
            fed_logger.info(Fore.RED + f"Total time: {total_training_time}")
            fed_logger.info(Fore.RED + f"Total computation wall-time: {server.total_computation_time}")
            fed_logger.info(Fore.RED + f"Each client computation time: {server.computation_time_of_each_client}")
            fed_logger.info(Fore.RED + f"Each process computation wall-time: {server.process_wall_time}")

            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            aggregation_start_time = time.time()
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            aggregation_end_time = time.time()
            aggregation_time = aggregation_end_time - aggregation_start_time

            fed_logger.info(Fore.GREEN + f"Receiving Clients' Attribute")
            energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)

            server.e_client_attendance(config.CLIENTS_LIST)


            fed_logger.info(f"computation time of each client on server[wall-time]: {server.process_wall_time}")
            fed_logger.info(
                f"computation time of each client on edge[wall-time]: {server.computation_time_of_each_client_on_edges}")
            fed_logger.info(
                f"Total computation time on each edge[ MAX(wall-time) ]: {server.total_computation_time_of_each_edge}")
            fed_logger.info(f"Transmission time of each client on server: {server.client_training_transmissionTime}")
            fed_logger.info(f"Aggregation Time Simnet bw : {aggregation_time}")
            server_sequential_transmission_time = float(energy_estimation.get_transmission_time())
            fed_logger.info(f"Server Sequential Transmission time: {server_sequential_transmission_time}")
            energy_estimation.reset_transmission_time()

            fed_logger.info(Fore.GREEN + f"==========================================================")
            fed_logger.info(
                Fore.GREEN + f"Clients  CompE  CommE  CompTT CommTT  TotalTT  RemaiE  Utiliz  CompOnEdge  CompOnServer  TotalCompOnEdge")
            energy = 0
            for client in energy_tt_list.keys():
                clientCompEnergy[client].append(energy_tt_list[client][0])
                clientCommEnergy[client].append(energy_tt_list[client][1])
                compTime = energy_tt_list[client][0] / (
                        energy_tt_list[client][4] * server.power_usage_of_client[client][0])
                commTime = energy_tt_list[client][1] / server.power_usage_of_client[client][1]
                clientCompTime[client].append(compTime)
                clientCommTime[client].append(commTime)
                clientConsumedEnergy[client].append(energy_tt_list[client][0] + energy_tt_list[client][1])
                clientRemainingEnergy[client].append(energy_tt_list[client][3])
                clientUtilization[client].append(energy_tt_list[client][4])
                comp_time_of_each_client_on_edge[client].append(server.computation_time_of_each_client_on_edges[client])
                comp_time_of_each_client_on_server[client].append(server.process_wall_time[client])
                energy += (energy_tt_list[client][0] + energy_tt_list[client][1])

                fed_logger.info(
                    Fore.GREEN + f"{client}  {energy_tt_list[client][0]:.3f}  {energy_tt_list[client][1]:.3f}  {compTime:.3f}  {commTime:.3f}  {energy_tt_list[client][2]:.3f}  {energy_tt_list[client][3]:.3f}  {energy_tt_list[client][4]:.3f}  {energy_tt_list[client][5]:.3f}  {server.process_wall_time[client]:.3f}  {energy_tt_list[client][6]:.3f}")

            if config.K != 0:
                avgEnergy.append(energy / int(config.K))
            else:
                avgEnergy.append(0)

            tt.append(training_time)

            if server.simnet:
                simulatedTT, total_tt_of_each_client = server.simnetTrainingTimeCalculation(aggregation_time,
                                                                                            server_sequential_transmission_time,
                                                                                            energy_tt_list)
                for client in energy_tt_list.keys():
                    clientTT[client].append(total_tt_of_each_client[client])
                simnet_tt.append(simulatedTT)
                if simulatedTT < server.best_tt_splitting_found['time']:
                    server.best_tt_splitting_found['time'] = simulatedTT
                    server.best_tt_splitting_found['splitting'] = server.split_layers

                fed_logger.info(f"Simulated training time: {simulatedTT}")
            else:
                fed_logger.info(f"Training time: {training_time}")

            fed_logger.info(f"Process-Time: {total_process_time}")

            server_flop, each_edge_flop, flop_of_each_edges_on_server, flop_of_each_client_on_edge, flop_of_each_client_on_server = server.getFlopsOnEdgeAndServer()
            flop_on_server.append(server_flop)
            time_on_server.append(server.total_computation_time)
            for edge in config.EDGE_SERVER_LIST:
                flop_on_each_edge[edge].append(each_edge_flop[edge])
                time_on_each_edge[edge].append(server.total_computation_time_of_each_edge[edge])
                flop_of_each_edge_on_server[edge].append(flop_of_each_edges_on_server[edge])
            for client in flop_of_each_client_on_edge.keys():
                flop_of_each_client_on_edge_list[client].append(flop_of_each_client_on_edge[client])
                flop_of_each_client_on_server_list[client].append(flop_of_each_client_on_server[client])
                edge_server_comm_time_list[client].append(server.client_training_transmissionTime[client])

            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            # with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
            #     pickle.dump(res, f)

            # fed_logger.info("testing accuracy")
            # test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            # res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
            # fed_logger.info(Fore.MAGENTA + f"Flop on server: {flop_on_server}")
            plot_graph(tt, simnet_tt, avgEnergy, clientConsumedEnergy, clientCompEnergy, clientCommEnergy, clientTT,
                       clientRemainingEnergy, clientBW, edge_server_BW, clientUtilization, res['test_acc_record'],
                       flop_on_each_edge, time_on_each_edge, flop_of_each_edge_on_server, flop_on_server,
                       time_on_server, clientCompTime, clientCommTime, server.approximated_energy_of_actions,
                       server.approximated_tt_of_actions, comp_time_of_each_client_on_edge,
                       comp_time_of_each_client_on_server, flop_of_each_client_on_edge_list,
                       flop_of_each_client_on_server_list, edge_server_comm_time_list, splitting=splitting_list,
                       splitting_method=options.get('splitting'))
        else:
            break

    fed_logger.info(f"{socket.gethostname()} quit")


def run_no_edge_offload(server: FedServerInterface, LR, options):
    avgEnergy, tt, simnet_tt = [], [], []
    clientBW = {}
    clientRemainingEnergy = {}
    clientConsumedEnergy = {}
    comp_time_of_each_client_on_server = {}
    clientCompEnergy = {}
    clientCommEnergy = {}
    clientCompTime = {}
    clientCommTime = {}
    clientUtilization = {}
    clientTT = {}

    flop_on_server = []
    time_on_server = []

    for client in config.CLIENTS_LIST:
        clientBW[client] = []
        clientRemainingEnergy[client] = []
        clientConsumedEnergy[client] = []
        clientUtilization[client] = []
        clientCompEnergy[client] = []
        clientCommEnergy[client] = []
        clientCompTime[client] = []
        clientCommTime[client] = []
        clientTT[client] = []
        comp_time_of_each_client_on_server[client] = []

    server.initialize(config.split_layer, LR)
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    fed_logger.info('Getting power usage from clients')
    server.get_power_of_client()

    for r in range(config.R):
        energy_estimation.reset_transmission_time()

        if config.K > 0:
            config.current_round = r

            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()

            if not server.simnet:
                fed_logger.info("receiving client network info")
                server.client_network(config.EDGE_SERVER_LIST)
                fed_logger.info("test edge servers network")
                server.test_network(config.EDGE_SERVER_LIST)
            else:
                fed_logger.info("receiving client simnet network info")
                server.client_network(config.EDGE_SERVER_LIST)

            for client in server.client_bandwidth.keys():
                clientBW[client].append(server.client_bandwidth[client])

            fed_logger.info("Sending split layer info to clients.")
            server.split_layer()

            fed_logger.info("sending global weights")
            server.no_offloading_global_weights()

            fed_logger.info("start training")
            server.no_edge_offloading_train(config.CLIENTS_LIST)

            fed_logger.info("receiving local weights")
            local_weights = server.c_local_weights(config.CLIENTS_LIST)

            aggregation_start = time.time()
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            aggregation_time = time.time() - aggregation_start

            fed_logger.info("receiving Energy, TT, Remaining-energy, Utilization")
            energy_tt_list = server.energy_tt(config.CLIENTS_LIST)
            fed_logger.info(f"Comp Energy, Comm Energy, TT, Remaining-energy, Utilization :{energy_tt_list}")

            server_sequential_transmission_time = float(energy_estimation.get_transmission_time())
            fed_logger.info(f"Server Sequential Transmission time: {server_sequential_transmission_time}")
            energy_estimation.reset_transmission_time()

            server.client_attendance(config.CLIENTS_LIST)

            e_time = time.time()

            energy = 0
            clientsTT = {}
            for client in energy_tt_list.keys():
                clientCompEnergy[client].append(energy_tt_list[client][0])
                clientCommEnergy[client].append(energy_tt_list[client][1])
                compTime = energy_tt_list[client][0] / (
                        energy_tt_list[client][4] * server.power_usage_of_client[client][0])
                commTime = energy_tt_list[client][1] / server.power_usage_of_client[client][1]
                clientCompTime[client].append(compTime)
                clientCommTime[client].append(commTime)
                clientConsumedEnergy[client].append(energy_tt_list[client][0] + energy_tt_list[client][1])
                clientRemainingEnergy[client].append(energy_tt_list[client][3])
                clientUtilization[client].append(energy_tt_list[client][4])
                comp_time_of_each_client_on_server[client].append(server.computation_time_of_each_client[client])
                energy += (energy_tt_list[client][0] + energy_tt_list[client][1])
                clientsTT[client] = energy_tt_list[client][2] + server.computation_time_of_each_client[client]

            if config.K != 0:
                avgEnergy.append(energy / int(config.K))
            else:
                avgEnergy.append(0)

            if server.simnet:
                simulatedTT, total_tt_of_each_client = server.simnetTrainingTimeCalculation(aggregation_time,
                                                                                            0,
                                                                                            energy_tt_list,
                                                                                            edgeBased=False)
                for client in energy_tt_list.keys():
                    clientTT[client].append(total_tt_of_each_client[client])
                simnet_tt.append(simulatedTT)

            server_flop, _, _ = server.getFlopsOnEdgeAndServer()
            flop_on_server.append(server_flop)
            time_on_server.append(server.total_computation_time)

            fed_logger.info("preparing state...")
            server.offloading = server.get_offloading(server.split_layers)

            fed_logger.info("clustering")
            server.cluster(options)

            fed_logger.info("getting state")
            ttpi = server.ttpi(config.CLIENTS_LIST, clientsTT)
            state = server.concat_norm(ttpi, server.offloading)

            fed_logger.info("splitting")
            server.split(state, options)

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            # Recording each round training time, bandwidth and test accuracy
            training_time = e_time - s_time
            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            # with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
            #     pickle.dump(res, f)
            # fed_logger.info("testing accuracy")
            # test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            # res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
            plot_graph(tt, simnet_tt, avgEnergy, clientConsumedEnergy, clientCompEnergy, clientCommEnergy, clientTT,
                       clientRemainingEnergy, clientBW, None, clientUtilization,
                       None, None, None, None,
                       flop_on_server, time_on_server, clientCompTime, clientCommTime, None,
                       None, None, comp_time_of_each_client_on_server)
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def run_no_edge(server: FedServerInterface, LR, options):
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    avgEnergy, tt, remainingEnergy = [], [], []
    clientBW, edgeBW = [], []
    for r in range(config.R):
        if config.K > 0:
            config.current_round = r
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            fed_logger.info("sending global weights")
            server.no_offloading_global_weights()
            s_time = time.time()
            server.cluster(options)
            fed_logger.info("start training")
            server.no_offloading_train(config.CLIENTS_LIST)
            fed_logger.info("receiving local weights")
            local_weights = server.c_local_weights(config.CLIENTS_LIST)

            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            server.client_attendance(config.CLIENTS_LIST)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            training_time = e_time - s_time
            tt.append(training_time)
            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)
            fed_logger.info("testing accuracy")
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
            plot_graph(tt=tt, accuracy=res['test_acc_record'])
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def plot_graph(tt=None, simnet_tt=None, avgEnergy=None, clientConsumedEnergy=None, clientCompEnergy=None,
               clientCommEnergy=None, clientTT=None, remainingEnergy=None, clientBW=None, edge_serverBW=None,
               clientUtilization=None, accuracy=None, flop_on_each_edge=None, time_on_each_edge=None,
               flop_of_each_edge_on_server=None, flop_on_server=None, time_on_server=None, clientCompTime=None,
               clientCommTime=None, approximated_energy=None, approximated_tt=None,
               computation_time_of_each_client_on_edge=None, computation_time_of_each_client_on_server=None,
               flop_of_each_client_on_edge=None, flop_of_each_client_on_server=None, edge_server_comm_time=None,
               splitting=None, splitting_method=None):
    base_dir = "/fed-flow/Graphs/results_npy"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    np.save(f"{base_dir}/simnet_tt_{splitting_method}.npy", simnet_tt)
    np.save(f"{base_dir}/tt_{splitting_method}.npy", tt)
    np.save(f"{base_dir}/avgEnergy_{splitting_method}.npy", avgEnergy)
    np.save(f"{base_dir}/accuracy_{splitting_method}.npy", accuracy)
    np.save(f"{base_dir}/flop_on_server_{splitting_method}.npy", flop_on_server)
    np.save(f"{base_dir}/time_on_server_{splitting_method}.npy", time_on_server)
    np.save(f"{base_dir}/splitting_{splitting_method}.npy", splitting)

    np.savez(f"{base_dir}/clientConsumedEnergy_{splitting_method}.npz", **clientConsumedEnergy)
    np.savez(f"{base_dir}/clientCompEnergy_{splitting_method}.npz", **clientCompEnergy)
    np.savez(f"{base_dir}/clientCommEnergy_{splitting_method}.npz", **clientCommEnergy)
    np.savez(f"{base_dir}/remainingEnergy_{splitting_method}.npz", **remainingEnergy)
    np.savez(f"{base_dir}/clientUtilization_{splitting_method}.npz", **clientUtilization)

    np.savez(f"{base_dir}/clientCompTime_{splitting_method}.npz", **clientCompTime)
    np.savez(f"{base_dir}/clientCommTime_{splitting_method}.npz", **clientCommTime)

    np.savez(f"{base_dir}/clientBW_{splitting_method}.npz", **clientBW)
    np.savez(f"{base_dir}/edge_serverBW_{splitting_method}.npz", **edge_serverBW)
    np.savez(f"{base_dir}/edge_server_comm_time_{splitting_method}.npz", **edge_server_comm_time)

    np.savez(f"{base_dir}/computation_time_of_each_client_on_edge_{splitting_method}.npz",
             **computation_time_of_each_client_on_edge)
    np.savez(f"{base_dir}/computation_time_of_each_client_on_server_{splitting_method}.npz",
             **computation_time_of_each_client_on_server)

    np.savez(f"{base_dir}/clientTT_{splitting_method}.npz", **clientTT)

    device_colormap = plt.cm.get_cmap('tab10', len(config.CLIENTS_LIST))
    edge_colormap = plt.cm.get_cmap('tab10', len(config.EDGE_SERVER_LIST))

    new_memory = {
        'splitting': splitting[-1],
        'clientInfo': {client: {} for client in config.CLIENTS_LIST},
        'edgeInfo': {edge: {} for edge in config.EDGE_SERVER_LIST},
        'serverInfo': {}
    }
    for edge in config.EDGE_SERVER_LIST:
        new_memory['edgeInfo'][edge]['totalFlopOnEdge'] = flop_on_each_edge[edge][-1]
        new_memory['edgeInfo'][edge]['totalTimeOnEdge'] = time_on_each_edge[edge][-1]
    new_memory['serverInfo']['flopOnServer'] = flop_on_server[-1]
    new_memory['serverInfo']['timeOnServer'] = time_on_server[-1]

    for client in config.CLIENTS_LIST:
        new_memory['clientInfo'][client]['flopOnEdge'] = flop_of_each_client_on_edge[client][-1]
        new_memory['clientInfo'][client]['flopOnServer'] = flop_of_each_client_on_server[client][-1]

    if len(simnet_tt) > 0:
        plt.figure(figsize=(int(10), int(5)))
        plt.title(f"Training time of FL Rounds")
        plt.xlabel("Round")
        plt.ylabel("Training time(s)")
        plt.plot(simnet_tt, color='blue', linewidth='3', label="SIMNET training time")
        # plt.plot(tt, color='red', linewidth='3', label="Real training time")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"trainingTime"))
        plt.close()
    else:
        rl_utils.draw_graph(10, 5, tt, "Training time", "FL Rounds", "Training Time", "/fed-flow/Graphs",
                            "trainingTime", True)
    if avgEnergy:
        rl_utils.draw_graph(10, 5, avgEnergy, "Energy", "FL Rounds", "Energy", "/fed-flow/Graphs",
                            "energy", True)
    if accuracy:
        rl_utils.draw_graph(10, 5, accuracy, "Accuracy", "FL Rounds", "Accuracy", "/fed-flow/Graphs",
                            "accuracy", True)

    # fed_logger.info(Fore.MAGENTA + f"Approx. E: {approximated_energy}")
    # fed_logger.info(Fore.MAGENTA + f"Approx. tt: {approximated_tt}")

    if approximated_energy:
        rl_utils.draw_graph(10, 5, approximated_energy, "Approximated energy", "FL Rounds",
                            "Approx. Energy", "/fed-flow/Graphs", "Approx Energy", True)
    if approximated_tt:
        rl_utils.draw_graph(10, 5, approximated_tt, "Approximated tt", "FL Rounds",
                            "Approx. Tt", "/fed-flow/Graphs", "Approx Tt", True)
    if clientConsumedEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientConsumedEnergy.keys():
            iotDevice_K = clientConsumedEnergy[k]
            plt.title(f"Consumed Energy of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Consumed Energy")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Consumed Energy"))
        plt.close()

    if clientCompEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCompEnergy.keys():
            iotDevice_K = clientCompEnergy[k]
            plt.title(f"Computation Energy of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Computation Energy consumed")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Computation Energy"))
        plt.close()

    if clientCommEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCommEnergy.keys():
            iotDevice_K = clientCommEnergy[k]
            plt.title(f"Communication energy of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("communication energy")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Communication Energy"))
        plt.close()

    if clientCompTime:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCompTime.keys():
            iotDevice_K = clientCompTime[k]

            new_memory['clientInfo'][k]['clientCompTime'] = iotDevice_K[-1]

            plt.title(f"Computation Time of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Computation energy")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Computation time of client"))
        plt.close()

    if clientCommTime:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCommTime.keys():
            iotDevice_K = clientCommTime[k]

            new_memory['clientInfo'][k]['clientCommTime'] = iotDevice_K[-1]

            plt.title(f"Communication Time of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Communication energy")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Communication time of client"))
        plt.close()

    if edge_server_comm_time:
        plt.figure(figsize=(int(25), int(5)))
        for k in edge_server_comm_time.keys():
            iotDevice_K = edge_server_comm_time[k]

            new_memory['clientInfo'][k]['edge_server_comm'] = iotDevice_K[-1]

            plt.title(f"Edge-Server Comm time of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Communication time")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Edge-Server comm time"))
        plt.close()

    if clientTT:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientTT.keys():
            iotDevice_K = clientTT[k]
            plt.title(f"Total time on client")
            plt.xlabel("FL Round")
            plt.ylabel("total time")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"total time on each client"))
        plt.close()

    if remainingEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in remainingEnergy.keys():
            iotDevice_K = remainingEnergy[k]
            plt.title(f"Remaining Energy of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("remaining energy")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Remaining Energies"))
        plt.close()

    if clientUtilization:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientUtilization.keys():
            iotDevice_K = clientUtilization[k]
            plt.title(f"Mean Utilization of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Utilization")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"client utilization"))
        plt.close()

    if clientBW:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientBW.keys():
            iotDevice_K = clientBW[k]
            plt.title(f"BW of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"clientBW"))
        plt.close()

    if edge_serverBW:
        plt.figure(figsize=(int(25), int(5)))
        for k in edge_serverBW.keys():
            edgeDevice_K = edge_serverBW[k]
            plt.title(f"BW of edge devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(edgeDevice_K, color=edge_colormap(config.EDGE_SERVER_LIST.index(k)), linewidth='3',
                     label=f"Edge Device: {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"edge_serverBW"))
        plt.close()

    if computation_time_of_each_client_on_edge:
        plt.figure(figsize=(int(25), int(5)))
        for k in computation_time_of_each_client_on_edge.keys():
            iotDevice_K = computation_time_of_each_client_on_edge[k]

            new_memory['clientInfo'][k]['edgeCompTime'] = iotDevice_K[-1]

            plt.title(f"Comp time of iot device on edge")
            plt.xlabel("Round")
            plt.ylabel("Time (S)")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Comp time on edge"))
        plt.close()

    if computation_time_of_each_client_on_server:
        plt.figure(figsize=(int(25), int(5)))
        for k in computation_time_of_each_client_on_server.keys():
            iotDevice_K = computation_time_of_each_client_on_server[k]

            new_memory['clientInfo'][k]['serverCompTime'] = iotDevice_K[-1]

            plt.title(f"Comp Time of each iot device on server")
            plt.xlabel("Round")
            plt.ylabel("Time (S)")
            plt.plot(iotDevice_K, color=device_colormap(config.CLIENTS_CONFIG[k]), linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Comp time on server"))
        plt.close()

    if flop_on_each_edge:
        flops_of_each_edge = {}
        for edge in flop_on_each_edge.keys():
            flops_of_each_edge[edge] = []

        for edge in flop_on_each_edge.keys():
            rl_utils.draw_scatter(time_on_each_edge[edge], flop_on_each_edge[edge], "FLOP-Time",
                                  "Total time", "FLOP", "/fed-flow/Graphs",
                                  f"FLOP-Time Scatter-{edge}", True)
            flops_of_each_edge[edge] = [w / t if t != 0 else float('inf') for w, t in
                                        zip(flop_on_each_edge[edge], time_on_each_edge[edge])]

        plt.figure(figsize=(int(25), int(5)))
        for k in flops_of_each_edge.keys():
            edgeDevice_K = flops_of_each_edge[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Flops of edge devices")
            plt.xlabel("round")
            plt.ylabel("FLOPS")
            plt.plot(edgeDevice_K, color=color, linewidth='3', label=f"Edge Device: {k}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"FLOPS of each edge"))
        plt.close()

        with open('/fed-flow/Graphs/edge_flop_time.csv', 'w', newline='') as file:
            array = []
            for edgeIndex in range(len(config.EDGE_SERVER_LIST)):
                edge = config.EDGE_SERVER_CONFIG[edgeIndex]
                for flop, timeTaken in zip(flop_on_each_edge[edge], time_on_each_edge[edge]):
                    array.append([edgeIndex, flop, timeTaken])
            writer = csv.writer(file)
            writer.writerows(array)

        model_utils.createFlopsPredictionModel(flop_time_csv_path='/fed-flow/Graphs/edge_flop_time.csv',
                                               isEdge=True)

    fed_logger.info(Fore.MAGENTA + f"{time_on_server}, {flop_on_server}")
    if flop_on_server:
        rl_utils.draw_scatter(time_on_server, flop_on_server, "FLOP-Time", "Total time", "FLOP",
                              "/fed-flow/Graphs", "FLOP-Time Scatter server", True)
        flops_on_server = [w / t if t != 0 else float('inf') for w, t in zip(flop_on_server, time_on_server)]
        plt.title(f"Flops of server")
        plt.xlabel("round")
        plt.ylabel("FLOPS")
        plt.plot(flops_on_server, color="Red", linewidth='3', label=f"Central Server")
        plt.grid()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"FLOPS of central server"))
        plt.close()

        fed_logger.info(Fore.MAGENTA + f"{flop_on_server}, {time_on_server}")
        with open('/fed-flow/Graphs/server_flop_time.csv', 'w', newline='') as file:
            array = []
            for flop, timeTaken in zip(flop_on_server, time_on_server):
                array.append([flop, timeTaken])
            writer = csv.writer(file)
            writer.writerows(array)
        model_utils.createFlopsPredictionModel(flop_time_csv_path='/fed-flow/Graphs/server_flop_time.csv', isEdge=False)

    # saving history
    MEMORY_FILE = "/fed-flow/app/model/memory.json"
    try:
        # Load existing memory
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = {'history': []}  # Create new memory if file doesn't exist

        # Append new decision
    memory["history"].append(new_memory)

    # Save back to JSON
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options_ins))
    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    simnet = options_ins.get("simulatebandwidth") == "True"

    if edge_based and offload:
        energy_estimation.init(os.getpid())
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based, simnet=simnet)
        run_edge_based_offload(server_ins, LR, options_ins)
    elif edge_based and not offload:
        energy_estimation.init(os.getpid())
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_edge_based_no_offload(server_ins, LR, options_ins)
    elif offload:
        energy_estimation.init(os.getpid())
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based, simnet=simnet)
        run_no_edge_offload(server_ins, LR, options_ins)
    else:
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based, simnet=simnet)
        run_no_edge(server_ins, LR, options_ins)
