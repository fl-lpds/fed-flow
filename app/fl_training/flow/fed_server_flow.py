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
    training_time = 0
    totalIOTNum = len(config.CLIENTS_INDEX.keys())
    totalEdgeNum = len(config.EDGE_MAP.keys())
    energy_tt_list = {}
    energy_x = []
    training_y = []

    avgEnergy, tt, simnet_tt = [], [], []
    clientBW, edge_server_BW = {}, {}
    clientRemainingEnergy = {}
    clientConsumedEnergy = {}
    clientCompEnergy = {}
    clientCommEnergy = {}
    clientCompTime = {}
    clientCommTime = {}
    clientUtilization = {}
    clientTT = {}

    flop_on_each_edge = {}
    time_on_each_edge = {}
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

    for edge in config.EDGE_SERVER_LIST:
        edge_server_BW[edge] = []
        flop_on_each_edge[edge] = []
        time_on_each_edge[edge] = []

    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    fed_logger.info(f"OPTION: {options}")

    fed_logger.info(Fore.MAGENTA + f"Calculation of Each layer's activation and gradient size started on server")
    server.calculate_each_layer_activation_gradiant_size()

    fed_logger.info(Fore.MAGENTA + f"Calculation of Each layer's FLOP started on server")
    server.calculate_each_layer_FLOP()

    flops_of_each_layer = server.model_flops_per_layer
    flops_of_each_layer = {key: flops_of_each_layer[key] for key in sorted(flops_of_each_layer)}
    flops_of_each_layer = list(flops_of_each_layer.values())

    test_load_on_edges_and_server = [[[config.model_len - 1, config.model_len - 1] for _ in range(config.K)]]

    # low load on edge 90% of each model on client
    op1, op2 = rl_utils.actionToLayer([0.9, 1.0], flops_of_each_layer)
    test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    # medium load on edge 50% of each model on client
    op1, op2 = rl_utils.actionToLayer([0.5, 1.0], flops_of_each_layer)
    test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    # high load on edge 100% of each model on edge
    op1, op2 = rl_utils.actionToLayer([0.0, 1.0], flops_of_each_layer)
    test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    # low load on server 90% of each model on client
    op1, op2 = rl_utils.actionToLayer([0.9, 0.0], flops_of_each_layer)
    test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    # medium load on server 50% of each model on client
    op1, op2 = rl_utils.actionToLayer([0.5, 0.0], flops_of_each_layer)
    test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    # high load on server 100% of each model on edge
    op1, op2 = rl_utils.actionToLayer([0.0, 0.0], flops_of_each_layer)
    test_load_on_edges_and_server.append([[op1, op2] for _ in range(len(config.CLIENTS_CONFIG.keys()))])

    fed_logger.info(Fore.RED + f"Load testing: {test_load_on_edges_and_server}")

    fed_logger.info('Getting power usage from edge servers')
    server.get_power_of_client()

    for r in range(config.R):
        fed_logger.debug(Fore.LIGHTBLUE_EX + f"number of final K: {config.K}")
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
                # setting BW between each edge and sever
                for edge in config.EDGE_SERVER_LIST:
                    server.edge_bandwidth[edge] = 200_000_000
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

            state = server.edge_based_state()
            fed_logger.info(Fore.RED + f"STATE: {str(state)}")

            if r < len(test_load_on_edges_and_server):
                server.split_layers = test_load_on_edges_and_server[r]
            else:
                fed_logger.info("splitting")
                server.split(state, options)
                fed_logger.info(f"Action : {server.split_layers}")

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
            server.edge_offloading_train(config.CLIENTS_LIST)
            total_training_time = time.time() - start_training_time

            server.total_computation_time = sum(server.computation_time_of_each_client.values())
            fed_logger.info(Fore.RED + f"Total time: {total_training_time}")
            fed_logger.info(Fore.RED + f"Total computation time: {server.total_computation_time}")
            fed_logger.info(Fore.RED + f"each client computation time: {server.computation_time_of_each_client}")

            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            aggregation_start_time = time.time()
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            aggregation_end_time = time.time()
            aggregation_time = aggregation_end_time - aggregation_start_time

            fed_logger.info("receiving Energy, TT, Remaining-energy, Utilization")
            energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
            fed_logger.info(f"Comp Energy, Comm Energy, TT, Remaining-energy, Utilization :{energy_tt_list}")
            server.e_client_attendance(config.CLIENTS_LIST)

            fed_logger.info(f"computation time of each client on server: {server.computation_time_of_each_client}")
            fed_logger.info(
                f"computation time of each client on edge: {server.computation_time_of_each_client_on_edges}")
            fed_logger.info(f"Total computation time on each edge: {server.total_computation_time_of_each_edge}")
            fed_logger.info(f"Transmission time of each client on server: {server.client_training_transmissionTime}")
            fed_logger.info(f"Aggregation Time Simnet bw : {aggregation_time}")
            server_sequential_transmission_time = float(energy_estimation.get_transmission_time())
            fed_logger.info(f"Server Sequential Transmission time: {server_sequential_transmission_time}")
            energy_estimation.reset_transmission_time()

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
                clientTT[client].append(energy_tt_list[client][2])
                clientRemainingEnergy[client].append(energy_tt_list[client][3])
                clientUtilization[client].append(energy_tt_list[client][4])
                energy += (energy_tt_list[client][0] + energy_tt_list[client][1])

            if config.K != 0:
                avgEnergy.append(energy / int(config.K))
            else:
                avgEnergy.append(0)

            e_time = time.time()

            training_time = e_time - s_time
            tt.append(training_time)

            if server.simnet:
                simnet_tt.append(server.simnetTrainingTimeCalculation(aggregation_time,
                                                                      server_sequential_transmission_time,
                                                                      energy_tt_list))

            fed_logger.info(f"Training Time using time.time(): {training_time}")
            server_flop, each_edge_flop, = server.getFlopsOnEdgeAndServer()
            flop_on_server.append(server_flop)
            time_on_server.append(server.total_computation_time)
            for edge in config.EDGE_SERVER_LIST:
                flop_on_each_edge[edge].append(each_edge_flop[edge])
                time_on_each_edge[edge].append(server.total_computation_time_of_each_edge[edge])

            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            # with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
            #     pickle.dump(res, f)

            fed_logger.info("testing accuracy")
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
            plot_graph(tt, simnet_tt, avgEnergy, clientConsumedEnergy, clientCompEnergy, clientCommEnergy, clientTT,
                       clientRemainingEnergy, clientBW, edge_server_BW, clientUtilization, res['test_acc_record'],
                       flop_on_each_edge, time_on_each_edge, flop_on_server, time_on_server, clientCompTime,
                       clientCommTime)
        else:
            break

    fed_logger.info(f"{socket.gethostname()} quit")


def run_no_edge_offload(server: FedServerInterface, LR, options):
    server.initialize(config.split_layer, LR)
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

    for r in range(config.R):
        if config.K > 0:
            config.current_round = r

            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))

            s_time = time.time()
            fed_logger.info("test clients network")
            server.test_network(config.CLIENTS_LIST)

            fed_logger.info("preparing state...")
            server.offloading = server.get_offloading(server.split_layers)

            fed_logger.info("clustering")
            server.cluster(options)

            fed_logger.info("getting state")
            ttpi = server.ttpi(config.CLIENTS_LIST)
            state = server.concat_norm(ttpi, server.offloading)

            fed_logger.info("splitting")
            server.split(state, options)

            server.split_layer()

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)

            fed_logger.info("sending global weights")
            server.no_offloading_global_weights()

            fed_logger.info("start training")
            server.no_edge_offloading_train(config.CLIENTS_LIST)

            fed_logger.info("receiving local weights")
            local_weights = server.c_local_weights(config.CLIENTS_LIST)
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            server.client_attendance(config.CLIENTS_LIST)
            e_time = time.time()

            # Recording each round training time, bandwidth and test accuracy
            training_time = e_time - s_time
            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
                pickle.dump(res, f)
            fed_logger.info("testing accuracy")
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)
            rl_utils.draw_graph(10, 5, res['test_acc_record'], "Accuracy", 'Round', 'Accuracy', "/fed-flow/Graphs/1_3",
                                "Accuracy", True)
            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
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
               flop_on_server=None, time_on_server=None, clientCompTime=None, clientCommTime=None):
    if len(simnet_tt) > 0:
        plt.figure(figsize=(int(10), int(5)))
        plt.title(f"Training time of FL Rounds")
        plt.xlabel("Round")
        plt.ylabel("Training time(s)")
        plt.plot(simnet_tt, color='blue', linewidth='3', label="SIMNET training time")
        plt.plot(tt, color='red', linewidth='3', label="Real training time")
        plt.legend()
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

    if clientConsumedEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientConsumedEnergy.keys():
            iotDevice_K = clientConsumedEnergy[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Consumed Energy of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Consumed Energy")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Consumed Energy"))
        plt.close()

    if clientCompEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCompEnergy.keys():
            iotDevice_K = clientCompEnergy[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Computation Energy of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Computation Energy consumed")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Computation Energy"))
        plt.close()

    if clientCommEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCommEnergy.keys():
            iotDevice_K = clientCommEnergy[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Communication energy of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("communication energy")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Communication Energy"))
        plt.close()

    if clientCompTime:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCompTime.keys():
            iotDevice_K = clientCompTime[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Computation Time of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Computation energy")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Computation time of client"))
        plt.close()

    if clientCommTime:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientCommTime.keys():
            iotDevice_K = clientCommTime[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Communication Time of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Communication energy")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Communication time of client"))
        plt.close()

    if clientTT:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientTT.keys():
            iotDevice_K = clientTT[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Total time on client")
            plt.xlabel("FL Round")
            plt.ylabel("total time")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"total time on each client"))
        plt.close()

    if remainingEnergy:
        plt.figure(figsize=(int(25), int(5)))
        for k in remainingEnergy.keys():
            iotDevice_K = remainingEnergy[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Remaining Energy of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("remaining energy")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Remaining Energies"))
        plt.close()

    if clientUtilization:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientUtilization.keys():
            iotDevice_K = clientUtilization[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Mean Utilization of iot devices")
            plt.xlabel("FL Round")
            plt.ylabel("Utilization")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"client utilization"))
        plt.close()

    if clientBW:
        plt.figure(figsize=(int(25), int(5)))
        for k in clientBW.keys():
            iotDevice_K = clientBW[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"BW of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"clientBW"))
        plt.close()

    if edge_serverBW:
        plt.figure(figsize=(int(25), int(5)))
        for k in edge_serverBW.keys():
            edgeDevice_K = edge_serverBW[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"BW of edge devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(edgeDevice_K, color=color, linewidth='3', label=f"Edge Device: {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"edge_serverBW"))
        plt.close()

    fed_logger.info(Fore.MAGENTA + f"{flop_on_each_edge}, {time_on_each_edge}")
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
    if flop_on_server:
        rl_utils.draw_scatter(time_on_server, flop_on_server, "FLOP-Time", "Total time", "FLOP",
                              "/fed-flow/Graphs", "FLOP-Time Scatter server", True)
        flops_on_server = [w / t if t != 0 else float('inf') for w, t in zip(flop_on_server, time_on_server)]
        plt.title(f"Flops of server")
        plt.xlabel("round")
        plt.ylabel("FLOPS")
        plt.plot(flops_on_server, color="Red", linewidth='3', label=f"Central Server")
        plt.savefig(os.path.join("/fed-flow/Graphs", f"FLOPS of central server"))
        plt.close()

        fed_logger.info(Fore.MAGENTA + f"{flops_on_server}, {time_on_server}")
        with open('/fed-flow/Graphs/server_flop_time.csv', 'w', newline='') as file:
            array = []
            for flop, timeTaken in zip(flop_on_server, time_on_server):
                array.append([flop, timeTaken])
            writer = csv.writer(file)
            writer.writerows(array)
        model_utils.createFlopsPredictionModel(flop_time_csv_path='/fed-flow/Graphs/server_flop_time.csv', isEdge=False)


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options_ins.values()))
    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    simnet = options_ins.get("simulatebandwidth") == "True"

    if edge_based and offload:
        energy_estimation.init(os.getpid())
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based, simnet=simnet)
        run_edge_based_offload(server_ins, LR, options_ins)
    elif edge_based and not offload:
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_edge_based_no_offload(server_ins, LR, options_ins)
    elif offload:
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_no_edge_offload(server_ins, LR, options_ins)
    else:
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
        run_no_edge(server_ins, LR, options_ins)
