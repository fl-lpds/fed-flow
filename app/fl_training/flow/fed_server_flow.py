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
    simnetbw = {}
    for edge in config.EDGE_SERVER_LIST:
        simnetbw[edge] = 580_000_000
    server.initialize(config.split_layer, LR, simnetbw=simnetbw)

    training_time = 0
    totalIOTNum = len(config.CLIENTS_INDEX.keys())
    totalEdgeNum = len(config.EDGE_MAP.keys())
    energy_tt_list = {}
    energy_x = []
    training_y = []

    avgEnergy, tt, simnet_tt = [], [], []
    iotBW, edge_server_BW = {}, {}
    clientRemainingEnergy = {}
    clientConsumedEnergy = {}

    for client in config.CLIENTS_LIST:
        iotBW[client] = []
        clientRemainingEnergy[client] = []
        clientConsumedEnergy[client] = []

    for edge in config.EDGE_SERVER_LIST:
        edge_server_BW[edge] = []

    # for c in config.CLIENTS_LIST:
    #     energy_tt_list.append([0, 0])

    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    fed_logger.info(f"OPTION: {options}")
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
                fed_logger.info("receiving client simnet network info")
                server.client_network(config.EDGE_SERVER_LIST)
                for edge in config.EDGE_SERVER_LIST:
                    simnetbw[edge] = 580_000_000
                # fed_logger.info("receiving edge simnet network info")
                # server.get_simnet_edge_network()

            for client in server.client_bandwidth.keys():
                iotBW[client].append(server.client_bandwidth[client])
            for edge_server in server.edge_bandwidth.keys():
                edge_server_BW[edge_server].append(server.edge_bandwidth[edge_server])

            fed_logger.info("preparing state...")
            server.offloading = server.get_offloading(server.split_layers)

            fed_logger.info("clustering")
            server.cluster(options)

            fed_logger.info("getting state")
            offloading = server.split_layers

            state = server.edge_based_state()
            fed_logger.info(Fore.RED + f"STATE: {str(state)}")

            fed_logger.info("splitting")
            server.split(state, options)
            fed_logger.info(f"Action : {server.split_layers}")
            # server.split_layers = split_list[r]

            fed_logger.info(f"Scattering splitting info to edges.")
            server.send_split_layers_config()

            if r > 49:
                LR = config.LR * 0.1

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR, simnetbw=simnetbw)
            fed_logger.info("sending global weights")
            server.edge_offloading_global_weights()
            # fed_logger.info('==> Reinitialization Finish')

            fed_logger.info("start training")
            server.edge_offloading_train(config.CLIENTS_LIST)

            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            aggregation_start_time = time.time()
            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)
            aggregation_end_time = time.time()
            aggregation_time = aggregation_end_time - aggregation_start_time

            fed_logger.info("receiving Energy, TT, Remaining-energy")
            energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
            fed_logger.info(f"Energy, TT, Remaining-energy :{energy_tt_list}")

            fed_logger.info(f"computation time of each client on server: {server.computation_time_of_each_client}")
            fed_logger.info(
                f"computation time of each client on edge: {server.computation_time_of_each_client_on_edges}")
            fed_logger.info(f"Transmission time of each client on server: {server.client_training_transmissionTime}")
            fed_logger.info(f"Aggregation Time Simnet bw : {aggregation_time}")
            fed_logger.info(f"Server Sequential Transmission time: {float(energy_estimation.get_transmission_time())}")

            energy = 0
            for client in energy_tt_list.keys():
                clientConsumedEnergy[client].append(energy_tt_list[client][0])
                clientRemainingEnergy[client].append(energy_tt_list[client][2])
                energy += energy_tt_list[client][0]
            avgEnergy.append(energy / int(config.K))

            server.e_client_attendance(config.CLIENTS_LIST)

            e_time = time.time()

            # Recording each round training time, bandwidth and test_app accuracy
            training_time = e_time - s_time
            tt.append(training_time)

            if server.simnet:
                total_computation_time_for_each_client = {}
                if len(config.CLIENTS_LIST) > 0:
                    for clientip in config.CLIENTS_LIST:
                        total_computation_time_for_each_client[clientip] = \
                            server.computation_time_of_each_client_on_edges[clientip] + \
                            server.computation_time_of_each_client[clientip] + \
                            energy_tt_list[clientip][1] + \
                            server.client_training_transmissionTime[clientip]
                    trainingTime_simnetBW = float(energy_estimation.get_transmission_time()) + \
                                            max(total_computation_time_for_each_client.values()) + \
                                            aggregation_time
                    simnet_tt.append(trainingTime_simnetBW)
                    fed_logger.info(f"Training time using Simnet bw : {trainingTime_simnetBW}")
                else:
                    fed_logger.info("All Clients had been Turned off.")

            fed_logger.info(f"Training Time using time.time(): {training_time}")

            res['training_time'].append(training_time)
            res['bandwidth_record'].append(server.bandwith())
            # with open(config.home + '/results/FedAdapt_res.pkl', 'wb') as f:
            #     pickle.dump(res, f)

            fed_logger.info("testing accuracy")
            test_acc = model_utils.test(server.uninet, server.testloader, server.device, server.criterion)
            res['test_acc_record'].append(test_acc)

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
            plot_graph(tt, simnet_tt, avgEnergy, clientConsumedEnergy, clientRemainingEnergy, iotBW, edge_server_BW,
                       res['test_acc_record'])
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

            fed_logger.info('Round Finish')
            fed_logger.info('==> Round Training Time: {:}'.format(training_time))
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def run_no_edge(server: FedServerInterface, LR, options):
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    avgEnergy, tt, remainingEnergy = [], [], []
    iotBW, edgeBW = [], []
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


def plot_graph(tt=None, simnet_tt=None, avgEnergy=None, clientConsumedEnergy=None, remainingEnergy=None, iotBW=None,
               edge_serverBW=None, accuracy=None):
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

    if iotBW:
        plt.figure(figsize=(int(25), int(5)))
        for k in iotBW.keys():
            iotDevice_K = iotBW[k]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"BW of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"iotBW"))
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
