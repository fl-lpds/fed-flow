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
    server.initialize(config.split_layer, LR)
    training_time = 0
    energy_tt_list = []
    # all_splitting = rl_utils.allPossibleSplitting(7, 1)
    energy_x = []
    training_y = []
    # split_list = [[[0, 1]], [[1, 2]], [[2, 3]], [[3, 4]], [[4, 5]], [[5, 6]]]
    avgEnergy, tt, remainingEnergy = [], [], []
    iotBW, edgeBW = [], []
    x = []
    for c in config.CLIENTS_LIST:
        energy_tt_list.append([0, 0])
    res = {}
    res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []
    fed_logger.info(f"OPTION: {options}")
    for r in range(config.R):
        fed_logger.debug(Fore.LIGHTBLUE_EX + f"number of final K: {config.K}")
        if config.K > 0:
            config.current_round = r
            x.append(r)
            fed_logger.info('====================================>')
            fed_logger.info('==> Round {:} Start'.format(r))
            s_time = time.time()
            fed_logger.info("receiving client network info")
            server.client_network(config.EDGE_SERVER_LIST)

            fed_logger.info("test edge servers network")
            server.test_network(config.EDGE_SERVER_LIST)

            fed_logger.info("preparing state...")
            server.offloading = server.get_offloading(server.split_layers)

            fed_logger.info("clustering")
            server.cluster(options)

            fed_logger.info("getting state")
            offloading = server.split_layers

            state = server.edge_based_state()
            fed_logger.info("state: " + str(state))
            normalizedState = []
            for bw in state[:config.K + config.S]:
                if r < 50:
                    normalizedState.append(bw / 100_000_000)
                else:
                    normalizedState.append(bw / 10_000_000)
            iotBW.append(normalizedState[:config.K])
            edgeBW.append(normalizedState[config.K:config.K + config.S])

            fed_logger.info("splitting")
            server.split(normalizedState, options)
            fed_logger.info(f"Agent Action : {server.split_layers}")
            # server.split_layers = split_list[r]
            server.get_split_layers_config_from_edge()

            if r > 49:
                LR = config.LR * 0.1

            fed_logger.info("initializing server")
            server.initialize(server.split_layers, LR)
            fed_logger.info("sending global weights")
            server.edge_offloading_global_weights()
            # fed_logger.info('==> Reinitialization Finish')

            fed_logger.info("start training")
            server.edge_offloading_train(config.CLIENTS_LIST)

            fed_logger.info("receiving local weights")
            local_weights = server.e_local_weights(config.CLIENTS_LIST)

            fed_logger.info("aggregating weights")
            server.call_aggregation(options, local_weights)

            energy_tt_list = server.e_energy_tt(config.CLIENTS_LIST)
            print(f"E TT :{energy_tt_list}")
            clientEnergy = []
            for i in range(config.K):
                clientEnergy.append(energy_tt_list[i][0])
                remainingEnergy.append(energy_tt_list[i][2])
            avgEnergy.append(sum(clientEnergy) / int(config.K))
            server.e_client_attendance(config.CLIENTS_LIST)

            e_time = time.time()

            # Recording each round training time, bandwidth and test_app accuracy
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

            # plot_graph(x, tt, avgEnergy, remainingEnergy, iotBW, edgeBW, res['test_acc_record'])
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
    x = []
    server.initialize(config.split_layer, LR)
    for r in range(config.R):
        if config.K > 0:
            config.current_round = r
            x.append(r)
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
            plot_graph(x=x, tt=tt, accuracy=res['test_acc_record'])
        else:
            break
    fed_logger.info(f"{socket.gethostname()} quit")


def plot_graph(x, tt=None, avgEnergy=None, remainingEnergy=None, iotBW=None, edgeBW=None, accuracy=None):
    if tt:
        rl_utils.draw_graph(10, 5, x, tt, "Training time", "FL Rounds", "Training Time", "/fed-flow/Graphs",
                            "trainingTime", True)
    if avgEnergy:
        rl_utils.draw_graph(10, 5, x, avgEnergy, "Energy time", "FL Rounds", "Energy", "/fed-flow/Graphs",
                            "energy", True)
    if accuracy:
        rl_utils.draw_graph(10, 5, x, accuracy, "Accuracy", "FL Rounds", "Accuracy", "/fed-flow/Graphs",
                            "accuracy", True)

    if remainingEnergy:
        x = [i for i in range(len(remainingEnergy) - 1)]
        plt.figure(figsize=(int(25), int(5)))
        iotDevice_K = []
        for k in range(config.K):
            for i in range(len(remainingEnergy)):
                iotDevice_K.append(remainingEnergy[i][k])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"Remaining Energy of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("remaining energy")
            plt.plot(x, iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"Remaining Energies"))
        plt.close()

    if iotBW:
        x = [i for i in range(len(iotBW) - 1)]
        plt.figure(figsize=(int(25), int(5)))
        iotDevice_K = []
        for k in range(config.K):
            for i in range(len(iotBW)):
                iotDevice_K.append(iotBW[i][k])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"BW of iot devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(x, iotDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"iotBW"))
        plt.close()

    if edgeBW:
        x = [i for i in range(len(edgeBW) - 1)]
        plt.figure(figsize=(int(25), int(5)))
        edgeDevice_K = []
        for k in range(config.S):
            for i in range(len(edgeBW)):
                edgeDevice_K.append(edgeBW[i][k])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.title(f"BW of edge devices")
            plt.xlabel("timestep")
            plt.ylabel("BW")
            plt.plot(x, edgeDevice_K, color=color, linewidth='3', label=f"Device {k}")
        plt.legend()
        plt.savefig(os.path.join("/fed-flow/Graphs", f"edgeBW"))
        plt.close()


def run(options_ins):
    LR = config.LR
    fed_logger.info('Preparing Sever.')
    fed_logger.info("start mode: " + str(options_ins.values()))
    offload = options_ins.get('offload')
    edge_based = options_ins.get('edgebased')
    if edge_based and offload:
        server_ins = FedServer(options_ins.get('model'),
                               options_ins.get('dataset'), offload, edge_based)
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
