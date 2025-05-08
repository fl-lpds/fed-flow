import copy
import json
import random
from collections import Counter

import joblib
import numpy as np
import torch
from colorama import Fore

from app.config import config
from app.config.logger import fed_logger
from app.model.entity.rl_model import PPO
from app.util import model_utils


# def edge_based_rl_splitting(state, labels):
#     env = rl_utils.CustomEnv()
#     agent = DDPG.load('/fed-flow/app/agent/160.zip', env=env,
#                       custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
#     floatAction, _ = agent.predict(observation=state, deterministic=True)
#     actions = []
#     for i in range(0, len(floatAction), 2):
#         actions.append([rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[0],
#                         rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[1]])
#
#     return actions


def edge_based_heuristic_splitting(state: dict, label):
    fed_logger.info(Fore.MAGENTA + f"Heuristic Splitting algorithm =======================")
    approximated_tt = 0
    approximated_energy = 0

    # ALPHA == 1 try minimum energy and ALPHA == 0 is vice versa
    ALPHA = 1
    score_threshold = 0.5

    candidate_splitting = []
    min_energy_splitting_for_each_client = {}
    min_time_splitting_for_each_client = {}
    filtered_min_time_splitting_for_each_client = {}
    min_energy_trainingTime_splitting_for_each_client = {}
    comp_time_of_each_layer_on_clients = {}
    for client in state['client_bw'].keys():
        min_energy_splitting_for_each_client[client] = []
        min_time_splitting_for_each_client[client] = []
        filtered_min_time_splitting_for_each_client[client] = []
        min_energy_trainingTime_splitting_for_each_client[client] = []
        comp_time_of_each_layer_on_clients[client] = []

    MODEL_PATH = '/fed-flow/app/model'
    edge_linear_model = joblib.load(f"{MODEL_PATH}/edge_flops_prediction_linear_model.pkl")
    edge_poly_model = joblib.load(f"{MODEL_PATH}/edge_flops_prediction_poly_model.pkl")

    server_linear_model = joblib.load(f"{MODEL_PATH}/server_flops_prediction_linear_model.pkl")
    server_poly_model = joblib.load(f"{MODEL_PATH}/server_flops_prediction_linear_model.pkl")
    fed_logger.info(Fore.MAGENTA + f"Edge and server flops prediction model loaded =======================")

    client_remaining_runtime = {}
    previous_action = state['prev_action']
    previous_edge_nice_value = state['prev_edge_nice_value']
    previous_server_nice_value = state['prev_server_nice_value']

    total_model_size = state['total_model_size']
    activation_size = state['activation_size']
    activation_size[config.model_len - 1] = 0
    gradient_size = state['gradient_size']
    flops_of_each_layer = state['flops_of_each_layer']
    flops_of_each_layer = {key: flops_of_each_layer[key] for key in sorted(flops_of_each_layer)}
    flops_of_each_layer = list(flops_of_each_layer.values())
    fed_logger.info(Fore.MAGENTA + f"Total flops: {flops_of_each_layer}")

    client_bw = state['client_bw']
    edge_server_bw = state['edge_bw']
    current_round = state['current_round']

    client_comp_energy = state['client_comp_energy']
    client_comm_energy = state['client_comm_energy']
    client_utilization = state['client_utilization']
    client_remaining_energy: dict = state['client_remaining_energy']
    client_power_usage = state['client_power']
    client_comp_time = state['client_comp_time']

    runningClients = []
    for clientIndex in range(len(previous_action)):
        if previous_action[clientIndex] != [config.model_len, config.model_len]:
            runningClients.append(config.CLIENTS_INDEX[clientIndex])

    for client in runningClients:
        comp_time_of_each_layer_on_clients[client].append(client_comp_time[client][0])
        for op1 in range(1, config.model_len - 1):
            comp_time_of_each_layer_on_clients[client].append(client_comp_time[client][op1 + 1] - client_comp_time[client][op1])

    client_comm_time = {client: comm_energy / client_power_usage[client][1] for client, comm_energy in
                        client_comm_energy.items()}
    edge_server_comm_time = state['edge_server_comm_time']
    comp_time_of_each_client_on_edges = state['comp_time_of_each_client_on_edge']
    comp_time_of_each_client_on_server = state['comp_time_of_each_client_on_server']
    total_time_on_each_edge = {edgeIP: 0 for edgeIP in config.EDGE_SERVER_LIST}
    total_time_on_server = sum(comp_time_of_each_client_on_server.values())
    action = previous_action

    best_tt_splitting_found = state['best_tt_splitting_found']

    fed_logger.info(Fore.GREEN + f"STATE:")
    fed_logger.info(Fore.GREEN + f"=============================================================================")
    fed_logger.info(Fore.GREEN + f"Previous action: {action}")
    for client in runningClients:
        fed_logger.info(Fore.GREEN + f"Client: {client}")
        fed_logger.info(Fore.GREEN + f"   Client BW: {client_bw[client]}")
        fed_logger.info(Fore.GREEN + f"   Client Comp Energy: {client_comp_energy[client]}")
        fed_logger.info(Fore.GREEN + f"   Client Comm Energy: {client_comm_energy[client]}")
        fed_logger.info(Fore.GREEN + f"   Client Comp Time: {client_comp_time[client]}")
        fed_logger.info(Fore.GREEN + f"   Client Comm Time: {client_comm_time[client]}")
        fed_logger.info(Fore.GREEN + f"   Edge Comp Time Of Client: {comp_time_of_each_client_on_edges[client]}")
        fed_logger.info(Fore.GREEN + f"   Server Comp Time of Client: {comp_time_of_each_client_on_server[client]}")
        fed_logger.info(Fore.GREEN + f"   Edge-Server Comm Time of Client: {edge_server_comm_time[client]}")
        fed_logger.info(Fore.GREEN + f"-------------------------------------------------------------------------")

    batchNumber = (config.N / len(config.CLIENTS_CONFIG.keys())) / config.B

    each_splitting_share = {op1: {op2: {} for op2 in range(op1, config.model_len)} for op1 in range(config.model_len)}
    max_computation_on_client = sum(flops_of_each_layer)
    max_computation_on_edge_and_server = sum(flops_of_each_layer[1:])
    max_comm = (2 * batchNumber * max(activation_size.values())) + (2 * total_model_size)

    for op1 in range(config.model_len):
        for op2 in range(op1, config.model_len):
            each_splitting_share[op1][op2] = {
                'client_comp': sum(flops_of_each_layer[:op1 + 1]) / max_computation_on_client,
                'edge_comp': sum(flops_of_each_layer[op1 + 1:op2 + 1]) / max_computation_on_edge_and_server,
                'server_comp': sum(flops_of_each_layer[op2 + 1:]) / max_computation_on_edge_and_server,
                'client_comm': (2 * batchNumber * activation_size[op1]) + (2 * total_model_size) / max_comm if (op1 != config.model_len - 1)
                else (2 * total_model_size) / max_comm,
                'edge_server_comm': (2 * batchNumber * activation_size[op2]) + (2 * total_model_size) / max_comm if op2 != config.model_len - 1
                else (2 * total_model_size) / max_comm,
            }

    clients_computation_e, clients_communication_e, clients_totals_e = energyEstimator(previous_action,
                                                                                       client_bw, activation_size,
                                                                                       batchNumber, total_model_size,
                                                                                       client_comp_energy,
                                                                                       client_power_usage)
    for client in client_remaining_energy.keys():
        client_remaining_runtime[client] = client_remaining_energy[client] / (clients_totals_e[client])

    client_remaining_runtime_comp_score = normalizer(client_remaining_runtime)
    client_score = client_remaining_runtime_comp_score

    fed_logger.info(f"RUN TIME SCORE: {client_remaining_runtime_comp_score.items()}")

    classicFL_action = [[config.model_len - 1, config.model_len - 1] for _ in range(len(config.CLIENTS_CONFIG))]
    classicFL_tt, _, _, _, _, _, _, _ = trainingTimeEstimator(classicFL_action, client_comp_time, client_bw,
                                                              edge_server_bw, flops_of_each_layer, activation_size,
                                                              total_model_size, batchNumber, edge_poly_model, server_poly_model,
                                                              None,
                                                              None)

    clients_classicFL_comp_energy, clients_classicFL_comm_energy, clients_classicFL_total_energy = energyEstimator(
        classicFL_action, client_bw, activation_size, batchNumber, total_model_size, client_comp_energy,
        client_power_usage)

    if config.K != 0:
        classicFL_avg_energy = sum(clients_classicFL_total_energy.values()) / config.K
    else:
        classicFL_avg_energy = 0

    fed_logger.info(Fore.MAGENTA + f"Classic FL training time approximation: {classicFL_tt}")
    fed_logger.info(Fore.MAGENTA + f"Classic FL average energy approximation: {classicFL_avg_energy}")
    fed_logger.info(Fore.MAGENTA + f"Classic FL communication energy approximation: {clients_classicFL_comm_energy}")
    fed_logger.info(Fore.MAGENTA + f"Classic FL computation energy approximation: {clients_classicFL_comp_energy}")

    currentAction = previous_action

    fed_logger.info(Fore.GREEN + f"Splitting Map:")
    fed_logger.info(Fore.GREEN + f"===============================================================================")

    clients_time_for_each_op1 = {client: {} for client in runningClients}

    for client, score in client_score.items():
        fed_logger.info(Fore.GREEN + f"Client: {client}")
        fed_logger.info(Fore.GREEN + f"Running Clients: {runningClients}")
        fed_logger.info(Fore.GREEN + f"clients_time_for_each_op1: {clients_time_for_each_op1}")

        client_op1_energy = []
        client_op1_time = []
        for layer in range(config.model_len):
            op1 = layer
            if layer != config.model_len - 1:
                size = activation_size[layer]
                tt_trans = ((2 * size * batchNumber) + (2 * total_model_size)) / client_bw[client]
            else:
                tt_trans = (2 * total_model_size) / client_bw[client]
            comm_energy = tt_trans * client_power_usage[client][1]
            comp_energy = client_comp_energy[client][op1]
            total_energy = comp_energy + comm_energy
            total_time = client_comp_time[client][op1] + tt_trans
            client_op1_energy.append((op1, total_energy))
            client_op1_time.append((op1, total_time))
            clients_time_for_each_op1[client][op1] = total_time
        min_energy_splitting_for_each_client[client] = sorted(client_op1_energy, key=lambda x: x[1])
        min_time_splitting_for_each_client[client] = sorted(client_op1_time, key=lambda x: x[1])
        filtered_min_time_splitting_for_each_client[client] = [item for item in
                                                               min_time_splitting_for_each_client[client] if
                                                               item[1] <= classicFL_tt]
        for op1, energy in min_energy_splitting_for_each_client[client]:
            if clients_time_for_each_op1[client][op1] < classicFL_tt:
                min_energy_trainingTime_splitting_for_each_client[client].append((op1, energy, clients_time_for_each_op1[client][op1]))

        # fed_logger.info(Fore.GREEN + f"   Energy[Ascending]: {min_energy_splitting_for_each_client[client]}")
        # fed_logger.info(Fore.GREEN + f"   Time[Ascending]: {min_time_splitting_for_each_client[client]}")
        # fed_logger.info(
        #     Fore.GREEN + f"   Energy-TrainingTime[Ascending]: {min_energy_trainingTime_splitting_for_each_client[client]}")
        fed_logger.info(Fore.GREEN + f"---------------------------------------------------------------------------")

    isEnergyEfficient = {client: tuple() for client in runningClients}
    for client in runningClients:
        clientOP1 = previous_action[config.CLIENTS_CONFIG[client]][0]
        optimalEnergyOP1 = min_energy_trainingTime_splitting_for_each_client[client][0][0]
        if clientOP1 == optimalEnergyOP1:
            isEnergyEfficient[client] = (True, optimalEnergyOP1)
        else:
            isEnergyEfficient[client] = (False, optimalEnergyOP1)

    baseline_tt = classicFL_tt

    clients_score = sorted(client_score.items(), key=lambda item: item[1])
    fed_logger.info(Fore.MAGENTA + f"Current Round: {config.current_round}, model_len: {config.model_len}")

    if config.current_round == config.model_len:
        action = [[op1s[0][0], config.model_len - 1] for client, op1s in
                  min_energy_trainingTime_splitting_for_each_client.items()]
        for client in runningClients:
            clientOP1 = action[config.CLIENTS_CONFIG[client]][0]
            edgeIP = config.CLIENT_MAP[client]

            op2_bigger_than_op1 = {layer: size for layer, size in activation_size.items() if layer >= clientOP1}
            layer_with_min_size = min(op2_bigger_than_op1, key=op2_bigger_than_op1.get)

            if layer_with_min_size != config.model_len - 1:
                edge_server_comm_time_temp = (
                        ((2 * (activation_size[layer_with_min_size]) * batchNumber) + (2 * total_model_size)) /
                        edge_server_bw[edgeIP])
            else:
                edge_server_comm_time_temp = (2 * total_model_size) / edge_server_bw[edgeIP]

            if (min_energy_trainingTime_splitting_for_each_client[client][0][2] + edge_server_comm_time_temp) < classicFL_tt:
                action[config.CLIENTS_CONFIG[client]][1] = layer_with_min_size
        edge_nice_value = {client: 0 for client in config.CLIENTS_CONFIG.keys()}
        server_nice_value = {client: 0 for client in config.CLIENTS_CONFIG.keys()}
        return action, 0, 0, edge_nice_value, server_nice_value

    new_edge_nice_value = copy.deepcopy(previous_edge_nice_value)
    new_server_nice_value = copy.deepcopy(previous_server_nice_value)
    prev_action_tt, prev_action_each_client_total_tt, prev_action_each_client_tt, _, _, _, _, _ = (
        trainingTimeEstimator(previous_action, client_comp_time, client_bw, edge_server_bw, flops_of_each_layer,
                              activation_size, total_model_size, batchNumber, edge_poly_model, server_poly_model,
                              comp_time_of_each_client_on_edges, comp_time_of_each_client_on_server))

    fed_logger.info(Fore.MAGENTA + f"Prev Action tt: {prev_action_tt}, Baseline tt: {baseline_tt}")
    fed_logger.info(Fore.MAGENTA + f"Action's training time: {prev_action_tt}")
    fed_logger.info(Fore.MAGENTA + f"Action's training time per client: {prev_action_each_client_total_tt}")
    fed_logger.info(Fore.MAGENTA + f"Action's training time per client[per section]: {prev_action_each_client_tt}")

    baseline_tt *= 1.05
    # if prev_action_tt <= baseline_tt:
    #     return previous_action, 0, 0, previous_edge_nice_value, previous_server_nice_value

    # Phase 1: Finding bottleneck devices
    bad_clients = {client: time for client, time in prev_action_each_client_total_tt.items() if time > baseline_tt}
    bad_clients_cluster = {
        edge: {'edge_comp': [], 'client_comp': [], 'server_comp': [], 'client_comm': [], 'edge_server_comm': []} for
        edge in config.EDGE_SERVER_LIST}

    good_clients = {client: time for client, time in prev_action_each_client_total_tt.items() if
                    time <= baseline_tt}
    good_clients_cluster = {
        edge: {'edge_comp': [], 'client_comp': [], 'server_comp': [], 'client_comm': [], 'edge_server_comm': []} for
        edge in config.EDGE_SERVER_LIST}

    fed_logger.info(Fore.MAGENTA + f"Bad Clients: {bad_clients}")
    fed_logger.info(Fore.MAGENTA + f"Good Clients: {good_clients}")

    shares = {client: {} for client in config.CLIENTS_CONFIG.keys()}
    bottlenecks = {client: [] for client in config.CLIENTS_CONFIG.keys()}

    high_prio_bad_client = []
    # Sort bad clients based on their score
    for scoreItem in clients_score:
        client = scoreItem[0]
        if client in bad_clients.keys():
            high_prio_bad_client.append(client)

    # finding slow part of training of bad device and try to solve it
    for client in config.CLIENTS_LIST:
        isBadClient = True if prev_action_each_client_total_tt[client] > baseline_tt else False
        total_client_time = prev_action_each_client_total_tt[client]

        clientOP1 = previous_action[config.CLIENTS_CONFIG[client]][0]
        clientOP2 = previous_action[config.CLIENTS_CONFIG[client]][1]

        client_comp_share = prev_action_each_client_tt[client]['client_comp'] / total_client_time
        edge_comp_share = prev_action_each_client_tt[client]['edge_comp'] / total_client_time
        server_comp_share = prev_action_each_client_tt[client]['server_comp'] / total_client_time
        client_comm_share = prev_action_each_client_tt[client]['client_comm'] / total_client_time
        edge_server_comm_share = prev_action_each_client_tt[client]['edge_server_comm'] / total_client_time

        op2_bigger_than_op1 = {layer: size for layer, size in activation_size.items() if layer >= clientOP1}
        layer_with_min_size = min(op2_bigger_than_op1, key=op2_bigger_than_op1.get)

        shares[client] = {'client_comp': 0,
                          'edge_comp': edge_comp_share,
                          'server_comp': server_comp_share,
                          'client_comm': 0,
                          'edge_server_comm': edge_server_comm_share}
        if clientOP2 == layer_with_min_size:
            shares[client]['edge_server_comm'] = 0

        bottlenecks[client] = sorted(shares[client].items(), key=lambda item: item[1], reverse=True)

        fed_logger.info(Fore.MAGENTA + f"{client} SHARES: {shares}")
        fed_logger.info(Fore.MAGENTA + f"Client comp time: {prev_action_each_client_tt[client]['client_comp']}")
        fed_logger.info(Fore.MAGENTA + f"Edges comp time: {prev_action_each_client_tt[client]['edge_comp']}")
        fed_logger.info(Fore.MAGENTA + f"Server comp time: {prev_action_each_client_tt[client]['server_comp']}")
        fed_logger.info(Fore.MAGENTA + f"Client comm time: {prev_action_each_client_tt[client]['client_comm']}")
        fed_logger.info(
            Fore.MAGENTA + f"Edge-Server comm time: {prev_action_each_client_tt[client]['edge_server_comm']}")
        fed_logger.info(Fore.MAGENTA + f"Client Bottleneck: {bottlenecks[client]}")

        if isBadClient:
            client_obj = {'client': client,
                          'bottleneck_time': prev_action_each_client_tt[client][bottlenecks[client][0][0]]}
            bad_clients_cluster[config.CLIENT_MAP[client]][bottlenecks[client][0][0]].append(client)
        else:
            client_obj = {'client': client,
                          'bottleneck_time': prev_action_each_client_tt[client][bottlenecks[client][0][0]]}
            good_clients_cluster[config.CLIENT_MAP[client]][bottlenecks[client][0][0]].append(client)

    fed_logger.info(Fore.MAGENTA + f"Bad Clients' edge : {bad_clients_cluster}")
    fed_logger.info(Fore.MAGENTA + f"Good Clients' edge : {good_clients_cluster}")

    new_action = copy.deepcopy(previous_action)

    high_prio_bad_energy_consuming_client = [client for client, (status, _) in isEnergyEfficient.items() if status is False]

    changed_device = []
    current_bad_device = high_prio_bad_client
    time_for_each_client = prev_action_each_client_tt
    total_time_for_each_client = prev_action_each_client_total_tt

    bad_tt_device_and_bad_energy = copy.deepcopy(current_bad_device)
    for client in clients_score:
        if (client in high_prio_bad_energy_consuming_client) and (client not in bad_tt_device_and_bad_energy):
            bad_tt_device_and_bad_energy.append(client)

    while len(bad_tt_device_and_bad_energy) != 0:
        badClient = bad_tt_device_and_bad_energy[0]
        edgeIP = config.CLIENT_MAP[badClient]
        bottleneck = bottlenecks[badClient][0][0]
        deviceChanged = False
        currentOffloading = new_action[config.CLIENTS_CONFIG[badClient]]
        currentOffloadingShares = each_splitting_share[currentOffloading[0]][currentOffloading[1]]
        fed_logger.info(Fore.GREEN + f"Bad Client: {badClient}")
        fed_logger.info(Fore.GREEN + f"Bad Client Bottleneck: {bottleneck}")
        fed_logger.info(Fore.GREEN + f"Bad Client times: {time_for_each_client[badClient]}")
        fed_logger.info(Fore.GREEN + f"High Priority Energy: {high_prio_bad_energy_consuming_client}")
        fed_logger.info(Fore.GREEN + f"IsEnergy Efficient: {isEnergyEfficient}")

        if (badClient in high_prio_bad_energy_consuming_client) and (badClient not in current_bad_device):
            best_time_section = sorted(((k, v) for k, v in time_for_each_client[badClient].items() if k != 'client_comp' or k != 'client_comm'),
                                       key=lambda item: item[1])

            for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                temp = copy.deepcopy(new_action)
                temp[config.CLIENTS_CONFIG[badClient]][0] = op1
                sorted_op2 = sorted(each_splitting_share[op1].items(), key=lambda item: item[1][best_time_section[0][0]])
                for op2, share in sorted_op2:
                    if share[best_time_section[0][0]] > currentOffloadingShares[best_time_section[0][0]]:
                        if op2 != config.model_len - 1:
                            edge_server_comm_time_temp = (
                                    ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) / edge_server_bw[edgeIP])
                        else:
                            edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                        if (clients_time_for_each_op1[badClient][op1] + edge_server_comm_time_temp) < baseline_tt:
                            temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                            isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                       runningClients,
                                                                                                       current_bad_device,
                                                                                                       time_for_each_client,
                                                                                                       baseline_tt,
                                                                                                       client_comp_time,
                                                                                                       client_bw,
                                                                                                       edge_server_bw,
                                                                                                       activation_size,
                                                                                                       total_model_size,
                                                                                                       batchNumber)
                            if not isViolated:
                                if memoryAvailability == 'BOTH':
                                    if ((clients_time_for_each_op1[badClient][op1] +
                                         newTimes[badClient]['edge_comp'] +
                                         newTimes[badClient]['server_comp'] +
                                         edge_server_comm_time_temp) < baseline_tt):
                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        time_for_each_client = newTimes
                                        total_time_for_each_client = newTotalTimes
                                        deviceChanged = True
                                        break
                                elif memoryAvailability == 'EDGE':
                                    if ((clients_time_for_each_op1[badClient][op1] +
                                         newTimes[badClient]['edge_comp'] +
                                         edge_server_comm_time_temp) < baseline_tt):
                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        deviceChanged = True
                                        break
                                elif memoryAvailability == 'SERVER':
                                    if ((newTimes[badClient]['server_comp'] +
                                         edge_server_comm_time_temp +
                                         clients_time_for_each_op1[badClient][op1]) < baseline_tt):
                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        deviceChanged = True
                                        break
                                elif memoryAvailability == 'NONE':
                                    if (clients_time_for_each_op1[badClient][op1] + edge_server_comm_time_temp) < baseline_tt:
                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        deviceChanged = True
                                        break
                                else:
                                    raise Exception('Invalid memory availability in check violation function')
                if deviceChanged:
                    break
        else:
            if bottleneck == 'edge_comp':
                if badClient in high_prio_bad_energy_consuming_client:
                    for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                        _, _, clients_totals_e = energyEstimator(previous_action, client_bw, activation_size, batchNumber, total_model_size,
                                                                 client_comp_energy, client_power_usage)
                        if energy < clients_totals_e[badClient]:
                            sorted_op2_by_edge_comp = sorted(each_splitting_share[op1].items(), key=lambda item: item[1]['edge_comp'],
                                                             reverse=True)
                            for op2, share in sorted_op2_by_edge_comp:
                                if share['edge_comp'] < currentOffloadingShares['edge_comp']:
                                    if op2 != config.model_len - 1:
                                        edge_server_comm_time_temp = (
                                                ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) /
                                                edge_server_bw[edgeIP])
                                    else:
                                        edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                                    if (clients_time_for_each_op1[badClient][op1] + edge_server_comm_time_temp) < baseline_tt:
                                        temp_energy_action = copy.deepcopy(new_action)
                                        temp_energy_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp_energy_action,
                                                                                                                   badClient,
                                                                                                                   runningClients,
                                                                                                                   current_bad_device,
                                                                                                                   time_for_each_client,
                                                                                                                   baseline_tt,
                                                                                                                   client_comp_time,
                                                                                                                   client_bw,
                                                                                                                   edge_server_bw,
                                                                                                                   activation_size,
                                                                                                                   total_model_size,
                                                                                                                   batchNumber)
                                        if not isViolated:
                                            if memoryAvailability == 'BOTH':
                                                time_for_each_client = newTimes
                                                total_time_for_each_client = newTotalTimes
                                            new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                            deviceChanged = True
                                            break
                            if deviceChanged:
                                break

                if not deviceChanged:
                    edges_client_comp_time = {client: times['edge_comp'] for client, times in
                                              time_for_each_client.items() if
                                              client in config.EDGE_MAP[edgeIP] and client in runningClients and times['edge_comp'] != 0}

                    triedNiceValue = True
                    high_outlier, low_outlier = detect_outliers_using_iqr(edges_client_comp_time)
                    for client in edges_client_comp_time.keys():
                        if client in high_outlier:
                            if previous_edge_nice_value[client] != -10:
                                triedNiceValue = False
                                break
                        if client in low_outlier:
                            if previous_edge_nice_value[client] != 10:
                                triedNiceValue = False

                    if are_times_close_enough(edges_client_comp_time) or triedNiceValue:
                        for client in edges_client_comp_time.keys():
                            new_edge_nice_value[client] = 0

                        time_diff = total_time_for_each_client[badClient] - baseline_tt
                        op2_selected = False
                        if time_for_each_client[badClient]['edge_comp'] - time_for_each_client[badClient]['server_comp'] > time_diff:
                            sorted_op2_by_edge_comp = sorted(each_splitting_share[currentOffloading[0]].items(),
                                                             key=lambda item: item[1]['edge_comp'],
                                                             reverse=True)
                            for op2, share in sorted_op2_by_edge_comp:
                                if share['edge_comp'] < currentOffloadingShares['edge_comp']:

                                    if op2 != config.model_len - 1:
                                        edge_server_comm_time_temp = (
                                                ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) / edge_server_bw[edgeIP])
                                    else:
                                        edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                                    if (clients_time_for_each_op1[badClient][currentOffloading[0]] + edge_server_comm_time_temp) < baseline_tt:
                                        temp = copy.deepcopy(new_action)
                                        temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                                        isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                   runningClients,
                                                                                                                   current_bad_device,
                                                                                                                   time_for_each_client,
                                                                                                                   baseline_tt,
                                                                                                                   client_comp_time,
                                                                                                                   client_bw,
                                                                                                                   edge_server_bw,
                                                                                                                   activation_size,
                                                                                                                   total_model_size,
                                                                                                                   batchNumber)
                                        if not isViolated:
                                            if memoryAvailability == 'BOTH':
                                                if ((clients_time_for_each_op1[badClient][currentOffloading[0]] +
                                                     newTimes[badClient]['edge_comp'] + newTimes[badClient]['server_comp'] +
                                                     newTimes[badClient]['edge_server_comm']) < baseline_tt):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    time_for_each_client = newTimes
                                                    total_time_for_each_client = newTotalTimes
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'EDGE':
                                                if ((newTimes[badClient]['edge_comp'] + edge_server_comm_time_temp) <
                                                        time_for_each_client[badClient]['edge_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'SERVER':
                                                if ((newTimes[badClient]['server_comp'] + edge_server_comm_time_temp) <
                                                        time_for_each_client[badClient]['server_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'NONE':
                                                fed_logger.info(Fore.GREEN + f"Bad Client time: {time_for_each_client}")
                                                if edge_server_comm_time_temp < time_for_each_client[badClient]['edge_comp']:
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    op2_selected = True
                                                    break
                                            else:
                                                raise Exception('Invalid memory availability in check violation function')
                        if not op2_selected:
                            temp = copy.deepcopy(new_action)
                            # we must increase energy consumption, because of edge-server bw
                            for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                                if op1 > currentOffloading[0]:
                                    temp[config.CLIENTS_CONFIG[badClient]][0] = op1
                                    if op1 <= currentOffloading[1]:
                                        isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                   runningClients,
                                                                                                                   current_bad_device,
                                                                                                                   time_for_each_client,
                                                                                                                   baseline_tt,
                                                                                                                   client_comp_time,
                                                                                                                   client_bw,
                                                                                                                   edge_server_bw,
                                                                                                                   activation_size,
                                                                                                                   total_model_size,
                                                                                                                   batchNumber)
                                        if not isViolated:
                                            if memoryAvailability == 'BOTH':
                                                new_action[config.CLIENTS_CONFIG[badClient]][0] = op1
                                                time_for_each_client = newTimes
                                                total_time_for_each_client = newTotalTimes
                                                break
                                            elif memoryAvailability == 'EDGE':
                                                if ((newTimes[badClient]['edge_comp'] +
                                                     newTimes[badClient]['client_comm'] +
                                                     newTimes[badClient]['client_comp']) <
                                                        time_for_each_client[badClient]['edge_comp'] +
                                                        time_for_each_client[badClient]['client_comm'] +
                                                        time_for_each_client[badClient]['client_comp']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][0] = op1
                                                    break
                                            elif memoryAvailability == 'SERVER':
                                                if ((newTimes[badClient]['server_comp'] +
                                                     newTimes[badClient]['client_comm'] +
                                                     newTimes[badClient]['client_comp'] +
                                                     newTimes[badClient]['edge_server_comm']) <
                                                        time_for_each_client[badClient]['edge_comp'] +
                                                        time_for_each_client[badClient]['client_comm'] +
                                                        time_for_each_client[badClient]['client_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][0] = op1
                                                    break
                                            elif memoryAvailability == 'NONE':
                                                new_action[config.CLIENTS_CONFIG[badClient]][0] = op1
                                                break
                                    else:
                                        for op2 in range(op1, config.model_len):
                                            temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                                            isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                       runningClients,
                                                                                                                       current_bad_device,
                                                                                                                       time_for_each_client,
                                                                                                                       baseline_tt,
                                                                                                                       client_comp_time,
                                                                                                                       client_bw,
                                                                                                                       edge_server_bw,
                                                                                                                       activation_size,
                                                                                                                       total_model_size,
                                                                                                                       batchNumber)

                                            if not isViolated:
                                                if memoryAvailability == 'BOTH':
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    time_for_each_client = newTimes
                                                    total_time_for_each_client = newTotalTimes
                                                    op2_selected = True
                                                    break
                                                elif memoryAvailability == 'EDGE':
                                                    if ((newTimes[badClient]['edge_comp'] +
                                                         newTimes[badClient]['client_comm'] +
                                                         newTimes[badClient]['client_comp'] +
                                                         newTimes[badClient]['edge_server_comm']) <
                                                            time_for_each_client[badClient]['edge_comp'] +
                                                            time_for_each_client[badClient]['client_comm'] +
                                                            time_for_each_client[badClient]['client_comp'] +
                                                            time_for_each_client[badClient]['edge_server_comm']):
                                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                        op2_selected = True
                                                        break
                                                elif memoryAvailability == 'SERVER':
                                                    if ((newTimes[badClient]['server_comp'] +
                                                         newTimes[badClient]['client_comm'] +
                                                         newTimes[badClient]['client_comp'] +
                                                         newTimes[badClient]['edge_server_comm']) <
                                                            time_for_each_client[badClient]['server_comp'] +
                                                            time_for_each_client[badClient]['client_comm'] +
                                                            time_for_each_client[badClient]['client_comp'] +
                                                            time_for_each_client[badClient]['edge_server_comm']):
                                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                        op2_selected = True
                                                        break
                                                elif memoryAvailability == 'NONE':
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    break
                                        if op2_selected:
                                            break
                    else:
                        # Give bad device a high priority and vice versa
                        high_outlier, low_outlier = detect_outliers_using_iqr(edges_client_comp_time)
                        for highOutlierClient in high_outlier:
                            if previous_edge_nice_value[highOutlierClient] != -10:
                                new_edge_nice_value[highOutlierClient] = -10
                        for lowOutlierClient in low_outlier:
                            if previous_edge_nice_value[lowOutlierClient] != 10:
                                new_edge_nice_value[lowOutlierClient] = 10
            elif bottleneck == 'server_comp':
                if badClient in high_prio_bad_energy_consuming_client:
                    for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                        _, _, clients_totals_e = energyEstimator(previous_action, client_bw, activation_size, batchNumber, total_model_size,
                                                                 client_comp_energy, client_power_usage)
                        if energy < clients_totals_e[badClient]:
                            sorted_op2_by_server_comp = sorted(each_splitting_share[op1].items(), key=lambda item: item[1]['server_comp'],
                                                               reverse=True)
                            for op2, share in sorted_op2_by_server_comp:
                                if share['server_comp'] < currentOffloadingShares['server_comp']:
                                    if op2 != config.model_len - 1:
                                        edge_server_comm_time_temp = (
                                                ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) / edge_server_bw[edgeIP])
                                    else:
                                        edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                                    if (clients_time_for_each_op1[badClient][op1] + edge_server_comm_time_temp) < baseline_tt:
                                        temp_energy_action = copy.deepcopy(new_action)
                                        temp_energy_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp_energy_action,
                                                                                                                   badClient,
                                                                                                                   runningClients,
                                                                                                                   current_bad_device,
                                                                                                                   time_for_each_client,
                                                                                                                   baseline_tt,
                                                                                                                   client_comp_time,
                                                                                                                   client_bw,
                                                                                                                   edge_server_bw,
                                                                                                                   activation_size,
                                                                                                                   total_model_size,
                                                                                                                   batchNumber)
                                        if not isViolated:
                                            if memoryAvailability == 'BOTH':
                                                time_for_each_client = newTimes
                                                total_time_for_each_client = newTotalTimes
                                            new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                            deviceChanged = True
                                            break
                            if deviceChanged:
                                break
                if not deviceChanged:
                    servers_client_comp_time = {client: times['server_comp'] for client, times in prev_action_each_client_tt.items()
                                                if times['server_comp'] != 0}
                    triedNiceValue = True
                    high_outlier, low_outlier = detect_outliers_using_iqr(servers_client_comp_time)
                    for client in servers_client_comp_time.keys():
                        if client in high_outlier:
                            if previous_server_nice_value[client] != -10:
                                triedNiceValue = False
                                break
                        if client in low_outlier:
                            if previous_server_nice_value[client] != 10:
                                triedNiceValue = False

                    if are_times_close_enough(servers_client_comp_time) or triedNiceValue:
                        for client in servers_client_comp_time.keys():
                            new_server_nice_value[client] = 0
                        time_diff = total_time_for_each_client[badClient] - baseline_tt
                        op2_selected = False
                        if time_for_each_client[badClient]['server_comp'] - time_for_each_client[badClient]['edge_comp'] > time_diff:
                            sorted_op2_by_server_comp = sorted(each_splitting_share[currentOffloading[0]].items(),
                                                               key=lambda item: item[1]['server_comp'], reverse=True)
                            for op2, share in sorted_op2_by_server_comp:
                                if share['server_comp'] < currentOffloadingShares['server_comp']:

                                    if op2 != config.model_len - 1:
                                        edge_server_comm_time_temp = (
                                                ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) / edge_server_bw[edgeIP])
                                    else:
                                        edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                                    if (clients_time_for_each_op1[badClient][currentOffloading[0]] + edge_server_comm_time_temp) < baseline_tt:
                                        temp = copy.deepcopy(new_action)
                                        temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                                        isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                   runningClients,
                                                                                                                   current_bad_device,
                                                                                                                   time_for_each_client,
                                                                                                                   baseline_tt,
                                                                                                                   client_comp_time,
                                                                                                                   client_bw,
                                                                                                                   edge_server_bw,
                                                                                                                   activation_size,
                                                                                                                   total_model_size,
                                                                                                                   batchNumber)
                                        if not isViolated:
                                            if memoryAvailability == 'BOTH':
                                                if ((clients_time_for_each_op1[badClient][currentOffloading[0]] +
                                                     newTimes[badClient]['edge_comp'] + newTimes[badClient]['server_comp'] +
                                                     newTimes[badClient]['edge_server_comm']) < baseline_tt):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    time_for_each_client = newTimes
                                                    total_time_for_each_client = newTotalTimes
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'EDGE':
                                                if ((newTimes[badClient]['edge_comp'] + edge_server_comm_time_temp) <
                                                        time_for_each_client[badClient]['edge_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'SERVER':
                                                if ((newTimes[badClient]['server_comp'] + edge_server_comm_time_temp) <
                                                        time_for_each_client[badClient]['server_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'NONE':
                                                if edge_server_comm_time_temp < time_for_each_client[badClient]['edge_comp']:
                                                    new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                    op2_selected = True
                                                    break
                                            else:
                                                raise Exception('Invalid memory availability in check violation function')
                        if not op2_selected:
                            temp = copy.deepcopy(new_action)
                            for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                                if op1 > currentOffloading[0]:
                                    temp[config.CLIENTS_CONFIG[badClient]][0] = op1
                                    if op1 < currentOffloading[1]:
                                        for op2 in range(currentOffloading[1], config.model_len):
                                            temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                                            isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                       runningClients,
                                                                                                                       current_bad_device,
                                                                                                                       time_for_each_client,
                                                                                                                       baseline_tt,
                                                                                                                       client_comp_time,
                                                                                                                       client_bw,
                                                                                                                       edge_server_bw,
                                                                                                                       activation_size,
                                                                                                                       total_model_size,
                                                                                                                       batchNumber)

                                            if not isViolated:
                                                if memoryAvailability == 'BOTH':
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    time_for_each_client = newTimes
                                                    total_time_for_each_client = newTotalTimes
                                                    op2_selected = True
                                                    break
                                                elif memoryAvailability == 'EDGE':
                                                    if ((newTimes[badClient]['edge_comp'] +
                                                         newTimes[badClient]['client_comm'] +
                                                         newTimes[badClient]['client_comp'] +
                                                         newTimes[badClient]['edge_server_comm']) <
                                                            time_for_each_client[badClient]['edge_comp'] +
                                                            time_for_each_client[badClient]['client_comm'] +
                                                            time_for_each_client[badClient]['client_comp'] +
                                                            time_for_each_client[badClient]['edge_server_comm']):
                                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                        op2_selected = True
                                                        break
                                                elif memoryAvailability == 'SERVER':
                                                    if ((newTimes[badClient]['server_comp'] +
                                                         newTimes[badClient]['client_comm'] +
                                                         newTimes[badClient]['client_comp'] +
                                                         newTimes[badClient]['edge_server_comm']) <
                                                            time_for_each_client[badClient]['server_comp'] +
                                                            time_for_each_client[badClient]['client_comm'] +
                                                            time_for_each_client[badClient]['client_comp'] +
                                                            time_for_each_client[badClient]['edge_server_comm']):
                                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                        op2_selected = True
                                                        break
                                                elif memoryAvailability == 'NONE':
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    break
                                        if op2_selected:
                                            break
                                    else:
                                        for op2 in range(op1, config.model_len):
                                            temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                                            isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                       runningClients,
                                                                                                                       current_bad_device,
                                                                                                                       time_for_each_client,
                                                                                                                       baseline_tt,
                                                                                                                       client_comp_time,
                                                                                                                       client_bw,
                                                                                                                       edge_server_bw,
                                                                                                                       activation_size,
                                                                                                                       total_model_size,
                                                                                                                       batchNumber)

                                            if not isViolated:
                                                if memoryAvailability == 'BOTH':
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    time_for_each_client = newTimes
                                                    total_time_for_each_client = newTotalTimes
                                                    op2_selected = True
                                                    break
                                                elif memoryAvailability == 'EDGE':
                                                    if ((newTimes[badClient]['edge_comp'] +
                                                         newTimes[badClient]['client_comm'] +
                                                         newTimes[badClient]['client_comp'] +
                                                         newTimes[badClient]['edge_server_comm']) <
                                                            time_for_each_client[badClient]['edge_comp'] +
                                                            time_for_each_client[badClient]['client_comm'] +
                                                            time_for_each_client[badClient]['client_comp'] +
                                                            time_for_each_client[badClient]['edge_server_comm']):
                                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                        op2_selected = True
                                                        break
                                                elif memoryAvailability == 'SERVER':
                                                    if ((newTimes[badClient]['server_comp'] +
                                                         newTimes[badClient]['client_comm'] +
                                                         newTimes[badClient]['client_comp'] +
                                                         newTimes[badClient]['edge_server_comm']) <
                                                            time_for_each_client[badClient]['server_comp'] +
                                                            time_for_each_client[badClient]['client_comm'] +
                                                            time_for_each_client[badClient]['client_comp'] +
                                                            time_for_each_client[badClient]['edge_server_comm']):
                                                        new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                        op2_selected = True
                                                        break
                                                elif memoryAvailability == 'NONE':
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    break
                                        if op2_selected:
                                            break
                    else:
                        # Give bad device a high priority and vic versa
                        high_outlier, low_outlier = detect_outliers_using_iqr(servers_client_comp_time)
                        for highOutlierClient in high_outlier:
                            if previous_server_nice_value[highOutlierClient] != -10:
                                new_server_nice_value[highOutlierClient] = -10
                        for lowOutlierClient in low_outlier:
                            if previous_server_nice_value[lowOutlierClient] != 10:
                                new_server_nice_value[lowOutlierClient] = 10
            else:
                fed_logger.info(Fore.YELLOW + f"Edge-Server comm")

                currentOffloading = new_action[config.CLIENTS_CONFIG[badClient]]
                currentOffloadingShares = each_splitting_share[currentOffloading[0]][currentOffloading[1]]

                if badClient in high_prio_bad_energy_consuming_client:
                    fed_logger.info(Fore.YELLOW + f"Edge-Server comm => improving energy")

                    for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                        fed_logger.info(Fore.YELLOW + f"Edge-Server comm => improving energy [{op1}, ]")

                        _, _, clients_totals_e = energyEstimator(previous_action, client_bw, activation_size, batchNumber, total_model_size,
                                                                 client_comp_energy, client_power_usage)
                        sorted_op2_by_edge_server_comm = sorted(each_splitting_share[op1].items(),
                                                                key=lambda item: item[1]['edge_server_comm'], reverse=True)
                        fed_logger.info(Fore.YELLOW + f"Edge-Server comm => improving energy [{op1}, ] => op2 by edge-server comm: "
                                                      f"{sorted_op2_by_edge_server_comm}")

                        if energy < clients_totals_e[badClient]:
                            for op2, share in sorted_op2_by_edge_server_comm:
                                fed_logger.info(Fore.YELLOW + f"Edge-Server comm => improving energy [{op1}, {op2}]")

                                if share['edge_server_comm'] < currentOffloadingShares['edge_server_comm']:
                                    if op2 != config.model_len - 1:
                                        edge_server_comm_time_temp = (
                                                ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) / edge_server_bw[edgeIP])
                                    else:
                                        edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                                        if (clients_time_for_each_op1[badClient][op1] + edge_server_comm_time_temp) < baseline_tt:
                                            temp_energy_action = copy.deepcopy(new_action)
                                            temp_energy_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                            isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp_energy_action,
                                                                                                                       badClient,
                                                                                                                       runningClients,
                                                                                                                       current_bad_device,
                                                                                                                       time_for_each_client,
                                                                                                                       baseline_tt,
                                                                                                                       client_comp_time,
                                                                                                                       client_bw,
                                                                                                                       edge_server_bw,
                                                                                                                       activation_size,
                                                                                                                       total_model_size,
                                                                                                                       batchNumber)
                                            if not isViolated:
                                                if memoryAvailability == 'BOTH':
                                                    time_for_each_client = newTimes
                                                    total_time_for_each_client = newTotalTimes
                                                new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                deviceChanged = True
                                                break
                            if deviceChanged:
                                break
                if not deviceChanged:
                    fed_logger.info(Fore.YELLOW + f"Edge-Server comm => device not changed")

                    sorted_op2_by_edge_server_comm = sorted(each_splitting_share[currentOffloading[0]].items(),
                                                            key=lambda item: item[1]['edge_server_comm'], reverse=True)
                    time_diff = total_time_for_each_client[badClient] - baseline_tt
                    op2_selected = False
                    if abs(time_for_each_client[badClient]['server_comp'] - time_for_each_client[badClient]['edge_comp']) > time_diff:
                        for op2, share in sorted_op2_by_edge_server_comm:
                            fed_logger.info(Fore.YELLOW + f"Edge-Server comm => deviceNotChanged => op2: {op2}")

                            if share['edge_server_comm'] < currentOffloadingShares['edge_server_comm']:

                                if op2 != config.model_len - 1:
                                    edge_server_comm_time_temp = (
                                            ((2 * (activation_size[op2]) * batchNumber) + (2 * total_model_size)) / edge_server_bw[edgeIP])
                                else:
                                    edge_server_comm_time_temp = (2 * total_model_size / edge_server_bw[edgeIP])

                                if (clients_time_for_each_op1[badClient][currentOffloading[0]] + edge_server_comm_time_temp) < baseline_tt:
                                    temp = copy.deepcopy(new_action)
                                    temp[config.CLIENTS_CONFIG[badClient]][1] = op2
                                    isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                               runningClients,
                                                                                                               current_bad_device,
                                                                                                               time_for_each_client,
                                                                                                               baseline_tt,
                                                                                                               client_comp_time,
                                                                                                               client_bw,
                                                                                                               edge_server_bw,
                                                                                                               activation_size,
                                                                                                               total_model_size,
                                                                                                               batchNumber)
                                    if not isViolated:
                                        if memoryAvailability == 'BOTH':
                                            if ((clients_time_for_each_op1[badClient][currentOffloading[0]] +
                                                 newTimes[badClient]['edge_comp'] + newTimes[badClient]['server_comp'] +
                                                 newTimes[badClient]['edge_server_comm']) < baseline_tt):
                                                new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                time_for_each_client = newTimes
                                                total_time_for_each_client = newTotalTimes
                                                op2_selected = True
                                                break
                                        elif memoryAvailability == 'EDGE':
                                            if ((newTimes[badClient]['edge_comp'] + edge_server_comm_time_temp) <
                                                    time_for_each_client[badClient]['edge_comp'] +
                                                    time_for_each_client[badClient]['edge_server_comm']):
                                                new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                op2_selected = True
                                                break
                                        elif memoryAvailability == 'SERVER':
                                            if ((newTimes[badClient]['server_comp'] + edge_server_comm_time_temp) <
                                                    time_for_each_client[badClient]['server_comp'] +
                                                    time_for_each_client[badClient]['edge_server_comm']):
                                                new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                op2_selected = True
                                                break
                                        elif memoryAvailability == 'NONE':
                                            if edge_server_comm_time_temp < time_for_each_client[badClient]['edge_comp']:
                                                new_action[config.CLIENTS_CONFIG[badClient]][1] = op2
                                                op2_selected = True
                                                break
                                        else:
                                            raise Exception('Invalid memory availability in check violation function')

                    if not op2_selected:
                        temp = copy.deepcopy(new_action)
                        for op1, energy, tt in min_energy_trainingTime_splitting_for_each_client[badClient]:
                            _, _, clients_totals_e = energyEstimator(previous_action, client_bw, activation_size, batchNumber, total_model_size,
                                                                     client_comp_energy, client_power_usage)

                            if energy > clients_totals_e[badClient]:
                                sorted_op2_by_edge_server_comm = sorted(each_splitting_share[op1].items(),
                                                                        key=lambda item: item[1]['edge_server_comm'], reverse=True)
                                for op2, share in sorted_op2_by_edge_server_comm:
                                    if share['edge_server_comm'] < currentOffloadingShares['edge_server_comm']:
                                        temp[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                        isViolated, memoryAvailability, newTimes, newTotalTimes = checkTTViolation(temp, badClient,
                                                                                                                   runningClients,
                                                                                                                   current_bad_device,
                                                                                                                   time_for_each_client,
                                                                                                                   baseline_tt,
                                                                                                                   client_comp_time,
                                                                                                                   client_bw,
                                                                                                                   edge_server_bw,
                                                                                                                   activation_size,
                                                                                                                   total_model_size,
                                                                                                                   batchNumber)

                                        if not isViolated:
                                            if memoryAvailability == 'BOTH':
                                                new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                time_for_each_client = newTimes
                                                total_time_for_each_client = newTotalTimes
                                                op2_selected = True
                                                break
                                            elif memoryAvailability == 'EDGE':
                                                if ((newTimes[badClient]['edge_comp'] +
                                                     newTimes[badClient]['client_comm'] +
                                                     newTimes[badClient]['client_comp'] +
                                                     newTimes[badClient]['edge_server_comm']) <
                                                        time_for_each_client[badClient]['edge_comp'] +
                                                        time_for_each_client[badClient]['client_comm'] +
                                                        time_for_each_client[badClient]['client_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'SERVER':
                                                if ((newTimes[badClient]['server_comp'] +
                                                     newTimes[badClient]['client_comm'] +
                                                     newTimes[badClient]['client_comp'] +
                                                     newTimes[badClient]['edge_server_comm']) <
                                                        time_for_each_client[badClient]['server_comp'] +
                                                        time_for_each_client[badClient]['client_comm'] +
                                                        time_for_each_client[badClient]['client_comp'] +
                                                        time_for_each_client[badClient]['edge_server_comm']):
                                                    new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                    op2_selected = True
                                                    break
                                            elif memoryAvailability == 'NONE':
                                                new_action[config.CLIENTS_CONFIG[badClient]] = [op1, op2]
                                                break
                                if op2_selected:
                                    break

        # _, _, clients_totals_e = energyEstimator(new_action, client_bw, activation_size, batchNumber, total_model_size, client_comp_energy,
        #                                          client_power_usage)
        #
        # for client in client_remaining_energy.keys():
        #     client_remaining_runtime[client] = client_remaining_energy[client] / (clients_totals_e[client])
        #
        # client_remaining_runtime_comp_score = normalizer(client_remaining_runtime)
        # client_score = client_remaining_runtime_comp_score
        # clients_score = sorted(client_score.items(), key=lambda item: item[1], reverse=True)

        changed_device.append(badClient)
        current_bad_device.remove(badClient)
        bad_tt_device_and_bad_energy.remove(badClient)
        fed_logger.info(Fore.GREEN + f"NEW ACTION2: {new_action}")

        for client in runningClients:
            total_client_time = total_time_for_each_client[client]

            clientOP1 = new_action[config.CLIENTS_CONFIG[client]][0]
            clientOP2 = new_action[config.CLIENTS_CONFIG[client]][1]
            edgeIP = config.CLIENT_MAP[client]

            client_comp_share = time_for_each_client[client]['client_comp'] / total_client_time
            edge_comp_share = time_for_each_client[client]['edge_comp'] / total_client_time
            server_comp_share = time_for_each_client[client]['server_comp'] / total_client_time
            client_comm_share = time_for_each_client[client]['client_comm'] / total_client_time
            edge_server_comm_share = time_for_each_client[client]['edge_server_comm'] / total_client_time

            op2_bigger_than_op1 = {layer: size for layer, size in activation_size.items() if layer >= clientOP1}
            fed_logger.info(Fore.GREEN + f"op2_bigger_than_op1: {op2_bigger_than_op1}")
            fed_logger.info(Fore.GREEN + f"NEW ACTION: {new_action}")
            fed_logger.info(Fore.GREEN + f"clientOP1, OP2: {clientOP1} , {clientOP2}")

            layer_with_min_size = min(op2_bigger_than_op1, key=op2_bigger_than_op1.get)

            shares[client] = {'client_comp': 0,
                              'edge_comp': edge_comp_share,
                              'server_comp': server_comp_share,
                              'client_comm': 0,
                              'edge_server_comm': edge_server_comm_share}
            if clientOP2 == layer_with_min_size:
                shares[client]['edge_server_comm'] = 0

            bottlenecks[client] = sorted(shares[client].items(), key=lambda item: item[1], reverse=True)

    return new_action, 0, 0, new_edge_nice_value, new_server_nice_value


def normalizer(input_dict):
    input_list = list(input_dict.values())
    min_item = min(input_list)
    max_item = max(input_list)
    if max_item == min_item:
        return {client: 0.5 for client in input_dict}
    normalized = {
        client: (usage - min_item) / (max_item - min_item)
        for client, usage in input_dict.items()
    }
    return normalized


def rl_splitting(state, labels):
    state_dim = 2 * config.G
    action_dim = config.G
    agent = None
    if agent is None:
        # Initialize trained RL agent
        agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma,
                        config.K_epochs, config.eps_clip)
        agent.policy.load_state_dict(torch.load('/fed-flow/app/agent/PPO_FedAdapt.pth'))
    action = agent.exploit(state)
    action = expand_actions(action, config.CLIENTS_LIST, labels)

    result = action_to_layer(action)
    config.split_layer = result
    return result


def none(state, labels):
    split_layer = []
    for c in config.CLIENTS_LIST:
        split_layer.append(model_utils.get_unit_model_len() - 1)

    config.split_layer = split_layer
    return config.split_layer


def no_edge_fake(state, labels):
    split_list = []
    for i in range(config.K):
        split_list.append(3)
    return split_list


def fake(state, labels):
    """
    a fake splitting list of tuples
    """

    split_list = []
    for i in range(len(config.CLIENTS_INDEX.keys())):
        split_list.append([3, 3])
    return split_list


def no_splitting(state, labels):
    split_list = []
    for i in range(config.K):
        split_list.append([config.model_len - 1, config.model_len - 1])
    return split_list


def only_edge_splitting(state, labels):
    split_list = []
    for i in range(len(config.CLIENTS_CONFIG.keys())):
        split_list.append([0, config.model_len - 1])
    return split_list


def only_server_splitting(state, labels):
    split_list = []
    for i in range(len(config.CLIENTS_CONFIG.keys())):
        split_list.append([0, 0])
    return split_list


# HFLP used random partitioning for splitting
def randomSplitting(state, labels):
    """ Randomly split the model between clients edge devices and cloud server """

    splittingArray = []
    for i in range(len(config.CLIENTS_CONFIG.keys())):
        op1 = random.randint(0, config.model_len - 1)
        op2 = random.randint(op1, config.model_len - 1)
        splittingArray.append([op1, op2])
    return splittingArray


# FedMec: which empirically deploys the convolutional layers of a DNN on the device-side while
# assigning the remaining part to the edge server
def FedMec(state, labels):
    lastConvolutionalLayerIndex = 0
    for i in config.model_cfg["VGG5"]:
        """ C means convolutional layer """
        if i[0] == 'C':
            lastConvolutionalLayerIndex = config.model_cfg["VGG5"].index(i)

    splittingArray = [[lastConvolutionalLayerIndex, config.model_len - 1] for _ in range(len(config.CLIENTS_CONFIG))]
    return splittingArray


def expand_actions(actions, clients_list, group_labels):  # Expanding group actions to each device
    full_actions = []

    for i in range(len(clients_list)):
        full_actions.append(actions[group_labels[i]])

    return full_actions


def action_to_layer(action):  # Expanding group actions to each device
    # first caculate cumulated flops
    model_state_flops = []
    cumulated_flops = 0

    for l in model_utils.get_unit_model().cfg:
        cumulated_flops += l[5]
        model_state_flops.append(cumulated_flops)

    model_flops_list = np.array(model_state_flops)
    model_flops_list = model_flops_list / cumulated_flops

    split_layer = []
    for v in action:
        idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min())

        idx = idx[0][-1]
        if idx >= 5:  # all FC layers combine to one option
            idx = 6
        split_layer.append([int(idx), int(idx)])
    return split_layer


def actionToLayerEdgeBase(splitDecision: list[float]) -> tuple[int, int]:
    """ It returns the offloading points for the given action ( op1 , op2 )"""
    op1: int
    op2: int  # Offloading points op1, op2
    workLoad = []
    model_state_flops = []

    for l in model_utils.get_unit_model().cfg:
        workLoad.append(l[5])
        model_state_flops.append(sum(workLoad))

    totalWorkLoad = sum(workLoad)
    model_flops_list = np.array(model_state_flops)
    model_flops_list = model_flops_list / totalWorkLoad
    idx = np.where(np.abs(model_flops_list - splitDecision[0]) == np.abs(model_flops_list - splitDecision[0]).min())
    op1 = idx[0][-1]

    op2_totalWorkload = sum(workLoad[op1:])
    model_state_flops = []
    for l in range(op1, model_utils.get_unit_model_len()):
        model_state_flops.append(sum(workLoad[op1:l + 1]))
    model_flops_list = np.array(model_state_flops)
    model_flops_list = model_flops_list / op2_totalWorkload

    idx = np.where(np.abs(model_flops_list - splitDecision[1]) == np.abs(model_flops_list - splitDecision[1]).min())
    op2 = idx[0][-1] + op1

    return op1, op2


def energyEstimator(action, client_bw, activation_size, batchNumber, total_model_size, client_comp_energy,
                    client_power_usage):
    client_computation_energy = {client: 0 for client, _ in client_bw.items()}
    client_communication_energy = {client: 0 for client, _ in client_bw.items()}
    client_total_energy = {client: 0 for client, _ in client_bw.items()}

    for client in config.CLIENTS_CONFIG.keys():
        client_op1 = action[config.CLIENTS_CONFIG[client]][0]
        if client_op1 != config.model_len - 1:
            tt_trans = ((2 * activation_size[client_op1] * batchNumber) + (2 * total_model_size)) / client_bw[client]
        else:
            tt_trans = (2 * total_model_size) / client_bw[client]
        client_computation_energy[client] = client_comp_energy[client][client_op1]
        client_communication_energy[client] = tt_trans * client_power_usage[client][1]
        client_total_energy[client] = client_communication_energy[client] + client_computation_energy[client]

    return client_computation_energy, client_communication_energy, client_total_energy


def checkTTViolation(currentAction, changedClient, active_clients, bad_clients, time_for_each_client, baseline_tt, comp_time_on_each_client,
                     clients_bw, edge_server_bw, activation_size, total_model_size, batchNumber):
    fed_logger.info(Fore.GREEN + f"Check Violation")
    fed_logger.info(Fore.GREEN + f"=======================================")
    fed_logger.info(Fore.GREEN + f"CURRENT SPLITTING: {currentAction}")
    edge_memory = triedBefore(currentAction, config.CLIENT_MAP[changedClient])
    server_memory = triedBefore(currentAction)

    temp_time_for_each_client = copy.deepcopy(time_for_each_client)

    for client in active_clients:
        edgeIP = config.CLIENT_MAP[client]
        ClientOP1 = currentAction[config.CLIENTS_CONFIG[client]][0]
        ClientOP2 = currentAction[config.CLIENTS_CONFIG[client]][1]

        if ClientOP1 != config.model_len - 1:
            transmission_time_on_each_client = (((2 * activation_size[ClientOP1] * batchNumber) + (2 * total_model_size)) /
                                                clients_bw[client])
        else:
            transmission_time_on_each_client = (2 * total_model_size) / clients_bw[client]

        if ClientOP2 != config.model_len - 1:
            edge_server_transmission_time_for_each_client = (((2 * activation_size[ClientOP2] * batchNumber) + (2 * total_model_size)) /
                                                             edge_server_bw[edgeIP])
        else:
            edge_server_transmission_time_for_each_client = (2 * total_model_size) / edge_server_bw[edgeIP]

        temp_time_for_each_client[client] = {'client_comp': comp_time_on_each_client[client][ClientOP1],
                                             'client_comm': transmission_time_on_each_client,
                                             'edge_server_comm': edge_server_transmission_time_for_each_client}
        if ClientOP1 == ClientOP2:
            temp_time_for_each_client[client]['edge_comp'] = 0
        if ClientOP2 == config.model_len - 1:
            temp_time_for_each_client[client]['server_comp'] = 0

    if (edge_memory is None) and (server_memory is None):
        if ((temp_time_for_each_client[changedClient]['client_comp'] +
             temp_time_for_each_client[changedClient]['client_comm'] +
             temp_time_for_each_client[changedClient]['edge_server_comm']) < baseline_tt):
            return False, 'NONE', None, None
        else:
            return True, 'NONE', None, None
    else:
        total_time_for_each_client = {}
        edgeIP = config.CLIENT_MAP[changedClient]

        if (edge_memory is not None) and (server_memory is None):
            neighbour_client_on_edge = [client for client in config.EDGE_MAP[edgeIP] if client in active_clients]
            load_tt_map, _, _, _ = edge_memory
            for neighbour_client in neighbour_client_on_edge:
                if (neighbour_client not in bad_clients) or (neighbour_client == changedClient):
                    clientIndex = config.CLIENTS_CONFIG[neighbour_client]
                    clientOP1 = currentAction[clientIndex][0]
                    clientOP2 = currentAction[clientIndex][1]
                    client_edge_comp = load_tt_map[f'{clientOP1},{clientOP2}']
                    temp_time_for_each_client[neighbour_client]['edge_comp'] = client_edge_comp

                    total_time_for_each_client[neighbour_client] = (temp_time_for_each_client[neighbour_client]['edge_comp'] +
                                                                    temp_time_for_each_client[neighbour_client]['client_comp'] +
                                                                    temp_time_for_each_client[neighbour_client]['client_comm'] +
                                                                    temp_time_for_each_client[neighbour_client]['edge_server_comm'])
                    if total_time_for_each_client[neighbour_client] > baseline_tt:
                        return True, 'EDGE', None, None
            return False, 'EDGE', temp_time_for_each_client, None

        elif (edge_memory is None) and (server_memory is not None):
            load_tt_map_on_server, _, _, _ = server_memory
            neighbour_client_on_server = active_clients

            for neighbour_client in neighbour_client_on_server:
                if (neighbour_client not in bad_clients) or (neighbour_client == changedClient):
                    clientIndex = config.CLIENTS_CONFIG[neighbour_client]
                    clientOP2 = currentAction[clientIndex][1]
                    client_server_comp = load_tt_map_on_server[f'{clientOP2}']
                    temp_time_for_each_client[neighbour_client]['server_comp'] = client_server_comp

                    total_time_for_each_client[neighbour_client] = (temp_time_for_each_client[neighbour_client]['server_comp'] +
                                                                    temp_time_for_each_client[neighbour_client]['client_comp'] +
                                                                    temp_time_for_each_client[neighbour_client]['client_comm'] +
                                                                    temp_time_for_each_client[neighbour_client]['edge_server_comm'])
                    if total_time_for_each_client[neighbour_client] > baseline_tt:
                        return True, 'SERVER', None, None
            return False, 'SERVER', temp_time_for_each_client, None

        elif (edge_memory is not None) and (server_memory is not None):
            neighbour_client_on_edge = [client for client in config.EDGE_MAP[edgeIP] if client in active_clients]
            neighbour_client_on_server = active_clients

            load_tt_map, _, _, _ = edge_memory
            load_tt_map_on_server, _, _, _ = server_memory

            for neighbour_client in neighbour_client_on_edge:
                clientIndex = config.CLIENTS_CONFIG[neighbour_client]
                clientOP1 = currentAction[clientIndex][0]
                clientOP2 = currentAction[clientIndex][1]
                client_edge_comp = load_tt_map[f'{clientOP1},{clientOP2}']
                temp_time_for_each_client[neighbour_client]['edge_comp'] = client_edge_comp

            for neighbour_client in active_clients:
                clientIndex = config.CLIENTS_CONFIG[neighbour_client]
                clientOP2 = currentAction[clientIndex][1]
                client_server_comp = load_tt_map_on_server[f'{clientOP2}']
                temp_time_for_each_client[neighbour_client]['server_comp'] = client_server_comp

            for modified_client in active_clients:
                total_time_for_each_client[modified_client] = sum(time_for_each_client[modified_client].values())

            for client in active_clients:
                if (client not in bad_clients) or (client == changedClient):
                    if total_time_for_each_client[client] > baseline_tt:
                        return True, 'BOTH', None, None
            return False, 'BOTH', temp_time_for_each_client, total_time_for_each_client
        else:
            raise Exception("Exception in checking TT violation")


def trainingTimeEstimator(action, comp_time_on_each_client, clients_bw, edge_server_bw, flops_of_each_layer,
                          activation_size, total_model_size, batchNumber, edge_flops_model, server_flops_model,
                          comp_time_on_edge_for_each_client=None, comp_time_on_server_for_each_client=None):
    edge_flops = {edgeIP: 0.0 for edgeIP in config.EDGE_SERVER_LIST}
    flop_of_each_edge_on_server = {edgeIP: 0.0 for edgeIP in config.EDGE_SERVER_LIST}
    server_flops = 0
    transmission_time_on_each_client = {}
    edge_server_transmission_time_for_each_client = {}
    total_time_for_each_client = {}
    time_for_each_client = {}
    for client in config.CLIENTS_LIST:
        time_for_each_client[client] = {}

    if comp_time_on_edge_for_each_client is None:
        comp_time_on_edge_for_each_client = {client: 0 for client, _ in clients_bw.items()}
    if comp_time_on_server_for_each_client is None:
        comp_time_on_server_for_each_client = {client: 0 for client, _ in clients_bw.items()}

    for i in range(len(action)):
        clientAction = action[i]
        clientIP = config.CLIENTS_INDEX[i]
        edgeIP = config.CLIENT_MAP[clientIP]
        op1 = clientAction[0]
        op2 = clientAction[1]

        if op1 != config.model_len - 1:
            transmission_time_on_each_client[clientIP] = ((2 * activation_size[op1] * batchNumber) + (
                    2 * total_model_size)) / clients_bw[clientIP]

        else:
            transmission_time_on_each_client[clientIP] = (2 * total_model_size) / clients_bw[clientIP]

        if op2 != config.model_len - 1:
            edge_server_transmission_time_for_each_client[clientIP] = ((2 * activation_size[op2] * batchNumber) + (
                    2 * total_model_size)) / edge_server_bw[edgeIP]
        else:
            edge_server_transmission_time_for_each_client[clientIP] = (2 * total_model_size) / edge_server_bw[edgeIP]

        # offloading on client, edge and server
        if op1 < op2 < config.model_len - 1:
            edge_flops[edgeIP] += sum(flops_of_each_layer[op1 + 1:op2 + 1])
            flop_of_each_edge_on_server[edgeIP] += sum(flops_of_each_layer[op2 + 1:])
            server_flops += sum(flops_of_each_layer[op2 + 1:])
        # offloading on client and edge
        elif (op1 < op2) and op2 == config.model_len - 1:
            edge_flops[edgeIP] += sum(flops_of_each_layer[op1 + 1:op2 + 1])
        # offloading on client and server
        elif (op1 == op2) and op1 < config.model_len - 1:
            server_flops += sum(flops_of_each_layer[op2 + 1:])
            flop_of_each_edge_on_server[edgeIP] += sum(flops_of_each_layer[op2 + 1:])

    EDGE_INDEX: list = [(edge, config.EDGE_SERVER_LIST.index(edge)) for edge in config.EDGE_SERVER_LIST]
    comp_time_on_each_edge = {edge: edge_flops_model.predict([[index, edge_flops[edge]]])[0] for edge, index in
                              EDGE_INDEX}
    comp_time_on_server = server_flops_model.predict([[server_flops]])
    # fed_logger.info(Fore.GREEN + f"Comp time on edges prediction: {comp_time_on_each_edge}")

    fed_logger.info(Fore.GREEN + f"Client's TT Approximation:")
    fed_logger.info(Fore.GREEN + f"Action: {action}")
    fed_logger.info(Fore.GREEN + f"===============================================================================")
    for clientIP in config.CLIENTS_LIST:
        op1 = action[config.CLIENTS_CONFIG[clientIP]][0]
        time_for_each_client[clientIP] = {'client_comp': comp_time_on_each_client[clientIP][op1],
                                          'client_comm': transmission_time_on_each_client[clientIP],
                                          'edge_server_comm': edge_server_transmission_time_for_each_client[
                                              clientIP],
                                          'edge_comp': comp_time_on_edge_for_each_client[clientIP],
                                          'server_comp': comp_time_on_server_for_each_client[clientIP]
                                          }
        total_time_for_each_client[clientIP] = sum(time_for_each_client[clientIP].values())

        fed_logger.info(Fore.GREEN + f"Client: {clientIP}:")
        fed_logger.info(Fore.GREEN + f"     Client Comp: {time_for_each_client[clientIP]['client_comp']}")
        fed_logger.info(Fore.GREEN + f"     Edge Comp: {time_for_each_client[clientIP]['edge_comp']}")
        fed_logger.info(Fore.GREEN + f"     Server Comp: {time_for_each_client[clientIP]['server_comp']}")
        fed_logger.info(Fore.GREEN + f"     Client Comm: {time_for_each_client[clientIP]['client_comm']}")
        fed_logger.info(Fore.GREEN + f"     Edge-Server Comm: {time_for_each_client[clientIP]['edge_server_comm']}")
        fed_logger.info(Fore.GREEN + f"     SUM: {total_time_for_each_client[clientIP]}")
        fed_logger.info(Fore.GREEN + f"---------------------------------------------------------------------------")

    NICE_MIN = -20
    NICE_MAX = 19

    normalized_total_time_of_clients = normalizer(total_time_for_each_client)

    def map_to_nice(normalized_value, nice_min, nice_max):
        # Invert normalized value (slower process gets lower nice value)
        inverted_value = 1 - normalized_value
        return int(inverted_value * (nice_max - nice_min) + nice_min)

    nice_values = {client: map_to_nice(norm_time, NICE_MIN, NICE_MAX) for client, norm_time in
                   normalized_total_time_of_clients.items()}

    total_trainingTime = max(total_time_for_each_client.values())
    fed_logger.info(Fore.GREEN + f"Action's Training Time: {total_trainingTime}")

    return (total_trainingTime, total_time_for_each_client, time_for_each_client, nice_values, comp_time_on_server,
            comp_time_on_each_edge, transmission_time_on_each_client, edge_server_transmission_time_for_each_client)


def triedBefore(current_splitting, edgeName=None):
    """
    check if we decided this splitting before or not

    Args:
        current_splitting: current splitting that we want to check in memory
        edgeName: the name of the edge

    Returns:
        dict, dict, dict: if we found any previous action similar to cuurent splitting and None if we do not.

    Example:
        >>> triedBefore([[1,2]], 'edge1')
    """

    memory = load_memory('/fed-flow/app/model/memory.json')['history']
    load_time_map = {}

    # check memory for similar load on server
    if edgeName is None:
        current_op2s = []
        for clientsSplitting in current_splitting:
            op2 = clientsSplitting[1]
            if op2 != config.model_len - 1:
                current_op2s.append(op2)
        for item in memory:
            memory_op2s = []
            splitting = item['splitting']
            for clientsSplitting in splitting:
                op2 = clientsSplitting[1]
                if op2 != config.model_len - 1:
                    memory_op2s.append(op2)
            if sorted(memory_op2s) == sorted(current_op2s):
                currentSplitting_op2 = [client_splitting[1] for client_splitting in current_splitting]
                memorySplitting_op2 = [client_splitting[1] for client_splitting in splitting]
                value_to_index = {value: idx for idx, value in enumerate(memorySplitting_op2)}
                matched_device_index = {i: value_to_index[val] for i, val in enumerate(currentSplitting_op2) if val != config.model_len - 1}
                client_info = item['clientInfo']
                edges_info = item['edgeInfo']
                server_info = item['serverInfo']
                for clientIndex in range(len(current_splitting)):
                    if current_splitting[clientIndex][1] != config.model_len - 1:
                        load_time_map[f'{current_splitting[clientIndex][1]}'] = \
                            client_info[config.CLIENTS_INDEX[matched_device_index[clientIndex]]]['serverCompTime']
                    else:
                        load_time_map[f'{current_splitting[clientIndex][1]}'] = 0
                return load_time_map, client_info, edges_info, server_info
        return None
    elif edgeName in config.EDGE_SERVER_LIST:
        edges_client = config.EDGE_MAP[edgeName]
        clients_index = []
        for client in edges_client:
            if client in config.CLIENTS_LIST:
                clients_index.append(config.CLIENTS_CONFIG[client])
        clients_index = sorted(clients_index)
        edges_splitting = []
        for clientIndex in clients_index:
            if current_splitting[clientIndex][0] != current_splitting[clientIndex][1]:
                edges_splitting.append(current_splitting[clientIndex])
        for item in memory:
            memory_splitting = item['splitting']
            memory_edges_splitting = []
            for client_index in clients_index:
                if memory_splitting[client_index][0] != memory_splitting[client_index][1]:
                    memory_edges_splitting.append(memory_splitting[client_index])

            if Counter(map(tuple, edges_splitting)) == Counter(map(tuple, memory_edges_splitting)):
                fed_logger.info(Fore.GREEN + f"CURRENT SPLITTING ON EDGE: {memory_edges_splitting}")
                fed_logger.info(Fore.GREEN + f"MATCHED SPLITTING: {memory_splitting}")
                client_info = item['clientInfo']
                edges_info = item['edgeInfo']
                server_info = item['serverInfo']

                for load in memory_edges_splitting:
                    clientID_with_this_load = [i for i, val in enumerate(memory_splitting) if val == load]
                    times_with_this_load = []
                    for clientID in clientID_with_this_load:
                        if clientID in clients_index:
                            times_with_this_load.append(client_info[config.CLIENTS_INDEX[clientID]]['edgeCompTime'])
                    load_time_map[f'{load[0]},{load[1]}'] = sum(times_with_this_load) / len(times_with_this_load)

                for index in clients_index:
                    if current_splitting[index][0] == current_splitting[index][1]:
                        load_time_map[f'{current_splitting[index][0]},{current_splitting[index][1]}'] = 0

                fed_logger.info(Fore.GREEN + f"CLIENT INFO: {client_info}")

                fed_logger.info(Fore.GREEN + f"LOAD TIME MAP: {load_time_map}")

                return load_time_map, client_info, edges_info, server_info
        return None
    else:
        raise Exception("Invalid argument for triedBefore Function")


def load_memory(memory_path):
    try:
        with open(memory_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"history": []}  # Return empty memory if file doesn't exist


#
#
# def compareCurrentSplittingWithMemory(current_splitting: list, client_remaining_energy, flop_of_each_layer,
#                                       compare_aspect: str, edge_name: str = None):
#     memory = load_memory(memory_path='/fed-flow/app/model/memory.json')['history']
#
#     current_load_of_each_client_on_edge = {}
#     current_total_load_on_edge = 0
#     if compare_aspect == 'edge' and edge_name is None:
#         for clientIP in config.EDGE_MAP[edge_name]:
#             if client_remaining_energy[clientIP] > 1:
#                 op1 = current_splitting[config.CLIENTS_CONFIG[clientIP]][0]
#                 op2 = current_splitting[config.CLIENTS_CONFIG[clientIP]][1]
#                 current_load_of_each_client_on_edge[clientIP] = sum(flop_of_each_layer[op1 + 1: op2 + 1])
#         current_total_load_on_edge = sum(current_load_of_each_client_on_edge.values())
#
#     similar_splitting_in_memory = []
#     memory_load_of_each_client_on_edge = {}
#     for item in memory:
#         memory_splitting = item['splitting']
#         client_info = item['client_info']
#         edges_info = item['edge_info']
#         server_info = item['server_info']
#
#         if compare_aspect == 'edge':
#             if edge_name is None:
#                 raise Exception('Edge name cannot be None')
#             memory_edge_load = edges_info[edge_name]['flopOnEdge']
#             for clientIP in config.EDGE_MAP[edge_name]:
#                 op1 = memory_splitting[config.CLIENTS_CONFIG[clientIP]][0]
#                 op2 = memory_splitting[config.CLIENTS_CONFIG[clientIP]][1]
#                 memory_load_of_each_client_on_edge[clientIP] = sum(flop_of_each_layer[op1 + 1: op2 + 1])


def are_times_close_enough(times, similarity_threshold=0.2):
    """
    Determine if all client execution times are close enough to each other
    that no priority changes are needed.

    Returns:
        bool: True if times are close enough, False if significant differences exist
    """

    # Filter out clients with no history
    valid_times = [v for v in times.values() if v != 0]

    if len(valid_times) < 2:
        return True  # Not enough data to determine

    # Calculate coefficient of variation (CV)
    mean_time = np.mean(valid_times)
    std_time = np.std(valid_times)

    # Avoid division by zero
    if mean_time == 0:
        return True

    cv = std_time / mean_time

    # If CV is below threshold, times are considered close enough
    return cv < similarity_threshold


def detect_outliers_using_z_score(times, outlier_threshold=1.5):
    """
    Detect outlier clients using Z-score method
    Returns: tuple of (high_outliers, low_outliers) as lists of client IDs
    """
    # Filter out clients with no history
    valid_clients = {k: v for k, v in times.items() if v != 0}

    if len(valid_clients) < 2:
        return [], []  # Not enough data

    # Calculate mean and std of average times
    values = list(valid_clients.values())
    mean_time = np.mean(values)
    std_time = np.std(values)

    # Avoid division by zero
    if std_time == 0:
        return [], []

    high_outliers = []
    low_outliers = []

    # Calculate z-scores and identify outliers
    for client_id, avg_time in valid_clients.items():
        z_score = (avg_time - mean_time) / std_time

        if z_score > outlier_threshold:
            high_outliers.append(client_id)
        elif z_score < -outlier_threshold:
            low_outliers.append(client_id)

    return high_outliers, low_outliers


def detect_outliers_using_iqr(times):
    """
    Detect outlier clients using IQR method
    Returns: tuple of (high_outliers, low_outliers) as lists of client IDs
    """

    # Filter out clients with no history
    valid_clients = {k: v for k, v in times.items() if v != 0}

    if len(valid_clients) < 4:  # Need enough data for quartiles
        return detect_outliers_using_z_score(times)  # Fall back to z-score

    values = list(valid_clients.values())
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    # Define bounds for outliers
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    high_outliers = [client_id for client_id, avg_time in valid_clients.items()
                     if avg_time > upper_bound]
    low_outliers = [client_id for client_id, avg_time in valid_clients.items()
                    if avg_time < lower_bound]

    return high_outliers, low_outliers
