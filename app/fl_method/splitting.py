import copy
import json
import random

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
    for client in state['client_bw'].keys():
        min_energy_splitting_for_each_client[client] = []
        min_time_splitting_for_each_client[client] = []
        filtered_min_time_splitting_for_each_client[client] = []
        min_energy_trainingTime_splitting_for_each_client[client] = []

    MODEL_PATH = '/fed-flow/app/model'
    edge_linear_model = joblib.load(f"{MODEL_PATH}/edge_flops_prediction_linear_model.pkl")
    edge_poly_model = joblib.load(f"{MODEL_PATH}/edge_flops_prediction_poly_model.pkl")

    server_linear_model = joblib.load(f"{MODEL_PATH}/server_flops_prediction_linear_model.pkl")
    server_poly_model = joblib.load(f"{MODEL_PATH}/server_flops_prediction_linear_model.pkl")
    fed_logger.info(Fore.MAGENTA + f"Edge and server flops prediction model loaded =======================")

    client_remaining_runtime = {}
    previous_action = state['prev_action']

    total_model_size = state['total_model_size']
    activation_size = state['activation_size']
    gradient_size = state['gradient_size']
    flops_of_each_layer = state['flops_of_each_layer']
    flops_of_each_layer = {key: flops_of_each_layer[key] for key in sorted(flops_of_each_layer)}
    flops_of_each_layer = list(flops_of_each_layer.values())
    fed_logger.info(Fore.MAGENTA + f"Total flops: {flops_of_each_layer}")

    client_bw = state['client_bw']
    edge_server_bw = state['edge_bw']

    client_comp_energy = state['client_comp_energy']
    client_comm_energy = state['client_comm_energy']
    client_utilization = state['client_utilization']
    client_remaining_energy = state['client_remaining_energy']
    client_power_usage = state['client_power']
    client_comp_time = state['client_comp_time']
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
    for client in state['client_bw'].keys():
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
    score_estimation_action = [[4, config.model_len - 1] for client in config.CLIENTS_CONFIG.keys()]

    each_splitting_share = {op1: {op2: {} for op2 in range(op1, config.model_len)} for op1 in range(config.model_len)}
    max_computation_on_client = sum(flops_of_each_layer)
    max_computation_on_edge_and_server = sum(flops_of_each_layer[1:])
    max_comm = 2 * batchNumber * max(activation_size.values()) + total_model_size

    for op1 in range(config.model_len):
        for op2 in range(op1, config.model_len):
            each_splitting_share[op1][op2] = {
                'client_comp': sum(flops_of_each_layer[:op1 + 1]) / max_computation_on_client,
                'edge_comp': sum(flops_of_each_layer[op1 + 1:op2 + 1]) / max_computation_on_edge_and_server,
                'server_comp': sum(flops_of_each_layer[op2 + 1:]) / max_computation_on_edge_and_server,
                'client_comm': (2 * batchNumber * activation_size[
                    op1]) + total_model_size / max_comm if (
                        op1 != config.model_len - 1) else total_model_size / max_comm,
                'edge_server_comm': (2 * batchNumber * activation_size[
                    op2]) + total_model_size / max_comm if op2 != config.model_len - 1 else total_model_size / max_comm,
            }

    clients_computation_e, clients_communication_e, clients_totals_e = energyEstimator(score_estimation_action,
                                                                                       client_bw, activation_size,
                                                                                       batchNumber, total_model_size,
                                                                                       client_comp_energy,
                                                                                       client_power_usage)
    for client in client_remaining_energy.keys():
        client_remaining_runtime[client] = client_remaining_energy[client] / (clients_totals_e[client])

    client_remaining_runtime_comp_score = normalizer(client_remaining_runtime)
    client_score = client_remaining_runtime_comp_score

    high_prio_device = {device: score for device, score in client_score.items() if score <= 1.0}

    fed_logger.info(f"RUN TIME SCORE: {client_remaining_runtime_comp_score.items()}")
    fed_logger.info(f"HIGH PRIO DEVICES: {high_prio_device.items()}")

    classicFL_action = [[config.model_len - 1, config.model_len - 1] for _ in range(len(config.CLIENTS_CONFIG))]
    classicFL_tt, _, _, _, _, _, _, _ = trainingTimeEstimator(classicFL_action, client_comp_time, client_bw,
                                                              edge_server_bw,
                                                              flops_of_each_layer, activation_size, total_model_size,
                                                              batchNumber,
                                                              edge_poly_model, server_poly_model,
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
    for client, score in client_score.items():
        fed_logger.info(Fore.GREEN + f"Client: {client}")

        client_op1_energy = []
        client_op1_time = []
        for layer in range(config.model_len):
            op1 = layer
            if layer != config.model_len - 1:
                size = activation_size[layer]
                tt_trans = ((2 * size * batchNumber) / client_bw[client]) + (total_model_size / client_bw[client])
            else:
                tt_trans = total_model_size / client_bw[client]
            comm_energy = tt_trans * client_power_usage[client][1]
            comp_energy = client_comp_energy[client][op1]
            total_energy = comp_energy + comm_energy
            total_time = client_comp_time[client][op1] + tt_trans
            client_op1_energy.append((op1, total_energy))
            client_op1_time.append((op1, total_time))
        min_energy_splitting_for_each_client[client] = sorted(client_op1_energy, key=lambda x: x[1])
        min_time_splitting_for_each_client[client] = sorted(client_op1_time, key=lambda x: x[1])
        filtered_min_time_splitting_for_each_client[client] = [item for item in
                                                               min_time_splitting_for_each_client[client] if
                                                               item[1] <= classicFL_tt]
        for op1, energy in min_energy_splitting_for_each_client[client]:
            if any(item[0] == op1 for item in filtered_min_time_splitting_for_each_client[client]):
                min_energy_trainingTime_splitting_for_each_client[client].append((op1, energy))

        fed_logger.info(Fore.GREEN + f"   Energy[Ascending]: {min_energy_splitting_for_each_client[client]}")
        fed_logger.info(Fore.GREEN + f"   Time[Ascending]: {min_time_splitting_for_each_client[client]}")
        fed_logger.info(
            Fore.GREEN + f"   Energy-TrainingTime[Ascending]: {min_energy_trainingTime_splitting_for_each_client[client]}")
        fed_logger.info(Fore.GREEN + f"---------------------------------------------------------------------------")

    splitting_score = 0
    splitting_energy_score = 0
    splitting_time_score = 0
    best_score = -10
    best_action = None

    satisfied = False

    if not (ALPHA == 0 or ALPHA == 1):
        for client in config.CLIENTS_CONFIG.keys():
            if satisfied:
                break
            for op1 in range(config.model_len - 1):
                if satisfied:
                    break
                for op2 in range(op1, config.model_len - 1):
                    action[config.CLIENTS_CONFIG[client]] = [op1, op2]
                    training_time_of_action, _, _, _, _, _, _, _ = trainingTimeEstimator(action, client_comp_time,
                                                                                         client_bw,
                                                                                         edge_server_bw,
                                                                                         flops_of_each_layer,
                                                                                         activation_size,
                                                                                         total_model_size,
                                                                                         batchNumber, edge_poly_model,
                                                                                         server_poly_model)
                    clients_comp_e_of_action, clients_comm_e_of_action, clients_total_e_of_action = (
                        energyEstimator(action,
                                        client_bw,
                                        activation_size,
                                        batchNumber,
                                        total_model_size,
                                        client_comp_energy,
                                        client_power_usage))
                    avg_e_of_action = sum(clients_total_e_of_action.values()) / len(clients_total_e_of_action.values())

                    if avg_e_of_action > classicFL_avg_energy:
                        splitting_energy_score = (classicFL_avg_energy / avg_e_of_action) - 1
                    else:
                        splitting_energy_score = 1 - (avg_e_of_action / classicFL_avg_energy)

                    if training_time_of_action > classicFL_tt:
                        splitting_time_score = (classicFL_tt / training_time_of_action) - 1
                    else:
                        splitting_time_score = 1 - (training_time_of_action / classicFL_tt)

                    splitting_score = ALPHA * splitting_energy_score + (1 - ALPHA) * splitting_time_score

                    if splitting_score >= score_threshold:
                        satisfied = True
                        best_action = action
                        break
                    if splitting_score > best_score:
                        best_score = splitting_score
                        best_action = action
    elif ALPHA == 1:
        baseline_tt = classicFL_tt
        baseline_energy = classicFL_avg_energy

        best_score_client_index = 0
        best_layer_index = 1
        action_tt_and_baseline_tt_dif = 0
        action_e_and_baseline_e_diff = 0

        best_energy_action = action
        best_tt_action = None

        clients_score = sorted(client_score.items(), key=lambda item: item[1], reverse=True)
        fed_logger.info(Fore.MAGENTA + f"Current Round: {config.current_round}, model_len: {config.model_len}")

        if config.current_round == config.model_len:
            action = [[op1s[0][0], config.model_len - 1] for client, op1s in
                      min_energy_trainingTime_splitting_for_each_client.items()]
            return action, 0, 0, 0
        best_action_found = None
        best_score_found = 0
        satisfied = False
        notFound = False

        prev_action_tt, prev_action_each_client_total_tt, prev_action_each_client_tt, _, _, _, _, _ = (
            trainingTimeEstimator(previous_action, client_comp_time, client_bw, edge_server_bw, flops_of_each_layer,
                                  activation_size, total_model_size, batchNumber, edge_poly_model, server_poly_model,
                                  comp_time_of_each_client_on_edges, comp_time_of_each_client_on_server))

        fed_logger.info(Fore.MAGENTA + f"Prev Action tt: {prev_action_tt}, Baseline tt: {baseline_tt}")
        fed_logger.info(Fore.MAGENTA + f"Action's training time: {prev_action_tt}")
        fed_logger.info(Fore.MAGENTA + f"Action's training time per client: {prev_action_each_client_total_tt}")
        fed_logger.info(Fore.MAGENTA + f"Action's training time per client[per section]: {prev_action_each_client_tt}")
        baseline_tt *= 1.05
        if prev_action_tt <= baseline_tt:
            return previous_action, 0, 0, 0

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
        # finding slow part of training of bad device and try to solve it
        for client in config.CLIENTS_LIST:
            isBadClient = True if prev_action_each_client_total_tt[client] > baseline_tt else False
            op1 = previous_action[config.CLIENTS_CONFIG[client]][0]
            op2 = previous_action[config.CLIENTS_CONFIG[client]][1]
            edgeIP = config.CLIENT_MAP[client]

            time_diff = prev_action_each_client_total_tt[client] - baseline_tt

            client_comp_share = (
                    prev_action_each_client_tt[client]['client_comp'] / prev_action_each_client_total_tt[client])
            edge_comp_share = (
                    prev_action_each_client_tt[client]['edge_comp'] / prev_action_each_client_total_tt[client])
            server_comp_share = (
                    prev_action_each_client_tt[client]['server_comp'] / prev_action_each_client_total_tt[client])
            client_comm_share = (
                    prev_action_each_client_tt[client]['client_comm'] / prev_action_each_client_total_tt[client])
            edge_server_comm_share = (
                    prev_action_each_client_tt[client]['edge_server_comm'] / prev_action_each_client_total_tt[client])

            shares[client] = {'client_comp': 0,
                              'edge_comp': edge_comp_share,
                              'server_comp': server_comp_share,
                              'client_comm': 0,
                              'edge_server_comm': edge_server_comm_share}
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
        for edgeCluster in bad_clients_cluster.keys():
            oneClientChanged = False
            # for client in bad_clients_cluster[edgeCluster]['client_comp']:
            #     currentOffloading = previous_action[config.CLIENTS_CONFIG[client]]
            #     currentOffloadingShares = each_splitting_share[currentOffloading[0]][currentOffloading[1]]
            #     time_diff = prev_action_each_client_total_tt[client] - baseline_tt
            #
            #     if prev_action_each_client_tt[client]['edge_comp'] - prev_action_each_client_tt[client][
            #         'server_comp'] > time_diff:
            #         sorted_op2_by_edge_comp = sorted(each_splitting_share[currentOffloading[0]].items(),
            #                                          key=lambda item: item[1]['edge_comp'], reverse=True)
            #         for op2, share in sorted_op2_by_edge_comp:
            #             if share['edge_comp'] < currentOffloadingShares['edge_comp']:
            #                 new_action[config.CLIENTS_CONFIG[client]][1] = op2
            #
            #     for optimalOp1, energy in min_energy_splitting_for_each_client[client]:
            #         if optimalOp1 < currentOffloading[0]:
            #             new_action[config.CLIENTS_CONFIG[client]] = [optimalOp1, config.model_len - 1]
            for client in bad_clients_cluster[edgeCluster]['edge_comp']:
                time_diff = prev_action_each_client_total_tt[client] - baseline_tt
                currentOffloading = previous_action[config.CLIENTS_CONFIG[client]]
                currentOffloadingShares = each_splitting_share[currentOffloading[0]][currentOffloading[1]]
                if prev_action_each_client_tt[client]['edge_comp'] - prev_action_each_client_tt[client][
                    'server_comp'] > time_diff:
                    sorted_op2_by_edge_comp = sorted(each_splitting_share[currentOffloading[0]].items(),
                                                     key=lambda item: item[1]['edge_comp'], reverse=True)
                    for op2, share in sorted_op2_by_edge_comp:
                        if share['edge_comp'] < currentOffloadingShares['edge_comp']:
                            temp = copy.deepcopy(new_action)
                            temp[config.CLIENTS_CONFIG[client]][1] = op2
                            memory = triedBefore(temp)
                            if memory is not None:
                                client_info, edge_info, server_info = memory
                                if client_info[client]['edge_comp'] + client_info[client]['edge_server_comm'] + \
                                        client_info[client]['server_comp'] < \
                                        prev_action_each_client_tt[client]['edge_comp'] + \
                                        prev_action_each_client_tt[client]['server_comp'] + \
                                        prev_action_each_client_tt[client]['edge_server_comm']:
                                    new_action[config.CLIENTS_CONFIG[client]][1] = op2
                            else:
                                new_action[config.CLIENTS_CONFIG[client]][1] = op2
                            oneClientChanged = True
                            break

                if oneClientChanged:
                    break
            if not oneClientChanged:
                for client in bad_clients_cluster[edgeCluster]['server_comp']:
                    currentOffloading = previous_action[config.CLIENTS_CONFIG[client]]
                    currentOffloadingShares = each_splitting_share[currentOffloading[0]][currentOffloading[1]]
                    sorted_op2_by_server_comp = sorted(each_splitting_share[currentOffloading[0]].items(),
                                                       key=lambda item: item[1]['server_comp'], reverse=True)
                    for op2, share in sorted_op2_by_server_comp:
                        if share['server_comp'] < currentOffloadingShares['server_comp']:
                            temp = copy.deepcopy(new_action)
                            temp[config.CLIENTS_CONFIG[client]][1] = op2
                            memory = triedBefore(temp)
                            if memory is not None:
                                client_info, edge_info, server_info = memory
                                if client_info[client]['edge_comp'] + client_info[client]['edge_server_comm'] + \
                                        client_info[client]['server_comp'] < \
                                        prev_action_each_client_tt[client]['edge_comp'] + \
                                        prev_action_each_client_tt[client]['server_comp'] + \
                                        prev_action_each_client_tt[client]['edge_server_comm']:
                                    new_action[config.CLIENTS_CONFIG[client]][1] = op2
                            else:
                                new_action[config.CLIENTS_CONFIG[client]][1] = op2
                            oneClientChanged = True
                            break
                    if oneClientChanged:
                        break
            if not oneClientChanged:
                for client in bad_clients_cluster[edgeCluster]['edge_server_comm']:
                    currentOffloading = previous_action[config.CLIENTS_CONFIG[client]]
                    currentOffloadingShares = each_splitting_share[currentOffloading[0]][currentOffloading[1]]
                    sorted_op2_by_edge_server_comm = sorted(each_splitting_share[currentOffloading[0]].items(),
                                                            key=lambda item: item[1]['edge_server_comm'], reverse=True)
                    for op2, share in sorted_op2_by_edge_server_comm:
                        if share['edge_server_comm'] < currentOffloadingShares['edge_server_comm']:
                            new_action[config.CLIENTS_CONFIG[client]][1] = op2
            if oneClientChanged:
                continue

        return new_action, 0, 0, 0

    elif ALPHA == 0:
        action = [[op1s[0][0], config.model_len - 1] for client, op1s in min_time_splitting_for_each_client.items()]
        for client in config.CLIENTS_CONFIG.keys():
            best_op1 = min_time_splitting_for_each_client[client][0][0]
            for op2 in range(best_op1, config.model_len - 1):
                action[config.CLIENTS_CONFIG[client]] = [best_op1, op2]
                training_time_of_action, _, _, _, _, _, _, _ = trainingTimeEstimator(action, client_comp_time,
                                                                                     client_bw,
                                                                                     edge_server_bw,
                                                                                     flops_of_each_layer,
                                                                                     activation_size,
                                                                                     total_model_size,
                                                                                     batchNumber, edge_poly_model,
                                                                                     server_poly_model)
                if training_time_of_action < best_tt:
                    best_tt = training_time_of_action
                    best_action = action
    return action


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
        split_layer.append(int(idx))
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
            tt_trans = (2 * (activation_size[client_op1]) * batchNumber) / client_bw[client]
        else:
            tt_trans = total_model_size / client_bw[client]
        client_computation_energy[client] = client_comp_energy[client][client_op1]
        client_communication_energy[client] = tt_trans * client_power_usage[client][1]
        client_total_energy[client] = client_communication_energy[client] + client_computation_energy[client]

    return client_computation_energy, client_communication_energy, client_total_energy


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
    for client in config.CLIENTS_CONFIG.keys():
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
            transmission_time_on_each_client[clientIP] = ((2 * (activation_size[op1]) * batchNumber)
                                                          / clients_bw[clientIP]) + (
                                                                 (2 * total_model_size) / clients_bw[clientIP])
        else:
            transmission_time_on_each_client[clientIP] = 2 * (total_model_size / clients_bw[clientIP])

        if op2 != config.model_len - 1:
            edge_server_transmission_time_for_each_client[clientIP] = ((2 * (activation_size[op2]) * batchNumber) /
                                                                       edge_server_bw[edgeIP]) + (
                                                                              (2 * total_model_size) / edge_server_bw[
                                                                          edgeIP])
        else:
            edge_server_transmission_time_for_each_client[clientIP] = 2 * (total_model_size / edge_server_bw[edgeIP])

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
    for clientIP in config.CLIENTS_CONFIG.keys():
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


def triedBefore(current_splitting):
    memory = load_memory('/fed-flow/app/model/memory.json')['history']
    for item in memory:
        memory_splitting = item['splitting']
        if memory_splitting == current_splitting:
            client_info = item['clientInfo']
            edges_info = item['edgeInfo']
            server_info = item['serverInfo']
            return client_info, edges_info, server_info
    return None


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
