import random

import joblib
import numpy as np
from colorama import Fore

from app.config import config
from app.config.logger import fed_logger
# from app.model.entity.rl_model import PPO
from app.util import model_utils


# from stable_baselines3 import PPO, DDPG


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

    # ALPHA == 1 try minimum energy and ALPHA == 0 is vice versa
    ALPHA = 1
    score_threshold = 0.5

    candidate_splitting = []
    min_energy_splitting_for_each_client = {}
    min_time_splitting_for_each_client = {}
    for client in state['client_bw'].keys():
        min_energy_splitting_for_each_client[client] = []
        min_time_splitting_for_each_client[client] = []

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

    comp_time_of_each_client_on_edges = state['comp_time_of_each_client_on_edge']
    comp_time_of_each_client_on_server = state['comp_time_of_each_client_on_server']
    total_time_on_each_edge = {edgeIP: 0 for edgeIP in config.EDGE_SERVER_LIST}
    total_time_on_server = sum(comp_time_of_each_client_on_server.values())
    action = previous_action

    batchNumber = (config.N / len(config.CLIENTS_CONFIG.keys())) / config.B

    clients_computation_e, clients_communication_e, clients_totals_e = energyEstimator(previous_action, client_bw,
                                                                                       activation_size, batchNumber,
                                                                                       total_model_size,
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
    classicFL_tt, _ = trainingTimeEstimator(classicFL_action, client_comp_time, client_bw, edge_server_bw,
                                            flops_of_each_layer, activation_size, total_model_size, batchNumber,
                                            edge_poly_model, server_poly_model)

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
    for client, score in high_prio_device.items():
        client_op1 = currentAction[config.CLIENTS_CONFIG[client]][0]
        client_op2 = currentAction[config.CLIENTS_CONFIG[client]][1]
        best_op1 = client_op1
        best_op2 = client_op2
        if client_op1 != config.model_len - 1:
            tt_trans_now = (2 * (activation_size[client_op1]) * batchNumber) / client_bw[client]
        else:
            tt_trans_now = total_model_size / client_bw[client]

        comm_energy = tt_trans_now * client_power_usage[client][1]
        client_comm_energy_ratio = comm_energy / (comm_energy + client_comp_energy[client][client_op1])
        client_comp_energy_ratio = 1 - client_comm_energy_ratio

        fed_logger.info(Fore.GREEN + f"====================================================")
        fed_logger.info(Fore.GREEN + f"CLIENT: {client}")
        fed_logger.info(Fore.GREEN + f"Activation Layer: {activation_size}")
        fed_logger.info(Fore.GREEN + f"BW : {client_bw[client]}")
        fed_logger.info(Fore.GREEN + f"Approx. transmission time : {tt_trans_now}")
        fed_logger.info(Fore.GREEN + f"Approx. communication energy : {comm_energy}")
        fed_logger.info(Fore.GREEN + f"Computation energy : {client_comp_energy[client][client_op1]}")
        fed_logger.info(Fore.GREEN + f"Computation energy RATIO : {client_comp_energy_ratio}")
        fed_logger.info(Fore.GREEN + f"Communication Energy RATIO: {client_comm_energy_ratio}")

        min_total_energy = tt_trans_now * client_power_usage[client][1] + client_comp_energy[client][client_op1]
        client_op1_energy = []
        client_op1_time = []
        fed_logger.info(Fore.GREEN + f"CLIENTS COMP ENERGY: {client_comp_energy}")

        for layer, size in activation_size.items():
            candidate_op1 = layer
            if layer != config.model_len - 1:
                tt_trans = (2 * size * batchNumber) / client_bw[client]
            else:
                tt_trans = total_model_size / client_bw[client]
            comm_energy = tt_trans * client_power_usage[client][1]
            if candidate_op1 in client_comp_energy[client]:
                comp_energy = client_comp_energy[client][candidate_op1]
                total_energy = comp_energy + comm_energy
                client_op1_energy.append((candidate_op1, total_energy))
                client_op1_time.append((candidate_op1, client_comp_time[client][candidate_op1] + tt_trans))
                if total_energy < min_total_energy:
                    min_total_energy = comp_energy + comm_energy
                    best_op1 = candidate_op1
            else:
                best_op1 = candidate_op1
                break
            action[config.CLIENTS_CONFIG[client]] = [best_op1, 6]
        min_energy_splitting_for_each_client[client] = sorted(client_op1_energy, key=lambda x: x[1])
        min_time_splitting_for_each_client[client] = sorted(client_op1_time, key=lambda x: x[1])
        fed_logger.info(Fore.MAGENTA + f"Splitting-energy map: {min_energy_splitting_for_each_client}")
        fed_logger.info(Fore.MAGENTA + f"Splitting-time map: {min_time_splitting_for_each_client}")

    splitting_score = 0
    splitting_energy_score = 0
    splitting_time_score = 0
    best_score = -10
    best_action = None

    op1_pointers = {client: 0 for client, _ in config.CLIENTS_CONFIG.items()}
    op2_pointers = {client: 0 for client, _ in config.CLIENTS_CONFIG.items()}

    satisfied = False

    if ALPHA != 0 or ALPHA != 1:
        for client in config.CLIENTS_CONFIG.keys():
            if satisfied:
                break
            for op1 in range(config.model_len - 1):
                if satisfied:
                    break
                for op2 in range(op1, config.model_len - 1):
                    action[config.CLIENTS_CONFIG[client]] = [op1, op2]
                    training_time_of_action, _ = trainingTimeEstimator(action, client_comp_time, client_bw,
                                                                       edge_server_bw,
                                                                       flops_of_each_layer, activation_size,
                                                                       total_model_size,
                                                                       batchNumber, edge_poly_model, server_poly_model)
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
        not_found = False
        action = [[op1s[0][0], config.model_len - 1] for client, op1s in
                  min_energy_splitting_for_each_client.items()]
        best_score_client_index = 0
        best_layer_index = 0
        action_tt_and_baseline_tt_dif = 0
        action_e_and_baseline_e_diff = 0
        best_energy_action = None
        best_tt_action = None

        while not satisfied and not not_found:
            for client in config.CLIENTS_CONFIG.keys():
                if satisfied or not_found:
                    break
                client_op1 = action[config.CLIENTS_CONFIG[client]][0]
                for op2 in range(client_op1, config.model_len - 1):
                    action[config.CLIENTS_CONFIG[client]][1] = op2
                    training_time_of_action, _ = trainingTimeEstimator(action, client_comp_time, client_bw,
                                                                       edge_server_bw, flops_of_each_layer,
                                                                       activation_size, total_model_size, batchNumber,
                                                                       edge_poly_model, server_poly_model)
                    clients_comp_e_of_action, clients_comm_e_of_action, clients_total_e_of_action = (
                        energyEstimator(action,
                                        client_bw,
                                        activation_size,
                                        batchNumber,
                                        total_model_size,
                                        client_comp_energy,
                                        client_power_usage))
                    avg_e_of_action = sum(clients_total_e_of_action.values()) / len(clients_total_e_of_action.values())

                    if (avg_e_of_action - baseline_energy) < action_e_and_baseline_e_diff:
                        best_energy_action = action
                        action_e_and_baseline_e_diff = avg_e_of_action - baseline_energy
                        if training_time_of_action < baseline_tt:
                            best_action = action
                            satisfied = True
                            break

                    if (training_time_of_action - baseline_tt) < action_tt_and_baseline_tt_dif:
                        best_tt_action = action
                        action_tt_and_baseline_tt_dif = training_time_of_action - baseline_tt

                    if avg_e_of_action > baseline_energy:
                        not_found = True
                        break

            clients_score = sorted(client_score.items(), key=lambda item: item[1], reverse=True)
            if best_score_client_index < len(config.CLIENTS_CONFIG.keys()):
                best_score_client = clients_score[best_score_client_index][0]  # client_score[0] = ('client1', 1.0)
            if best_layer_index + 1 <= config.model_len - 1:
                action[config.CLIENTS_CONFIG[best_score_client]] = [
                    min_energy_splitting_for_each_client[best_score_client][best_layer_index + 1][0],
                    config.model_len - 1]
                best_layer_index += 1
            else:
                if best_score_client_index < len(config.CLIENTS_CONFIG.keys()) - 1:
                    best_score_client_index += 1
                    best_layer_index = 0
        if not_found:
            return best_energy_action
        else:
            return best_action

    elif ALPHA == 0:
        action = [[op1s[0][0], config.model_len - 1] for client, op1s in min_time_splitting_for_each_client.items()]
        for client in config.CLIENTS_CONFIG.keys():
            best_op1 = min_time_splitting_for_each_client[client][0][0]
            for op2 in range(best_op1, config.model_len - 1):
                action[config.CLIENTS_CONFIG[client]] = [best_op1, op2]
                training_time_of_action, _ = trainingTimeEstimator(action, client_comp_time, client_bw, edge_server_bw,
                                                                   flops_of_each_layer, activation_size,
                                                                   total_model_size,
                                                                   batchNumber, edge_poly_model, server_poly_model)
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


# def rl_splitting(state, labels):
#     state_dim = 2 * config.G
#     action_dim = config.G
#     agent = None
#     if agent is None:
#         # Initialize trained RL agent
#         agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma,
#                         config.K_epochs, config.eps_clip)
#         agent.policy.load_state_dict(torch.load('/fed-flow/app/agent/PPO_FedAdapt.pth'))
#     action = agent.exploit(state)
#     action = expand_actions(action, config.CLIENTS_LIST, labels)
#
#     result = action_to_layer(action)
#     config.split_layer = result
#     return result


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
        split_list.append([6, 6])
    return split_list


def only_edge_splitting(state, labels):
    split_list = []
    for i in range(config.K):
        split_list.append([0, config.model_len - 1])
    return split_list


def only_server_splitting(state, labels):
    split_list = []
    for i in range(config.K):
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

    splittingArray = [[lastConvolutionalLayerIndex, config.model_len - 1] for _ in range(config.K)]
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
        split_layer.append(idx)
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
                          activation_size, total_model_size, batchNumber, edge_flops_model, server_flops_model):
    edge_flops = {edgeIP: 0.0 for edgeIP in config.EDGE_SERVER_LIST}
    flop_of_each_edge_on_server = {edgeIP: 0.0 for edgeIP in config.EDGE_SERVER_LIST}
    server_flops = 0
    transmission_time_on_each_client = {}
    edge_server_transmission_time_for_each_client = {}
    total_time_for_each_client = {}

    for i in range(len(action)):
        clientAction = action[i]
        clientIP = config.CLIENTS_INDEX[i]
        edgeIP = config.CLIENT_MAP[clientIP]
        op1 = clientAction[0]
        op2 = clientAction[1]

        if op1 != config.model_len - 1:
            transmission_time_on_each_client[clientIP] = ((2 * (activation_size[op1]) * batchNumber)
                                                          / clients_bw[clientIP])
        else:
            transmission_time_on_each_client[clientIP] = total_model_size / clients_bw[clientIP]

        if op2 != config.model_len - 1:
            edge_server_transmission_time_for_each_client[clientIP] = ((2 * (activation_size[op2]) * batchNumber) /
                                                                       edge_server_bw[edgeIP])
        else:
            edge_server_transmission_time_for_each_client[clientIP] = total_model_size / edge_server_bw[edgeIP]

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
    comp_time_on_each_edge = {
        edge: edge_flops_model.predict([[index, edge_flops[edge], flop_of_each_edge_on_server[edge], server_flops]]) for
        edge, index in EDGE_INDEX}
    comp_time_on_server = server_flops_model.predict([[server_flops]])
    fed_logger.info(Fore.GREEN + f"{action}, {config.CLIENTS_CONFIG}")
    for clientIP in config.CLIENTS_CONFIG.keys():
        op1 = action[config.CLIENTS_CONFIG[clientIP]][0]
        total_time_for_each_client[clientIP] = comp_time_on_each_client[clientIP][op1] + \
                                               transmission_time_on_each_client[clientIP] + \
                                               edge_server_transmission_time_for_each_client[clientIP]

    total_trainingTime = max(total_time_for_each_client.values()) + comp_time_on_server + max(
        comp_time_on_each_edge.values())

    return total_trainingTime[0], total_time_for_each_client
