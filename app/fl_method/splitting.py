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

    for client in client_remaining_energy.keys():
        client_op1 = previous_action[config.CLIENTS_CONFIG[client]][0]
        tt_trans = 0
        if client_remaining_energy[client] != 0:
            if client_op1 != config.model_len - 1:
                tt_trans = (2 * (activation_size[client_op1]) * batchNumber) / client_bw[client]
            else:
                tt_trans = total_model_size / client_bw[client]
            client_remaining_runtime[client] = client_remaining_energy[client] / (
                    client_comp_energy[client][client_op1] + (tt_trans * client_power_usage[client][1]))

    client_remaining_runtime_comp_score = normalizer(client_remaining_runtime)
    client_score = {}
    for client, score in client_remaining_runtime_comp_score.items():
        client_score[client] = client_remaining_runtime_comp_score[client]

    high_prio_device = {device: score for device, score in client_score.items() if score <= 1.0}
    fed_logger.info(f"RUN TIME SCORE: {client_remaining_runtime_comp_score.items()}")
    fed_logger.info(f"HIGH PRIO DEVICES: {high_prio_device.items()}")

    classicFL_action = [[config.model_len - 1, config.model_len - 1] for _ in range(len(config.CLIENTS_CONFIG))]
    classicFL_tt = trainingTimeEstimator(classicFL_action, client_comp_time, client_bw, edge_server_bw,
                                         flops_of_each_layer, activation_size, total_model_size, batchNumber,
                                         edge_poly_model, server_poly_model)
    fed_logger.info(f"Classic FL training time approximation: {classicFL_tt}")

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
        for layer, size in activation_size.items():
            candidate_op1 = layer
            if layer != config.model_len - 1:
                tt_trans = (2 * size * batchNumber) / client_bw[client]
            else:
                tt_trans = total_model_size / client_bw[client]
            comm_energy = tt_trans * client_power_usage[client][1]
            if candidate_op1 in client_comp_energy[client]:
                comp_energy = client_comp_energy[client][candidate_op1]
                if (comp_energy + comm_energy) < min_total_energy:
                    min_total_energy = comp_energy + comm_energy
                    best_op1 = candidate_op1
            else:
                best_op1 = candidate_op1
                break
            action[config.CLIENTS_CONFIG[client]] = [best_op1, 6]
            min_total_training_time = trainingTimeEstimator(action, client_comp_time, client_bw, edge_server_bw,
                                                            flops_of_each_layer, activation_size, total_model_size,
                                                            batchNumber, edge_poly_model, server_poly_model)
            for candidate_op2 in range(best_op1, config.model_len):
                candidateAction = action
                candidateAction[config.CLIENTS_CONFIG[client]][1] = candidate_op2
                candidate_op2_tt = trainingTimeEstimator(candidateAction, client_comp_time, client_bw, edge_server_bw,
                                                         flops_of_each_layer, activation_size, total_model_size,
                                                         batchNumber, edge_poly_model, server_poly_model)
                fed_logger.info(
                    Fore.GREEN + f"Client {client}, OP2 Candidate: {candidate_op2}, candidate op2 training time: {candidate_op2_tt}")
                if candidate_op2_tt < min_total_training_time:
                    action[config.CLIENTS_CONFIG[client]] = [best_op1, candidate_op2]
                    min_total_training_time = candidate_op2_tt
        action[config.CLIENTS_CONFIG[client]] = [best_op1, best_op2]
        fed_logger.info(Fore.GREEN + f"client: {client}, best_op1: {best_op1}, best_op2: {best_op2}")
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
        split_list.append([0, 6])
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
    for i in range(config.K):
        op1 = random.randint(1, config.model_len - 1)
        op2 = random.randint(op1, config.model_len - 1)
        splittingArray.append([op1, op2])
    return splittingArray
    # split_list = []
    # for i in range(config.K):
    #     split_list.append([2, 6])
    # return split_list


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


def trainingTimeEstimator(action, comp_time_on_each_client, clients_bw, edge_server_bw, flops_of_each_layer,
                          activation_size, total_model_size, batchNumber, edge_flops_model, server_flops_model):
    edge_flops = {edgeIP: 0.0 for edgeIP in config.EDGE_SERVER_LIST}
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
            server_flops += sum(flops_of_each_layer[op2 + 1:])
        # offloading on client and edge
        elif (op1 < op2) and op2 == config.model_len - 1:
            edge_flops[edgeIP] += sum(flops_of_each_layer[op1 + 1:op2 + 1])
        # offloading on client and server
        elif (op1 == op2) and op1 < config.model_len - 1:
            server_flops += sum(flops_of_each_layer[op2 + 1:])

    EDGE_INDEX: list = [(edge, config.EDGE_SERVER_LIST.index(edge)) for edge in config.EDGE_SERVER_LIST]
    comp_time_on_each_edge = {edge: edge_flops_model.predict([[index, edge_flops[edge]]]) for edge, index in EDGE_INDEX}
    comp_time_on_server = server_flops_model.predict([[server_flops]])
    fed_logger.info(Fore.GREEN + f"{action}, {config.CLIENTS_CONFIG}")
    for clientIP in config.CLIENTS_CONFIG.keys():
        op1 = action[config.CLIENTS_CONFIG[clientIP]][0]
        total_time_for_each_client[clientIP] = comp_time_on_each_client[clientIP][op1] + \
                                               transmission_time_on_each_client[clientIP] + \
                                               edge_server_transmission_time_for_each_client[clientIP]

    total_trainingTime = max(total_time_for_each_client.values()) + comp_time_on_server + max(
        comp_time_on_each_edge.values())

    return total_trainingTime[0]
