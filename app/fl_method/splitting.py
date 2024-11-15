import random
from math import inf

import numpy as np
import torch
from stable_baselines3 import PPO, DDPG

from app.config import config
from app.config.logger import fed_logger
# from app.model.entity.rl_model import PPO
from app.util import model_utils, rl_utils
from colorama import Fore


def edge_based_rl_splitting(state, labels):
    env = rl_utils.CustomEnv()
    agent = DDPG.load('/fed-flow/app/agent/160.zip', env=env,
                      custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    floatAction, _ = agent.predict(observation=state, deterministic=True)
    actions = []
    for i in range(0, len(floatAction), 2):
        actions.append([rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[0],
                        rl_utils.actionToLayerEdgeBase([floatAction[i], floatAction[i + 1]])[1]])

    return actions


def edge_based_heuristic_splitting(state: dict, label):
    # state = [ client's BW, edge-server BW, client remaining energy , client power usage, each layers activation size]
    client_bw = {}
    client_utilization = {}
    edge_server_bw = {}
    client_remaining_energy = {}
    client_comp_energy = {}
    client_comm_energy = {}
    client_power_usage = {}
    client_remaining_runtime_comp = {}
    client_remaining_runtime_comm = {}
    previous_action = state['prev_action']

    activation_size = {}
    gradient_size = {}
    total_model_size = state['total_model_size']
    client_bw_score = {}

    for key, value in state.items():
        if ("client" in key) and ("bw" in key):
            client_bw[key[:-3]] = value
        elif "utilization" in key:
            client_utilization[key[:-12]] = value
        elif "comp_energy" in key:
            client_comp_energy[key[:-12]] = value
        elif "comm_energy" in key:
            client_comm_energy[key[:-12]] = value
        elif "bw" in key:
            edge_server_bw[key[:-3]] = value
        elif ("client" in key) and ("re" in key):
            client_remaining_energy[key[:-3]] = value
        elif "power" in key:
            client_power_usage[key[:-6]] = value
        elif "activation" in key:
            activation_size = value
        else:
            gradient_size = value

    fed_logger.info(Fore.MAGENTA + f"NEW COMP e: {client_comp_energy}")

    action = previous_action
    client_bw_score = normalizer(client_bw)

    layer_activation = {}
    layer_gradient = {}
    layer_num = 0
    current = None
    for num in list(activation_size.values()):
        if num != current:
            layer_activation[layer_num] = num
            layer_num += 1
            current = num
        elif num == 13107200 and current == 13107200 and 4 not in layer_activation:
            # Special case for the second occurrence of 12.5
            layer_activation[layer_num] = num
            layer_num += 1
    layer_gradient = layer_activation
    batchNumber = (config.N / len(config.CLIENTS_CONFIG.keys())) / config.B

    for client in client_remaining_energy.keys():
        client_op1 = previous_action[config.CLIENTS_CONFIG[client]][0]
        if client_remaining_energy[client] != 0:
            client_remaining_runtime_comp[client] = client_remaining_energy[client] / (
                    client_comp_energy[client][client_op1] + (
                    (((layer_activation[client_op1] + layer_gradient[client_op1]) * batchNumber) / client_bw[client]) *
                    client_power_usage[client][1]))

    client_remaining_runtime_comp_score = normalizer(client_remaining_runtime_comp)
    # client_remaining_runtime_comm_score = normalizer(client_remaining_runtime_comm)
    client_score = {}
    for client, score in client_remaining_runtime_comp_score.items():
        client_score[client] = client_remaining_runtime_comp_score[client]

    high_prio_device = {device: score for device, score in client_score.items() if score < 0.5}
    fed_logger.info(f"RUN TIME SCORE: {client_remaining_runtime_comp_score.items()}")
    fed_logger.info(f"HIGH PRIO DEVICES: {high_prio_device.items()}")

    for client, score in high_prio_device.items():
        client_op1 = previous_action[config.CLIENTS_CONFIG[client]][0]
        best_op1 = client_op1
        if client_op1 != config.model_len - 1:
            tt_trans_now = (2 * (layer_activation[client_op1]) * batchNumber) / client_bw[client]
        else:
            tt_trans_now = total_model_size / client_bw[client]

        comm_energy = tt_trans_now * client_power_usage[client][1]
        client_comm_energy_ratio = comm_energy / (comm_energy + client_comp_energy[client][client_op1])
        client_comp_energy_ratio = 1 - client_comm_energy_ratio

        fed_logger.info(Fore.GREEN + f"Activation Layer : {layer_activation}")
        fed_logger.info(Fore.GREEN + f"APPROXIMATED TT : {tt_trans_now}")
        fed_logger.info(Fore.GREEN + f"APPROXIMATED COMM E : {comm_energy}")
        fed_logger.info(Fore.GREEN + f"COMP E : {client_comp_energy[client][client_op1]}")
        fed_logger.info(Fore.GREEN + f"COMP E RATIO : {client_comp_energy_ratio}")
        fed_logger.info(Fore.GREEN + f"COMM E RATIO: {client_comm_energy_ratio}")

        if client_comm_energy_ratio > client_comp_energy_ratio * 1.5:
            condidate_op1 = int(min(layer_activation, key=layer_activation.get))
            tt_trans = (2 * (layer_activation[condidate_op1]) * batchNumber) / client_bw[client]
            comm_energy = tt_trans * client_power_usage[client][1]
            if condidate_op1 in client_comp_energy[client]:
                comp_energy = client_comp_energy[client][condidate_op1]
                if (comp_energy + comm_energy) < (tt_trans_now * client_power_usage[client][1] + client_comp_energy[client][client_op1]):
                    best_op1 = condidate_op1
                else:
                    best_op1 = client_op1
            else:
                best_op1 = condidate_op1
        elif client_comp_energy_ratio > client_comm_energy_ratio * 1.5:
            filtered = {k: v for k, v in layer_activation.items() if 0 <= k < client_op1}
            if len(filtered.values()) == 0:
                best_op1 = 0
            else:
                for layer, activation_size in filtered.items():
                    condidate_op1 = layer
                    tt_trans = (2 * (activation_size) * batchNumber) / client_bw[client]
                    comm_energy = tt_trans * client_power_usage[client][1]
                    if condidate_op1 in client_comp_energy[client]:
                        comp_energy = client_comp_energy[client][condidate_op1]
                        if (comp_energy + comm_energy) < (
                                tt_trans_now * client_power_usage[client][1] + client_comp_energy[client][client_op1]):
                            best_op1 = condidate_op1
                        else:
                            best_op1 = client_op1
                    else:
                        best_op1 = condidate_op1

        action[config.CLIENTS_CONFIG[client]] = [best_op1, 6]
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
        split_list.append(6)
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
