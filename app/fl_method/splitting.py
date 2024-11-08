import random

import numpy as np
import torch
from stable_baselines3 import PPO, DDPG

from app.config import config
from app.config.logger import fed_logger
# from app.model.entity.rl_model import PPO
from app.util import model_utils, rl_utils


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


# def heuristic_splitting(state, label):
#     # state = [ client's BW, edge's BW, server's BW, client remaining energy]
#     client_BW = {}
#     client_RE = {}
#     for i in range(len(config.CLIENTS_LIST)):
#         client_BW[f"device{i}"] = state[i]
#     for i in range(len(config.CLIENTS_LIST) + len(config.EDGE_SERVER_LIST), len(state)):
#         client_RE[f"device{i}"] = state[i]


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
    for i in range(config.K):
        split_list.append(3)
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
