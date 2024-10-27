import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import app.util.model_utils as model_utils
from app.config import config
from app.config.logger import fed_logger
from app.entity.node import Node


def report_results(node: Node, training_times: list[float], client_bandwidths: list[float],
                   accuracy: list[float], neighbor_bandwidths: Optional[list[float]] = None):
    current_time = time.strftime("%Y-%m-%d %H:%M")
    runtime_config = f'{current_time}'
    save_path = f"Results/{runtime_config}"
    rounds_count = config.R
    draw_graph(10, 5, range(1, rounds_count + 1), training_times, str(node), "FL Rounds", "Training Time (s)",
               save_path, f"training-time-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), client_bandwidths, str(node), "FL Rounds", "Bandwidths (bytes/s)",
               save_path, f"bandwidth-{str(node)}")
    draw_graph(10, 5, range(1, rounds_count + 1), accuracy, str(node), "FL Rounds", "Accuracy (%)",
               save_path, f"accuracy-{str(node)}")
    if neighbor_bandwidths:
        draw_graph(10, 5, range(1, rounds_count + 1), neighbor_bandwidths, str(node), "FL Rounds",
                   "Neighbors Bandwidths (bytes/s)",
                   save_path, f"neighbor-bandwidths-{str(node)}")
    fed_logger.info(f"Results created successfully at")


def draw_graph(figSizeX, figSizeY, x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, pictureName))
    # plt.show()
    plt.close()


def draw_hist(x, title, xlabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.hist(x, 10)
    plt.title(title)
    plt.xlabel(xlabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    plt.close()
    # plt.show()


def draw_scatter(x, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, pictureName))
    # plt.show()
    plt.close()


def draw_3dGraph(x, y, z, xlabel, ylabel, zlabel):
    fig = go.Figure(data=[go.Mesh3d(x=x,
                                    y=y,
                                    z=z,
                                    opacity=0.7, )])

    fig.update_layout(scene=dict(xaxis_title=xlabel,
                                 yaxis_title=ylabel,
                                 zaxis_title=zlabel,
                                 xaxis_showspikes=False,
                                 yaxis_showspikes=False))

    fig.show()


def sigmoidActivation(x: float) -> float:
    """ It returns 1/(1+exp(-x)). where the values lies between zero and one """

    return 1 / (1 + np.exp(-x))


def tanhActivation(x: float) -> float:
    """ It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1."""

    return np.tanh(x)


def normalizeReward_linear(maxAmount, minAmount, x, minNormalized, maxNormalized):
    P = [maxAmount, minNormalized]
    Q = [minAmount, maxNormalized]
    lineGradient = (P[1] - Q[1]) / (P[0] - Q[0])
    y = lineGradient * (x - Q[0]) + Q[1]
    return y


def normalizeReward_tan(x, turning_point):
    y = max(min(-pow(x - turning_point, 3) / pow(turning_point, 3), 1), -1)
    return y


def convert_To_Len_th_base(n, arr, modelLen, deviceNumber, allPossible):
    a: str = ""
    for i in range(deviceNumber * 2):
        a += str(arr[n % modelLen])
        n //= modelLen
    allPossible.append(a)


def allPossibleSplitting(modelLen, deviceNumber):
    arr = [i for i in range(0, modelLen + 1)]
    allPossible = list()
    for i in range(pow(modelLen, deviceNumber * 2)):
        # Convert i to Len th base
        convert_To_Len_th_base(i, arr, modelLen, deviceNumber, allPossible)
    result = list()
    for item in allPossible:
        temp = []
        isOk = True
        for j in range(0, len(item) - 1, 2):
            if int(item[j]) > int(item[j + 1]):
                isOk = False
        if isOk:
            for i in range(0, len(item), 2):
                temp.append([int(item[i]), int(item[i + 1])])
        if len(temp) != 0:
            result.append(temp)
    return result


def randomSelectionSplitting(modelLen, deviceNumber) -> list[list[int]]:
    splittingForOneDevice = []
    for i in range(0, modelLen):
        for j in range(0, i + 1):
            splittingForOneDevice.append([j, i])

    result = []
    for i in range(deviceNumber):
        rand = random.randint(0, len(splittingForOneDevice) - 1)
        result.append(splittingForOneDevice[rand])
    return result


# def createAgent(agentType, fraction, timestepNum, saveSummariesPath, environment=None):
#     if agentType == 'ac':
#         return AC.create(fraction=fraction, environment=environment, timestepNum=timestepNum,
#                          saveSummariesPath=saveSummariesPath)
#     elif agentType == 'tensorforce':
#         return TensorforceAgent.create(fraction=fraction, timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
#     elif agentType == 'trpo':
#         return TRPO.create(fraction=fraction, environment=environment,
#                            timestepNum=timestepNum, saveSummariesPath=saveSummariesPath)
#     elif agentType == 'random':
#         return RandomAgent.RandomAgent(environment=environment)
#     elif agentType == 'noSplitting':
#         return NoSplitting.NoSplitting(environment=environment)
#     elif agentType == 'tf_ppo':
#         return Agent.load(directory='/fed-flow/app/agent/ppo_1_1_1_agent')
#     else:
#         raise Exception('Invalid config select from [ppo, ac, tensorforce, random]')


def actionToLayerEdgeBase(splitDecision: list[float]) -> tuple[float, float]:
    """ It returns the offloading points for the given action ( op1 , op2 )"""
    op1: float
    op2: float  # Offloading points op1, op2
    workLoad = []
    model_state_flops = []
    splitDecision = splitDecision[0]
    for l in model_utils.get_unit_model().cfg:
        workLoad.append(l[5])
        model_state_flops.append(sum(workLoad))

    totalWorkLoad = sum(workLoad)
    model_flops_list = np.array(model_state_flops)
    model_flops_list = model_flops_list / totalWorkLoad
    idx = np.where(np.abs(model_flops_list - splitDecision[0]) == np.abs(model_flops_list - splitDecision[0]).min())
    op1 = int(idx[0][-1])

    op2_totalWorkload = sum(workLoad[op1:])
    model_state_flops = []
    for l in range(op1, model_utils.get_unit_model_len()):
        model_state_flops.append(sum(workLoad[op1:l + 1]))
    model_flops_list = np.array(model_state_flops)
    model_flops_list = model_flops_list / op2_totalWorkload

    idx = np.where(np.abs(model_flops_list - splitDecision[1]) == np.abs(model_flops_list - splitDecision[1]).min())
    op2 = int(idx[0][-1]) + op1

    return op1, op2


def rewardFun(fraction, energy, trainingTime, classicFlTrainingTime, maxEnergy, minEnergy):
    rewardOfEnergy = normalizeReward_linear(maxAmount=maxEnergy,
                                            minAmount=minEnergy,
                                            x=energy,
                                            minNormalized=-1,
                                            maxNormalized=1)
    rewardOfTrainingTime = trainingTime
    rewardOfTrainingTime -= classicFlTrainingTime
    rewardOfTrainingTime /= 100
    rewardOfTrainingTime *= -1
    rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

    if fraction <= 1:
        reward = (fraction * rewardOfEnergy) + ((1 - fraction) * rewardOfTrainingTime)
    else:
        raise Exception("Fraction must be less than 1")
    return reward


def rewardFunTan(fraction, energy, trainingTime, classicFlTrainingTime, classic_Fl_Energy):
    rewardOfEnergy = normalizeReward_tan(
        x=energy,
        turning_point=classic_Fl_Energy
    )
    rewardOfTrainingTime = trainingTime
    rewardOfTrainingTime -= classicFlTrainingTime
    rewardOfTrainingTime /= 100
    rewardOfTrainingTime *= -1
    rewardOfTrainingTime = min(max(rewardOfTrainingTime, -1), 1)

    if fraction <= 1:
        reward = (fraction * rewardOfEnergy) + ((1 - fraction) * rewardOfTrainingTime)
    else:
        raise Exception("Fraction must be less than 1")
    return reward
