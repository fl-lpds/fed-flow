import os
import random

# import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# from gymnasium import spaces


# from tensorforce import Agent
# from app.model.entity.rl_model import NoSplitting, TRPO, AC, TensorforceAgent, RandomAgent, TF_PPO

# class CustomEnv(gym.Env):
#     """Custom Environment that follows gym interface."""
#
#     def __init__(self):
#         super().__init__()
#
#         self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * K,), dtype=np.float32, seed=None)
#
#         # bandwidths : 0% fluctuation, 10% fluctuation, 20% fluctuation,..., 90% fluctuation.
#         # observation_spec = [10] * (self.iotDeviceNum + self.edgeDeviceNum)
#         # self.observation_space = spaces.MultiDiscrete(observation_spec)
#         self.observation_space = spaces.Box(low=0.0, high=20,
#                                             shape=((2 * K) + G,),
#                                             dtype=np.float32, seed=None)
#
#     def rewardFun(self, action):
#         pass
#
#     def step(self, action):
#         pass
#
#     def reset(self, seed=None, options=None):
#         pass
#
#     def render(self):
#         pass


def draw_graph(figSizeX, figSizeY, y, title, xlabel, ylabel, savePath, pictureName, saveFig=True):
    # Create a plot
    plt.figure(figsize=(int(figSizeX), int(figSizeY)))  # Set the figure size
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if saveFig:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
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
    plt.figure(figsize=(10, 15))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
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
    P = [maxAmount, maxNormalized]
    Q = [minAmount, minNormalized]
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


def actionToLayer(splitDecision: list[float], flop_per_layer) -> tuple[float, float]:
    """ It returns the offloading points for the given action ( op1 , op2 )"""
    op1: float
    op2: float
    workLoad = []
    model_state_flops = []

    for l in flop_per_layer:
        workLoad.append(l)
        model_state_flops.append(sum(workLoad))

    totalWorkLoad = sum(workLoad)
    model_flops_list = np.array(model_state_flops)
    model_flops_list = (model_flops_list / totalWorkLoad)

    idx = np.where(np.abs(model_flops_list - splitDecision[0]) == np.abs(model_flops_list - splitDecision[0]).min())
    op1 = int(idx[0][-1])

    op2_totalWorkload = sum(workLoad[op1 + 1:])
    model_state_flops = []
    import app.config.config as cfg
    for l in range(op1 + 1, cfg.model_len):
        model_state_flops.append(sum(workLoad[op1 + 1:l + 1]))
    model_flops_list = np.array(model_state_flops)
    model_flops_list = model_flops_list / op2_totalWorkload

    idx = np.where(np.abs(model_flops_list - splitDecision[1]) == np.abs(model_flops_list - splitDecision[1]).min())
    if splitDecision[1] != 0:
        op2 = (int(idx[0][-1]) + 1) + op1
    else:
        op2 = op1

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
