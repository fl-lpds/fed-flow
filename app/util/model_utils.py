import argparse
import collections
import os.path
import urllib.request

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.init as init
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.config import config
from app.config.logger import fed_logger
from app.model.entity.nn_model.vgg import *

np.random.seed(0)
torch.manual_seed(0)

MODEL_BASE_DIR = 'app.model.entity.nn_model.'


def get_model(location, layer, device, edge_based):
    net = get_class()()
    net = net.initialize(location, layer, edge_based)
    net = net.to(device)
    fed_logger.debug(str(net))
    return net


def split_weights_client(weights, cweights):
    """
    evaluate client weights
    """
    for key in cweights:
        assert cweights[key].size() == weights[key].size()
        cweights[key] = weights[key]
    return cweights


def split_weights_server(weights, cweights, sweights, eweights):
    """
    evaluate server weights
    """
    ckeys = list(cweights)
    skeys = list(sweights)
    ekeys = list(eweights)
    keys = list(weights)

    for i in range(len(skeys)):
        assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys) + len(ekeys)]].size()
        sweights[skeys[i]] = weights[keys[i + len(ckeys) + len(ekeys)]]

    return sweights


def split_weights_edgeserver(weights, cweights, eweights):
    ckeys = list(cweights)
    ekeys = list(eweights)
    keys = list(weights)

    for i in range(len(ekeys)):
        assert eweights[ekeys[i]].size() == weights[keys[i + len(ckeys)]].size()
        eweights[ekeys[i]] = weights[keys[i + len(ckeys)]]

    return eweights


def concat_weights(weights, cweights, sweights):
    concat_dict = collections.OrderedDict()

    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)

    for i in range(len(ckeys)):
        concat_dict[keys[i]] = cweights[ckeys[i]]

    for i in range(len(skeys)):
        concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

    return concat_dict


def zero_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.zeros_(m.weight)
            init.zeros_(m.bias)
            init.zeros_(m.running_mean)
            init.zeros_(m.running_var)
        elif isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    return net


def norm_list(alist):
    return [l / sum(alist) for l in alist]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test(uninet, testloader, device, criterion):
    uninet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = uninet(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    fed_logger.info('Test Accuracy: {}'.format(acc))

    # Save checkpoint.
    torch.save(uninet.state_dict(), './' + config.model_name + '.pth')

    return acc


def get_class():
    kls = MODEL_BASE_DIR + config.model_name + '.' + config.model_name
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def download_model(link):
    if not link == '':
        if not check_link_integrity():
            urllib.request.urlretrieve(link, MODEL_BASE_DIR)
            if not check_link_integrity():
                fed_logger.error('the file is not downloaded correctly or its name doesn\'t follow the proper format')


def check_link_integrity():
    return os.path.exists(MODEL_BASE_DIR + config.model_name)


def get_unit_model_len():
    return len(get_class()().get_config())


def get_unit_model() -> NNModel:
    return get_class()()


def createFlopsPredictionModel(flop_time_csv_path, isEdge=True):
    file_path = flop_time_csv_path  # Replace with your file path
    filtered_data = pd.read_csv(file_path, names=['flop', 'time'])
    if isEdge:
        filtered_data = pd.read_csv(file_path,
                                    names=['edgeIndex', 'flop', 'flop_of_edge_on_server', 'flop_on_server', 'time'])

    if len(filtered_data) > 0:
        X = filtered_data['flop'].values.reshape(-1, 1)
        y = filtered_data['time'].values
        if isEdge:
            X = filtered_data[['edgeIndex', 'flop', 'flop_of_edge_on_server', 'flop_on_server']].values
            y = filtered_data['time'].values

        fed_logger.info(f"X: {X}")
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        # fed_logger.info(f"X scaled: {X}")

        linear_model = LinearRegression()
        linear_model.fit(X, y)

        degree = 2
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(X, y)

        plt.figure(figsize=(12, 8))

        if isEdge:
            flop_on_edge_range = np.linspace(filtered_data['flop'].min(), filtered_data['flop'].max(), 500)
            flop_of_each_edge_on_server = np.linspace(filtered_data['flop_of_edge_on_server'].min(),
                                                      filtered_data['flop_of_edge_on_server'].max(), 500)
            flop_on_server = np.linspace(filtered_data['flop_on_server'].min(),
                                         filtered_data['flop_on_server'].max(), 500)

            index_mean = filtered_data['edgeIndex'].mean()
            fed_logger.info(
                f"FLOP Range: {flop_on_edge_range}, index_mean: {index_mean}, np,full_like: {np.full_like(flop_on_edge_range, 1)}, "
                f"kir khar: {np.c_[np.full_like(flop_on_edge_range, index_mean), flop_on_edge_range, flop_of_each_edge_on_server, flop_on_server]}")

            linear_predictions = linear_model.predict(np.c_[np.full_like(flop_on_edge_range, index_mean),
                                                        flop_on_edge_range,
                                                        flop_of_each_edge_on_server,
                                                        flop_on_server])
            poly_predictions = poly_model.predict(np.c_[np.full_like(flop_on_edge_range, index_mean),
                                                flop_on_edge_range,
                                                flop_of_each_edge_on_server,
                                                flop_on_server])

            plt.plot(flop_on_edge_range, linear_predictions, color='green', label='Linear Regression Model')
            plt.plot(flop_on_edge_range, poly_predictions, color='red', label=f'Polynomial Regression Model (Degree {degree})')

        else:
            flop_range = np.linspace(filtered_data['flop'].min(), filtered_data['flop'].max(), 1000).reshape(-1, 1)
            linear_predictions = linear_model.predict(flop_range)
            poly_predictions = poly_model.predict(flop_range)
            plt.plot(flop_range, linear_predictions, color='green', label='Linear Regression Model')
            plt.plot(flop_range, poly_predictions, color='red', label=f'Polynomial Regression Model (Degree {degree})')

        plt.scatter(filtered_data['flop'], filtered_data['time'], color='blue', label='Original Data', alpha=0.6)

        plt.xlabel('Workload (FLOP)')
        plt.ylabel('Time Taken (seconds)')
        plt.title('Workload vs Time with Linear and Polynomial Regression')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('/fed-flow/Graphs'):
            os.makedirs('/fed-flow/Graphs')

        if isEdge:
            plt.savefig(os.path.join('/fed-flow/Graphs', 'edge_Prediction'))
        else:
            plt.savefig(os.path.join('/fed-flow/Graphs', 'server_Prediction'))

        plt.close()
        if isEdge:
            joblib.dump(linear_model, '/fed-flow/app/model/edge_flops_prediction_linear_model.pkl')
            joblib.dump(poly_model, '/fed-flow/app/model/edge_flops_prediction_poly_model.pkl')
        else:
            joblib.dump(linear_model, '/fed-flow/app/model/server_flops_prediction_linear_model.pkl')
            joblib.dump(poly_model, '/fed-flow/app/model/server_flops_prediction_poly_model.pkl')
