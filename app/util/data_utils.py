import pickle

import torch
import torchvision.transforms as transforms
from colorama import Fore
from torch.utils.data import DataLoader, Subset

from app.config import config
from app.config.config import *
from app.config.logger import fed_logger
from app.dataset.entity.dataset_interface import DatasetInterface
from app.util import model_utils, message_utils

DATASET_BASE_DIR = "app.dataset.entity."


def get_trainset():
    transform_train = model_utils.get_class().data_transformer()
    dataset = get_class()()
    trainset: DatasetInterface = dataset.get_data()(
        root=config.dataset_path + config.dataset_name + '/', train=True, transform=transform_train)
    return trainset


def get_trainloader(trainset, part_tr, cpu_count):
    subset = Subset(trainset, part_tr)
    trainloader = DataLoader(
        subset, batch_size=B, shuffle=True, num_workers=cpu_count)
    return trainloader


def sizeofmessage(msg):
    size = 0
    for i in range(len(msg)):
        size += sys.getsizeof(pickle.dumps(msg[i]))
    # fed_logger.info(Fore.RED+f"{msg[0]},{size}")
    return size * 8


def get_testloader(testset, cpu_count):
    return torch.utils.data.DataLoader(testset, batch_size=B, shuffle=False, num_workers=cpu_count)


def get_testset():
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])
    dataset = get_class()()
    return dataset.get_data()(root=config.dataset_path + config.dataset_name + '/', train=False,
                              transform=transform_test)


def get_class():
    kls = DATASET_BASE_DIR + config.dataset_name + '.' + config.dataset_name
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m
