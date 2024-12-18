import os
import sys
from os import environ

from app.entity.node_coordinate import NodeCoordinate
from app.entity.node_identifier import NodeIdentifier

DEBUG = os.getenv('DEBUG', 'False') == 'True'
CLIENTS_BANDWIDTH = []
index = 0
simnet = False
# Dataset configration
dataset_name = ''
home = sys.path[0].split('fed-flow')[0] + 'fed-flow' + "/app"
dataset_path = home + '/dataset/data/'
N = 50000  # data # length

mq_url = "amqp://rabbitmq:rabbitmq@localhost:5672/"
current_node_mq_url = "Will be set by input options"
cluster = "fed-flow"
# Model configration
model_cfg = {
    # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
    'VGG5': [('C', 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3), ('M', 32, 32, 2, 32 * 16 * 16, 0),
             ('C', 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32), ('M', 64, 64, 2, 64 * 8 * 8, 0),
             ('C', 64, 64, 3, 64 * 8 * 8, 64 * 8 * 8 * 3 * 3 * 64),
             ('D', 8 * 8 * 64, 128, 1, 64, 128 * 8 * 8 * 64),
             ('D', 128, 10, 1, 10, 128 * 10)]
}
model_name = ''
# split_layer = [6]  # Initial split layers for no edge base
split_layer = [[6, 6]]  # Initial split layers
model_len = 7

# FL training configration
R = int(environ.get("ROUND_COUNT", "3"))  # FL rounds
learning_rate = 0.01  # Learning rate
B = 100  # Batch size
lr_step_size = 20
lr_gamma = 0.1

K = int(environ.get("DEVICE_COUNT", "1"))  # Number of devices
G = 1  # Number of groups
S = 1

# Topology configration for decentralized mode
CURRENT_NODE_NEIGHBORS: list[NodeIdentifier] = []  # (ip, port)
INITIAL_NODE_COORDINATE = NodeCoordinate

SCENARIO_DESCRIPTION = environ.get("SCENARIO_DESCRIPTION", "")
