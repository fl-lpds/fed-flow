import sys

index = 0
simnet = False
# Dataset configration
dataset_name = ''
home = sys.path[0].split('fed-flow')[0] + 'fed-flow' + "/app"
dataset_path = home + '/dataset/data/'

# Model configration
model_cfg = {
    # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
    'VGG5': [('C', 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3), ('M', 32, 32, 2, 32 * 16 * 16, 0),
             ('C', 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32), ('M', 64, 64, 2, 64 * 8 * 8, 0),
             ('C', 64, 64, 3, 64 * 8 * 8, 64 * 8 * 8 * 3 * 3 * 64),
             ('D', 8 * 8 * 64, 128, 1, 64, 128 * 8 * 8 * 64),
             ('D', 128, 10, 1, 10, 128 * 10)]
}

N = 500  # data length
mq_port = 5672
mq_url = "amqp://user:password@broker:5672/%2F"
mq_host = "broker"
mq_user = "user"
mq_pass = "password"
mq_vh = "/"
cluster = "fed-flow"
current_round = 0
model_name = 'alexnet'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [[7, 7], [7, 7], [7, 7], [7, 7], [7, 7]]  # Initial split layers
model_len = 8

# FL training configration

R = 100  # FL rounds
LR = 0.01  # Learning rate
B = 10  # Batch size

# RL training configration
max_episodes = 2000  # max training episodes
max_timesteps = 10  # max timesteps in one episode
exploration_times = 20  # exploration times without std decay
n_latent_var = 64  # number of variables in hidden layer
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
update_timestep = 10  # update policy every n timesteps
K_epochs = 50  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
rl_gamma = 0.9  # discount factor
rl_b = 100  # Batchsize
rl_lr = 0.0003  # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'127.0.0.1': 5}  # infer times for each device

random = True
random_seed = 0
# Network configration
SERVER_ADDR = 'server'

SERVER_PORT = 5002
EDGESERVER_PORT = {'edge1': 5001}

K = 5  # Number of devices
G = 3  # Number of groups
S = 1  # Number of server connecting devices

CLIENTS_BANDWIDTH = [40_000_000, 30_000_000, 10_000_000, 25_000_000, 5_000_000]

# Unique clients order
HOST2IP = {}
EDGE_MQ_MAP = {'edge1': 'broker1'}
CLIENTS_INDEX = {0: 'client1', 1: 'client2', 2: 'client3', 3: 'client4', 4: 'client5'}
CLIENTS_CONFIG = {'client1': 0, 'client2': 1, 'client3': 2, 'client4': 3, 'client5': 4}
EDGE_SERVER_LIST = ['edge1']
EDGE_SERVER_CONFIG = {0: 'edge1'}
CLIENTS_LIST = ['client1', 'client2', 'client3', 'client4', 'client5']
EDGE_MAP = {'edge1': ['client1', 'client2', 'client3', 'client4', 'client5']}
CLIENT_MAP = {'client1': 'edge1', 'client2': 'edge1', 'client3': 'edge1', 'client4': 'edge1', 'client5': 'edge1'}
