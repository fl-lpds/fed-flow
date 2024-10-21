"""
websocket messages
"""
from app.config import config


def get_round():
    return config.current_round


def initial_global_weights_server_to_client():
    return f'{get_round()}_MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT'


def initial_global_weights_server_to_edge():
    return f'{get_round()}_MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_EDGE'


def initial_global_weights_edge_to_client():
    return f'{get_round()}_MSG_INITIAL_GLOBAL_WEIGHTS_EDGE_TO_CLIENT'


def training_time_per_iteration_client_to_server():
    return f'{get_round()}_MSG_TRAINING_TIME_PER_ITERATION'


def local_weights_client_to_server():
    return f'{get_round()}_MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER'


def local_weights_client_to_edge():
    return f'{get_round()}_MSG_LOCAL_WEIGHTS_CLIENT_TO_EDGE'


def local_weights_edge_to_server():
    return f'{get_round()}_MSG_LOCAL_WEIGHTS_EDGE_TO_SERVER'


def local_activations_client_to_server():
    return f'{get_round()}_MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER'


def local_activations_client_to_edge():
    return f'{get_round()}_MSG_LOCAL_ACTIVATIONS_CLIENT_TO_EDGE'


def local_activations_edge_to_server():
    return f'{get_round()}_MSG_LOCAL_ACTIVATIONS_EDGE_TO_SERVER'


def split_layers_server_to_edge():
    return f'{get_round()}_SPLIT_LAYERS_SERVER_TO_EDGE'


def split_layers_edge_to_client():
    return f'{get_round()}_SPLIT_LAYERS_EDGE_TO_CLIENT'


def split_layers():
    return f'{get_round()}_MSG_SPLIT_LAYERS'


def test_client_network():
    return f'{get_round()}_MSG_TEST_CLIENT_NETWORK'


def test_server_network_from_server():
    return f'{get_round()}_MSG_TEST_SERVER_NETWORK_FROM_SERVER'


def test_server_network_from_connection():
    return f'{get_round()}_MSG_TEST_SERVER_NETWORK_FROM_CONNECTION'


def test_network_client_to_edge():
    return f'{get_round()}_MSG_TEST_NETWORK_CLIENT_TO_EDGE'


def test_network_edge_to_client():
    return f'{get_round()}_MSG_TEST_NETWORK_EDGE_TO_CLIENT'


def simnet_bw_client_to_edge():
    return f'{get_round()}_MSG_SIMNET_NETWORK_CLIENT_TO_EDGE'


def simnet_bw_edge_to_server():
    return f'{get_round()}_MSG_SIMNET_NETWORK_EDGE_TO_SERVER'


def client_network():
    return f'{get_round()}_MSG_CLIENT_NETWORK'


def server_gradients_server_to_client():
    return f'{get_round()}_MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'


def server_gradients_server_to_edge():
    return f'{get_round()}_MSG_SERVER_GRADIENTS_SERVER_TO_EDGE_'


def server_gradients_edge_to_client():
    return f'{get_round()}_MSG_SERVER_GRADIENTS_EDGE_TO_CLIENT_'


def local_iteration_flag_edge_to_server():
    return f'{get_round()}_MSG_LOCAL_ITERATION_FLAG_EDGE_TO_SERVER'


def local_iteration_flag_client_to_edge():
    return f'{get_round()}_MSG_LOCAL_ITERATION_FLAG_CLIENT_TO_EDGE'


def local_iteration_flag_client_to_server():
    return f'{get_round()}_MSG_LOCAL_ITERATION_FLAG_CLIENT_TO_SERVER'


def init_server_sockets_edge_to_server():
    return f'{get_round()}_MSG_INIT_SERVER_SOCKETS_EDGE_TO_SERVER'


def start_server_client_connection_sockets_edge_to_server():
    return f'{get_round()}_MSG_START_SERVER_CLIENT_CONNECTION_SOCKETS_EDGE_TO_SERVER'


def finish():
    return f'{get_round()}_MSG_FINISH'


def energy_tt_edge_to_server():
    return f'{get_round()}_MSG_ENERGY_EDGE_TO_SERVER'


def energy_client_to_edge():
    return f'{get_round()}_MSG_ENERGY_CLIENT_TO_EDGE_'


def client_quit_client_to_edge():
    return f'{get_round()}_MSG_CLIENT_QUIT_CLIENT_TO_EDGE'


def client_quit_client_to_server():
    return f'{get_round()}_MSG_CLIENT_QUIT_CLIENT_TO_SERVER'


def client_quit_done():
    return f'{get_round()}_MSG_CLIENT_QUIT_DONE'


def client_quit_edge_to_server():
    return f'{get_round()}_MSG_CLIENT_QUIT_EDGE_TO_SERVER'
