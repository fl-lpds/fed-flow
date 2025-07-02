import argparse
import logging
import socket

from app.entity.communicator import Communicator
from app.entity.node_type import NodeType
from app.fl_training.flow import fed_client_flow, fed_edgeserver_flow, fed_server_flow
from app.util import input_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)

# Log mobility status
logger.info(f"Mobility enabled: {options.get('mobility')}")

# Log IP address
ip_address = socket.gethostbyname(socket.gethostname())
logger.info(f"Container IP address: {ip_address}")

Communicator.purge_all_queues()

if __name__ == '__main__':
    node_type = options.get("node_type")
    if node_type == NodeType.CLIENT:
        fed_client_flow.run(options)
    elif node_type == NodeType.EDGE:
        fed_edgeserver_flow.run(options)
    elif node_type == NodeType.SERVER:
        fed_server_flow.run(options)
    else:
        raise ValueError("Node type not supported")
