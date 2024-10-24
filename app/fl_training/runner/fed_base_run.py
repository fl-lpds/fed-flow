import argparse
import sys


sys.path.append('../../../')
from app.entity.communicator import Communicator
from app.util import input_utils
from app.fl_training.flow import fed_client_flow, fed_edgeserver_flow, fed_server_flow
from app.entity.node_type import NodeType

parser = argparse.ArgumentParser()
options = input_utils.parse_argument(parser)
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
