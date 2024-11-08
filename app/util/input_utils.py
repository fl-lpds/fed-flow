import argparse
from app.config import config
from app.util import model_utils

options = {
    '-a': ['--aggregation', 'fed_avg', 'name of the aggregation method'],
    '-e': ['--edgebased', False, 'True if edge servers are available otherwise, False'],
    '-c': ['--clustering', 'none_clustering', 'name of the clustering method'],
    '-s': ['--splitting', 'none_splitting', 'name of the splitting method'],
    '-m': ['--model', 'vgg', 'class name of the training model'],
    '-d': ['--dataset', 'cifar10', 'the name of the using dataset'],
    '-o': ['--offload', False, 'offloading or classic FL mode'],
    '-dl': ['--datasetlink', '', 'the link to dataset  python file'],
    '-ml': ['--modellink', '', 'the link to model  python file'],
    '-i': ['--index', '0', 'the device index'],
    '-sb': ['--simulatebandwidth', 'False', 'the device network bandwidth'],
    '-en': ['--energy', 'False', 'enable or disable energy estimation'],

}


def parse_argument(parser: argparse.ArgumentParser()):
    """

    Args:
        parser:

    Returns: a map of options with their associated values

    """
    for op in options.keys():
        parser.add_argument(op, options.get(op)[0], help=options.get(op)[2], type=str,
                            default=options.get(op)[1])
    args = parser.parse_args()
    option = vars(args)
    if option.get("edgebased") == 'True':
        option["edgebased"] = True
    else:
        option["edgebased"] = False

    if option.get("offload") == 'True':
        option["offload"] = True
    else:
        option["offload"] = False
    config.dataset_name = option.get('dataset')
    config.model_name = option.get('model')
    config.index = int(option.get('index'))
    model_utils.download_model(option.get('modellink'))
    return option
