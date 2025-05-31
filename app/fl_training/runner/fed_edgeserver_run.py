import argparse
import sys

sys.path.append('../../../')
from app.util import input_utils
from app.fl_training.flow import fed_edgeserver_flow
import torch.multiprocessing as mp
from app.config.logger import fed_logger
from colorama import Fore

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    options = input_utils.parse_argument(parser)
    fed_logger.info(Fore.GREEN + f"OPTIONS: {options}")
    fed_edgeserver_flow.run(options)
