import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torchvision.datasets
from PIL import Image
from torchvision.datasets.utils import check_integrity

from app.dataset.entity.dataset_interface import DatasetInterface


class cifar100(DatasetInterface):

    def __init__(
            self
    ) -> None:

        super().__init__(torchvision.datasets.CIFAR100)
