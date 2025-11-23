import torch
from torch import device


def get_available_torch_device() -> device:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"CPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
