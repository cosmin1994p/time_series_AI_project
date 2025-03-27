import torch

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
