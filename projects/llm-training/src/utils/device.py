import platform
from typing import Literal

import torch


def get_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_sync() -> None:
    if get_device() == "cuda":
        torch.cuda.synchronize()
    elif get_device() == "mps":
        torch.mps.synchronize()


def get_default_device() -> torch.device:
    return torch.device(get_device())


def is_mps() -> bool:
    return get_device() == "mps"


def is_cuda() -> bool:
    return get_device() == "cuda"
