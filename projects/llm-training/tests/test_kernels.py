import torch

from src.utils.device import get_default_device, get_device


class TestDevice:
    def test_get_device(self):
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_get_default_device(self):
        device = get_default_device()
        assert isinstance(device, torch.device)
