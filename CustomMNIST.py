import torch
from torchvision.datasets import MNIST

class CustomMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def remove_example(self, idx):
        previous_length = len(self)
        self.data = self.data[torch.arange(previous_length) != idx]
        self.targets = self.targets[torch.arange(previous_length) != idx]
        