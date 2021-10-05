from pathlib import Path

import torch
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import DataLoader

from utils import *


class EfficientTransformerSR:
    def __init__(self, configs):
        ...
    def train(self):
        ...
    def valid(self):
        ...
    def test(self):
        ...

if __name__ == '__main__':
    a = EfficientTransformerSR(Paths / "configs/train.yaml"  )