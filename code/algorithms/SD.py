import torch
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

class SD():
    def __init__(self, num_sample_per_domain=3):
        super(SD, self).__init__()
        self.num_sample_per_domain=num_sample_per_domain
    def forward(self, pre_map):
        penalty = (pre_map ** 2).mean()
        return penalty