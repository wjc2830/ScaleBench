import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class VREx():
    """Invariant Risk Minimization"""

    def __init__(self, num_sample_per_domain):
        super(VREx, self).__init__()
        self.num_sample_per_domain = num_sample_per_domain
        
       

    @staticmethod
    def _vrex_penalty(loss):
        return (loss - loss.mean()) ** 2

    def forward(self, loss):
        penalty = 0
        nmb = len(loss)
        for i in range(0, nmb, self.num_sample_per_domain):
            penalty += (self._vrex_penalty(loss[i: i+self.num_sample_per_domain]))

        return (self.num_sample_per_domain * penalty / nmb).mean()
    