import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class IRM():
    """Invariant Risk Minimization"""

    def __init__(self, num_sample_per_domain=3):
        super(IRM, self).__init__()
        self.num_sample_per_domain = num_sample_per_domain

    @staticmethod
    def _irm_penalty(pre_map, gt_map):
        scale = torch.ones_like(pre_map).to('cuda').requires_grad_()
        loss = F.mse_loss(pre_map*scale, gt_map)
        all_g = autograd.grad(outputs=loss, inputs=scale, create_graph=True)[0]
        result = torch.norm(all_g, p=2)
        return result

    def forward(self, pre_map, gt_map):
        penalty = 0
        nmb = len(pre_map)
        for i in range(0, nmb, self.num_sample_per_domain):
            penalty += self._irm_penalty(pre_map[i: i+self.num_sample_per_domain], gt_map[i: i+self.num_sample_per_domain])

        return penalty