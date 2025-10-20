# coding=utf-8
import torch
import torch.nn.functional as F
import torch.autograd as autograd


class InfoBot():
    def __init__(self, mode_algo='ERM', num_sample_per_domain=3):
        super(InfoBot, self).__init__()
        self.num_sample_per_domain=num_sample_per_domain
        self.mode_algo = mode_algo
    
    @staticmethod
    def _ib_penalty(features):
        return features.var(dim=0).mean()

    @staticmethod
    def _irm_penalty(pre_map, gt_map):
        scale = torch.ones_like(pre_map).to('cuda').requires_grad_()
        loss = F.mse_loss(pre_map*scale, gt_map)
        all_g = autograd.grad(outputs=loss, inputs=scale, create_graph=True)[0]
        result = torch.norm(all_g, p=2)
        return result

    def forward(self, feats, pre_map, gt_map):
        penalty, ib_penalty = 0, 0
        nmb = len(feats)

        
        for i in range(0, nmb, self.num_sample_per_domain):
            ib_penalty += self._ib_penalty(feats[i: i+self.num_sample_per_domain])
            if self.mode_algo == 'IRM':
                penalty += self._irm_penalty(pre_map[i: i+self.num_sample_per_domain], gt_map[i: i+self.num_sample_per_domain])

        return penalty, ib_penalty
        