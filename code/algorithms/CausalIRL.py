# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F


class CausalIRL():
    def __init__(self, gaussian, num_sample_per_domain):
        super(CausalIRL, self).__init__()
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"
        self.num_sample_per_domain=num_sample_per_domain
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def forward(self, features):
        penalty = 0
        nmb = len(features)

        first = None
        second = None
        features = torch.cat([F.interpolate(data.unsqueeze(0), scale_factor=0.125).squeeze().view(data.size(0), -1) for data in features], dim=0)

        for i in range(0, nmb, self.num_sample_per_domain):
            slice = np.random.randint(0, self.num_sample_per_domain)
            if first is None:
                first = features[i:i+slice]
                second = features[i+slice: i+self.num_sample_per_domain]
            else:
                first = torch.cat((first, features[i:i+slice]), 0)
                second = torch.cat((second, features[i+slice: i+self.num_sample_per_domain]), 0)
        
        if len(first) > 1 and len(second) > 1:
            penalty = torch.nan_to_num(self.mmd(first, second))
        else:
            penalty = torch.tensor(0)
        return penalty

       
       
 