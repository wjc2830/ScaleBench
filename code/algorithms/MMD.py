# coding=utf-8
import torch
import torch.nn.functional as F



class MMD():
    def __init__(self, num_sample_per_domain=3):
        super(MMD, self).__init__()
        self.num_sample_per_domain = num_sample_per_domain
        self.kernel_type = "gaussian"

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
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def forward(self, feats):
        penalty = 0
        nmb = len(feats)
        features = torch.cat([F.interpolate(data.unsqueeze(0), scale_factor=0.125).squeeze().view(data.size(0), -1) for data in feats], dim=0)

        for i in range(0, nmb, self.num_sample_per_domain):
            for j in range(i + self.num_sample_per_domain, nmb, self.num_sample_per_domain):
                penalty += self.mmd(features[i: i+self.num_sample_per_domain],
                                    features[j: j+self.num_sample_per_domain])
        return penalty