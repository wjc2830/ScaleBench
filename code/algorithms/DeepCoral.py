# coding=utf-8
import torch
import torch.nn.functional as F


class CORAL():
    def __init__(self, num_sample_per_domain=3):
        super(CORAL, self).__init__()
        self.num_sample_per_domain=num_sample_per_domain
    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = torch.matmul(cent_x.t(), cent_x) / (len(x) - 1)
        cova_y = torch.matmul(cent_y.t(), cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def forward(self, feats):
        penalty = 0
        nmb = len(feats)

        features = torch.cat([F.interpolate(data.unsqueeze(0), scale_factor=0.125).squeeze().view(data.size(0), -1) for data in feats], dim=0)

        for i in range(0, nmb, self.num_sample_per_domain):
            for j in range(i + self.num_sample_per_domain, nmb, self.num_sample_per_domain):
                penalty += self.coral(features[i: i+self.num_sample_per_domain], features[j: j+self.num_sample_per_domain])
        return penalty
        