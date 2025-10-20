import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd


import torch
import torch.nn as nn


class SagRandomization():
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x_1, x_2): # supervised by y_1
        if len(x_1.size()) == 3:
            B, n_patch, hidden = x_1.size()
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            x_1 = x_1.permute(0, 2, 1)
            x_1 = x_1.contiguous().view(B, hidden, h, w)

            x_2 = x_2.permute(0, 2, 1)
            x_2 = x_2.contiguous().view(B, hidden, h, w)
        N, C, H, W = x_1.size()
        x_1 = x_1.view(N, C, -1)
        x_2 = x_2.view(N, C, -1)

        mean_1 = x_1.mean(-1, keepdim=True)
        var_1 = x_1.var(-1, keepdim=True)
        
        mean_2 = x_2.mean(-1, keepdim=True)
        var_2 = x_2.var(-1, keepdim=True)

        x_1 = (x_1 - mean_1) / (var_1 + self.eps).sqrt()
        x_2 = (x_2 - mean_2) / (var_2 + self.eps).sqrt()
        
        alpha = torch.rand(N, 1, 1)
        alpha = alpha.cuda()
        mean = alpha * mean_1 + (1 - alpha) * mean_2
        var = alpha * var_1 + (1 - alpha) * var_2

        x_pre = x_1 * (var + self.eps).sqrt() + mean
        x_pre = x_pre.view(N, C, H, W)

        x_style = x_2 * (var_1 + self.eps).sqrt() + mean_1
        x_style = x_style.view(N, C, H, W)
        return x_pre, x_style


class ContentRandomization():
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x_1, x_2): # supervised by y_2
        N, C, H, W = x_1.size()
        x_1 = x_1.view(N, C, -1)
        x_2 = x_2.view(N, C, -1)

        mean_1 = x_1.mean(-1, keepdim=True)
        var_1 = x_1.var(-1, keepdim=True)
        
        mean_2 = x_2.mean(-1, keepdim=True)
        var_2 = x_2.var(-1, keepdim=True)

        x_1 = (x_1 - mean_1) / (var_1 + self.eps).sqrt()
        x_2 = (x_2 - mean_2) / (var_2 + self.eps).sqrt()
        

        

        return x



class SagNet():
    def __init__(self, num_sample_per_domain, pixel_recor=False):
        super(SagNet, self).__init__()
        self.SR = SagRandomization()
        self.num_sample_per_domain = num_sample_per_domain
        self.pixel_recor = pixel_recor

    def forward(self, feature, gts, last_feature):
        nmb = len(feature)
        if last_feature is None:
            last_feature = []
        
        
        res_feats, style_feats = [], []
        res_gts, style_gts = [], []
        
        shared_last_feature = [[] for _ in range(len(last_feature))]
        
        for i in range(0, nmb, self.num_sample_per_domain):
            for j in range(i, nmb, self.num_sample_per_domain):
                pre_feat, style_feat = self.SR(feature[i: i+self.num_sample_per_domain], feature[j: j+self.num_sample_per_domain])
                for num_layer in range(len(last_feature)):
                    shared_last_feature[num_layer].append(last_feature[num_layer][i: i+self.num_sample_per_domain])
                res_feats.append(pre_feat)
                style_feats.append(style_feat)
                res_gts.append(gts[i: i+self.num_sample_per_domain])
                if self.pixel_recor:
                    style_gts.append(gts[i: i+self.num_sample_per_domain])
                else:
                    style_gts.append(gts[j: j+self.num_sample_per_domain])
        res_feats = torch.cat(res_feats, dim=0)
        style_feats = torch.cat(style_feats, dim=0)
        res_gts = torch.cat(res_gts, dim=0)
        style_gts = torch.cat(style_gts, dim=0)
        for num_layer in range(len(last_feature)):
            shared_last_feature[num_layer] = torch.cat(shared_last_feature[num_layer], dim=0)
        return res_feats, style_feats, res_gts, style_gts, shared_last_feature
            