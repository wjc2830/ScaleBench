import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class RSC():
    def __init__(self, num_sample_per_domain):
        super(RSC, self).__init__()
        self.num_sample_per_domain = num_sample_per_domain
        self.drop_f = (1 - 1/3) * 100
        self.drop_b = (1 - 1/3) * 100

    def update(self, feature, deeper_feature):

        all_g = autograd.grad(outputs=deeper_feature.sum(), inputs=feature)[0]
        if len(all_g.shape) == 3:
            B, n_patch, hidden = all_g.size()
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            all_g = all_g.permute(0, 2, 1)
            all_g = all_g.contiguous().view(B, hidden, h, w)
            feature = feature.permute(0, 2, 1)
            feature = feature.contiguous().view(B, hidden, h, w)

        ori_size = all_g.size()
        all_g = all_g.view(all_g.size(0), -1)
        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()
        mask_f = mask_f.view(ori_size[0], ori_size[1], ori_size[2], ori_size[3])
        # Equation (3): mute top-gradient-percentile activations
        muted_feat = feature * mask_f

        return muted_feat

    
    def forward(self, feature, deeper_feature):
        # muted_feats = []
        # for num_b in range(0, feature.size(0), self.num_sample_per_domain):
        #     muted_feats.append(self.update(feature[num_b: num_b+self.num_sample_per_domain], deeper_feature[num_b: num_b+self.num_sample_per_domain]))
        # return torch.cat(muted_feats, dim=0)
        return self.update(feature, deeper_feature)
