import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class EFDM():
    

    def __init__(self, efdm_content='img', shuffle_intra_batch=False, num_sample_per_domain=3):
        super(EFDM, self).__init__()
        self.efdm_content = efdm_content
        self.shuffle_intra_batch = shuffle_intra_batch
        self.num_sample_per_domain = num_sample_per_domain

    @staticmethod
    def exact_feature_distribution_matching(content, style):
        if len(content.size()) == 3:
            B, n_patch, hidden = content.size()
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            content = content.permute(0, 2, 1)
            content = content.contiguous().view(B, hidden, h, w)
            style = style.permute(0, 2, 1)
            style = style.contiguous().view(B, hidden, h, w)
        assert (content.size() == style.size()) ## content and style features should share the same shape
        B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
        _, index_content = torch.sort(content.view(B,C,-1))  ## sort content feature
        value_style, _ = torch.sort(style.view(B,C,-1))      ## sort style feature
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B,C,-1) + value_style.gather(-1, inverse_index) - content.view(B,C,-1).detach()
        return transferred_content.view(B, C, W, H)

    def referContent(self, index_src, index_tra, img=None, feature=None):
        if img is not None:
            return img[index_src].unsqueeze(0),\
                    img[index_tra].unsqueeze(0)
        else:
            return feature[index_src].unsqueeze(0), feature[index_tra].unsqueeze(0)

    def forward(self, feature=None, img=None):
        nmb = len(img) if img is not None else len(feature)
        left_indexes = list(np.arange(nmb))
        
        res_imgs, res_feats = [], []
        for i in range(0, nmb, self.num_sample_per_domain):
            tmp_candidates = list(np.arange(nmb))
            for elem in range(i, i+self.num_sample_per_domain):
                tmp_candidates.remove(elem)  
            tmp_indexes = np.random.choice(tmp_candidates, size=(self.num_sample_per_domain, ))
            for j in range(self.num_sample_per_domain):
                src_index, tra_index = i+j, tmp_indexes[j]
                
                srcContent, traContent = self.referContent(src_index, tra_index, img, feature)
                srcIntra = self.exact_feature_distribution_matching(srcContent, traContent)
                if self.efdm_content == 'img': 
                    res_imgs.append(srcIntra) 
                else:
                    res_feats.append(srcIntra)
      
        if self.efdm_content == 'img': 
            res_imgs = torch.cat(res_imgs, dim=0)
            return res_imgs
        else:
            res_feats = torch.cat(res_feats, dim=0)
            return res_feats