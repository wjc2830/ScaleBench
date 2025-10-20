import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class Mixup():
    def __init__(self, mixup_content='img', mixupalpha=.2, num_sample_per_domain=3):
        super(Mixup, self).__init__()
        self.mixup_content = mixup_content
        self.mixupalpha = mixupalpha
        self.num_sample_per_domain=num_sample_per_domain


    def mixup(self, content, style, c_gt=None, s_gt=None):
        lam = np.random.beta(self.mixupalpha, self.mixupalpha)
        img = (lam * content + (1 - lam) * style).to(content.device)
        if s_gt is not None:
            gt = (lam * c_gt + (1 - lam) * s_gt).to(content.device)
            return img, gt
        return img
    
    
    def mixContent(self, index_src, index_tra, img=None, gt_map=None, feature=None):
        if img is not None:
            return img[index_src].unsqueeze(0), gt_map[index_src].unsqueeze(0),\
                    img[index_tra].unsqueeze(0), gt_map[index_tra].unsqueeze(0)
        else:
            return feature[index_src].unsqueeze(0), feature[index_tra].unsqueeze(0)

    def forward(self, img=None, gt_map=None, feature=None):
        nmb = len(img) if img is not None else len(feature)
        left_indexes = list(np.arange(nmb))
        
        res_imgs, res_gts, res_feats = [], [], []
        for i in range(0, nmb, self.num_sample_per_domain):
            tmp_candidates = left_indexes.copy()
            for elem in range(i, i+self.num_sample_per_domain):
                tmp_candidates.remove(elem)  
            tmp_indexes = np.random.choice(tmp_candidates, size=(self.num_sample_per_domain, ))
            for j in range(self.num_sample_per_domain):
                src_index, tra_index = i+j, tmp_indexes[j]
                if self.mixup_content == 'img': 
                    src_img, src_gt, tra_img, tra_gt = self.mixContent(src_index, tra_index, 
                                                                       img, gt_map, feature)
                    mixed_img, mixed_gt = self.mixup(src_img, tra_img, src_gt, tra_gt)
                    res_imgs.append(mixed_img)
                    res_gts.append(mixed_gt)
                else:
                    src_feat, tra_feat = self.mixContent(src_index, tra_index, 
                                                        img, gt_map, feature)
                    mixed_feat = self.mixup(src_feat, tra_feat)
                    res_feats.append(mixed_feat)
                
        if self.mixup_content == 'img': 
            res_imgs = torch.cat(res_imgs, dim=0)
            res_gts = torch.cat(res_gts, dim=0)
            return res_imgs, res_gts
        else:
            res_feats = torch.cat(res_feats, dim=0)
            return res_feats