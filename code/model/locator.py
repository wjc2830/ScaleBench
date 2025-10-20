from model.HR_Net.seg_hrnet import get_seg_model
from model.HR_Net.seg_hr_DomainDrop import get_seg_DDmodel
from model.UNet import UNet
from model.UNetDomainDrop import DDUNet
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models
import torch.autograd as autograd
from model.ViTUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.ViTUNet.vit_seg_modelingDomainDrop import VisionTransformer_DD as ViT_seg_DD
from model.ViTUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class Crowd_locator(nn.Module):
    def __init__(self, net_name, gpu_id, sag_net=False, args=None):
        super(Crowd_locator, self).__init__()
        self.net_name = net_name
        if net_name == 'HR_Net':
            self.Extractor = get_seg_model(sag_net, args)
        elif net_name == 'DomainDropHR_Net':
            self.Extractor = get_seg_DDmodel(args)
            
        elif net_name == 'Res18UNet':
            self.Extractor = UNet(1, sag_net)
        elif net_name == 'DomainDropRes18UNet':
            self.Extractor = DDUNet(1, args)
        elif net_name == 'ViTUNet':
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = 1
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(512 / 16), int(512 / 16))
            self.Extractor = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes, sag_Net=sag_net)
            self.Extractor.load_from(weights=np.load(config_vit.pretrained_path))
        elif net_name == 'DomainDropViTUNet':
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = 1
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(512 / 16), int(512 / 16))
            self.Extractor = ViT_seg_DD(config_vit, img_size=512, num_classes=config_vit.n_classes, sag_Net=sag_net, args=args)
            self.Extractor.load_from(weights=np.load(config_vit.pretrained_path))
        
        if len(gpu_id) > 1:
            self.Extractor = torch.nn.DataParallel(self.Extractor).cuda()
        else:
            self.Extractor = self.Extractor.cuda()


    @property
    def loss(self):
        return  self.head_map_loss

    def forward(self, img, mask_gt, last_feature=None, OnlyEnc=False, OnlyDec=False, apply_loss=True, SagNet=False, 
                domain_labels=None, layer_drop_flag=None, mode = 'train'):
        if OnlyEnc:
            feature, deeper_feature, last_feature = self.Extractor(img, only_Enc=OnlyEnc, only_Dec=OnlyDec)
            return feature, deeper_feature, last_feature
        elif OnlyDec:
            feature, pre_map = self.Extractor(img, last_feature, only_Dec=OnlyDec, sag_Net=SagNet)
        else:
            if 'DomainDrop' not in self.net_name:
                feature, pre_map = self.Extractor(img)
            else:
                feature, pre_map, domain_logit = self.Extractor(img, domain_labels=domain_labels, layer_drop_flag=layer_drop_flag if layer_drop_flag is not None else [0, 0, 0, 0])
                if mode == 'train' and apply_loss:
                    assert pre_map.size(2) == mask_gt.size(2)   
                    self.head_map_loss = F.mse_loss(pre_map, mask_gt)

                    return feature, pre_map, domain_logit
                else:
                    return feature, pre_map

        if mode == 'train' and apply_loss:
            assert pre_map.size(2) == mask_gt.size(2)   
            self.head_map_loss = F.mse_loss(pre_map, mask_gt)

        return feature, pre_map

    def test_forward(self, img):
        feature, pre_map = self.Extractor(img)

        return feature, pre_map

