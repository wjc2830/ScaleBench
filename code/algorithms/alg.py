import torch
from torch import nn
from torch import autograd 

from algorithms.DANN import DANN
from algorithms.DeepCoral import CORAL
from algorithms.EFDM import EFDM
from algorithms.IRM import IRM
from algorithms.MMD import MMD
from algorithms.RSC import RSC
from algorithms.VREx import VREx
from algorithms.Mixup import Mixup
from algorithms.SagNet import SagNet
from algorithms.HGP import HGP
from algorithms.CausalIRL import CausalIRL
from algorithms.SAM import SAM
from algorithms.SD import SD
from algorithms.DomainDrop import DomainDrop
from algorithms.InfoBot import InfoBot
from algorithms.SemanticHook import SemanticHook
class DGAlg():
    def __init__(self, args) -> None:
        self.args = args
        
        
        if args.DGAlg in ['Mixup']:
            self.algo = eval(args.DGAlg+f'("{args.mixup_content}", {args.mixupalpha}, {args.batch_size})')
        elif args.DGAlg in ['CORAL', 'MMD', 'IRM', 'RSC', 'SD']:
            self.algo = eval(args.DGAlg+f'({args.batch_size})')
        elif args.DGAlg in ['DANN']:
            self.algo = eval(args.DGAlg+f'({args.batch_size}, {args.DANN_feat_dim}, {args.DANN_disc_alpha})')
        elif args.DGAlg in ['VREx']:
            self.algo = eval(args.DGAlg+f'({args.batch_size})')
            self.anneal_iters = args.anneal_iters
            self.VRexLam = args.penalty_param
            self.TaskLoss = nn.MSELoss(reduction='none').cuda()
            self.tmp_iter = 0
        elif args.DGAlg in ['EFDM']:
            self.algo = eval(args.DGAlg+f'(efdm_content="{args.mixup_content}", num_sample_per_domain={args.batch_size})')

        elif args.DGAlg in ['SagNet']:
            self.algo = eval(args.DGAlg+f'({args.batch_size}, {args.SagNet_pixel_recor})')
            self.eps = args.SagNet_eps
        elif args.DGAlg in ['SAGM', 'GAM']:
            pass
        
        elif args.DGAlg in ['CausalIRL']:
            self.algo = eval(args.DGAlg+f'("{args.CausalIRL_mode}", {args.batch_size})')
        elif args.DGAlg in ['SemanticHook']:
            self.algo = eval(args.DGAlg+f'(noise_scale={args.SemanticHook_NoiseScale}, lambda_0={args.SemanticHook_l0}, alpha={args.SemanticHook_alpha}, beta={args.SemanticHook_beta}, total_iter={args.num_iter})')
        elif args.DGAlg in ['SAM']:
            self.algo = eval(args.DGAlg+f'(num_sample_per_domain={args.batch_size})')
            self.loss_fn = nn.MSELoss().cuda()
    
        elif args.DGAlg in ['DomainDrop']:
            self.algo = eval(args.DGAlg+f'(discriminator_layers={args.DomainDrop_discriminator_layers}, num_sample_per_domain={args.batch_size})')
            self.domain_labels = []
            for domain_index in range(args.num_domains):
                self.domain_labels.extend([domain_index for _ in range(args.batch_size)])
            self.domain_labels = self.domain_labels * 2
            self.domain_labels = torch.tensor(self.domain_labels)#.unsqueeze(1)
        elif args.DGAlg in ['InfoBot', 'InfoBot']:
            self.algo = eval(args.DGAlg+f'("{args.InfoBot_mode}", num_sample_per_domain={args.batch_size})')
       





    def only_feature(self, feature, num_sample_per_domain=8):
        return self.algo(feature, num_sample_per_domain)

    def Mixup_forward(self, model, img=None, gt=None, feature=None):
        if self.args.mixup_content == 'img':
            img, gt_map = self.algo.forward(img=img, gt_map=gt, feature=feature)
            _, pre_map = model(img, gt_map)
            all_loss = model.loss
        else:
            gt_map = gt
            feature, deeper_feature, last_feature = model(img, gt, OnlyEnc=True)
            feature = self.algo.forward(img=None, gt_map=None, feature=feature)
            
            _, pre_map = model(feature, gt, last_feature=last_feature, OnlyDec=True)
            all_loss = model.loss
            
            # print(pre_map.size(), gt.size(), img.size(), gt_map.size())
        return all_loss, pre_map, all_loss, 0, img, gt 
            
    def CORAL_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty = self.algo.forward(feature)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty
        return all_loss, pre_map, head_map_loss, penalty, img, gt 
                     
    def MMD_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty = self.algo.forward(feature)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty
        return all_loss, pre_map, head_map_loss, penalty, img, gt 

    def SAM_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty = 0
        all_loss = self.args.sup_param * head_map_loss
        return all_loss, pre_map, head_map_loss, penalty, img, gt 

    def EFDM_forward(self, model, img, gt):
        if self.args.mixup_content == 'img':
            img = self.algo.forward(img=img)
            _, pre_map = model(img, gt)
            all_loss = model.loss
        else:
            feature, deeper_feature, last_feature = model(img, gt, OnlyEnc=True)
            feature = self.algo.forward(img=None, feature=feature)
            
            _, pre_map = model(feature, gt, last_feature=last_feature, OnlyDec=True)
            all_loss = model.loss
        return all_loss, pre_map, all_loss, 0, img, gt 
            
    def InfoBot_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty, ib_penalty = self.algo.forward(feature, pre_map, gt)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty + self.args.InfoBot_penalty_param * ib_penalty
        return all_loss, pre_map, head_map_loss, penalty + ib_penalty, img, gt 

    def SD_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty = self.algo.forward(pre_map)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty
        return all_loss, pre_map, head_map_loss, penalty, img, gt 


    def SAM_backward(self, all_loss, optimizer, model, img, gt):
        # self.algo.enable_running_stats(model)
        all_loss.backward()
        optimizer.first_step(zero_grad=True)

        # self.algo.disable_running_stats(model)
        self.loss_fn(model(img, gt)[1], gt).backward()  
        optimizer.second_step(zero_grad=True)
               

    def IRM_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty = self.algo.forward(pre_map, gt)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty
        return all_loss, pre_map, head_map_loss, penalty, img, gt 
            
    def DANN_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        # print(feature.size())
        head_map_loss = model.loss
        penalty = self.algo.forward(feature)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty
        return all_loss, pre_map, head_map_loss, penalty, img, gt 
            
    def RSC_forward(self, model, img, gt):
        feature, deeper_feature, last_feature = model(img, gt, OnlyEnc=True, apply_loss=False)
        
        feature = self.algo.forward(feature, deeper_feature)
        _, pre_map = model(feature, gt, last_feature=last_feature, OnlyDec=True, apply_loss=True)
        head_map_loss = model.loss
        penalty = 0
        all_loss = head_map_loss
        return all_loss, pre_map, head_map_loss, penalty, img, gt 
    
    def VREx_forward(self, model, img, gt):
        if self.tmp_iter >= self.anneal_iters:
            penalty_weight = self.VRexLam
        else:
            penalty_weight = 1.0
        self.tmp_iter += 1
        feature, pre_map = model(img, gt, apply_loss=False)
        head_map_loss = self.TaskLoss(pre_map, gt)
        penalty = self.algo.forward(head_map_loss)
        head_map_loss = head_map_loss.mean()
        all_loss = self.args.sup_param * head_map_loss + penalty_weight * penalty
        return all_loss, pre_map, head_map_loss, penalty, img, gt 

    def SagNet_forward(self, model, img, gt):
        
        feature, _, last_feature = model(img, gt, OnlyEnc=True, apply_loss=False)
       
        res_feats, style_feats, res_gts, style_gts, shared_last_feature = self.algo.forward(feature, gt, last_feature)
        # print(res_feats.size(), style_feats.size(), res_gts.size(), style_gts.size())
        _, pre_map = model(res_feats, res_gts, last_feature=shared_last_feature, OnlyDec=True, apply_loss=True)
        head_map_loss = model.loss
        _, adv_pre_map = model(style_feats, style_gts, last_feature=shared_last_feature, OnlyDec=True, apply_loss=False, SagNet=True)
        penalty = - torch.log(adv_pre_map + self.eps).view(feature.size(0), -1).mean(dim=-1).mean()
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty
        return all_loss, pre_map[:(pre_map.size(0)//2)], head_map_loss, penalty, img, res_gts[:(pre_map.size(0)//2)]

    def SAGM_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        all_loss = model.loss
        return all_loss, pre_map, head_map_loss, 0, img, gt

    def ERM_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        all_loss = model.loss
        return all_loss, pre_map, head_map_loss, 0, img, gt

    def DomainDrop_forward(self, model, img, gt):
        img = torch.cat([img, img], dim=0)
        gt = torch.cat([gt, gt], dim=0)
        self.domain_labels = self.domain_labels.to(img.device)
        layer_drop_flag = self.algo.select_layers(self.args.DomainDrop_layer_wise_prob)
        feature, pre_map, domain_logits = model(img, gt, domain_labels=self.domain_labels, 
                                                layer_drop_flag=layer_drop_flag)
        
        head_map_loss = model.loss
        penalty, consis_penalty = self.algo.forward(domain_logits, self.domain_labels, pre_map)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty + self.args.DomainDrop_consis_param * consis_penalty
        return all_loss, pre_map, head_map_loss, penalty+consis_penalty, img, gt 
 
    def GAM_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        all_loss = model.loss
        return all_loss, pre_map, head_map_loss, 0, img, gt


    
    def CausalIRL_forward(self, model, img, gt):
        feature, pre_map = model(img, gt)
        head_map_loss = model.loss
        penalty = self.algo.forward(feature)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty

        return all_loss, pre_map, head_map_loss, penalty, img, gt 

    def SemanticHook_forward(self, model, img, gt):
        feature, _, last_features = model(img, None, OnlyEnc=True)
        _, pre_map = model(feature, gt, last_feature=last_features, OnlyDec=True)
        head_map_loss = model.loss
        penalty, _ = self.algo.forward(img, feature, last_features, gt, model)
        all_loss = self.args.sup_param * head_map_loss + self.args.penalty_param * penalty

        return all_loss, pre_map, head_map_loss, penalty, img, gt 



