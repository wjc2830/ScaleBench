# coding=utf-8
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import pdb
import copy
import numpy as np
from collections import OrderedDict

class HGP():
    def __init__(self, penalty_alpha, penalty_beta, num_sample_per_domain=3):
        super(HGP, self).__init__()
        self.num_sample_per_domain=num_sample_per_domain
        self.penalty_alpha, self.penalty_beta = penalty_alpha, penalty_beta

    def forward(self, model, img, pre_map, gt_map):
        
        nmb = len(pre_map)
        envs = []
        for i in range(0, nmb, self.num_sample_per_domain):
            env = {}
            env['nll'] = F.mse_loss(pre_map[i: i+self.num_sample_per_domain], gt_map[i: i+self.num_sample_per_domain])
            env['sadg'], env['grad'] = self.compute_sadg_penalty(model, pre_map[i: i+self.num_sample_per_domain], gt_map[i: i+self.num_sample_per_domain])
            envs.append(env)
        
        
        train_nll = torch.stack([env['nll'] for env in envs]).mean()

        mean_grad = autograd.grad(train_nll, model.Extractor.parameters(),create_graph=True, retain_graph=True)
        flatten_mean_grad = self._flatten_grad(mean_grad)
        norm_of_mean_grad=flatten_mean_grad.pow(2).sum().sqrt()
        norm_of_mean_grad = norm_of_mean_grad+ 1e-16
        grad_of_norm_of_mean_grad = autograd.grad(norm_of_mean_grad, model.Extractor.parameters(), create_graph=True,retain_graph=True)
        flatten_grad_of_norm_of_mean_grad = self._flatten_grad(grad_of_norm_of_mean_grad)
        mean_hessian_grad= torch.mul(norm_of_mean_grad,flatten_grad_of_norm_of_mean_grad) 

        loss = train_nll.clone()
        
       
        sadg_penalty_list = []
        all_flatten_grads = [self._flatten_grad(env['grad']) for env in envs]

        
        grads_of_norm_of_grad = [autograd.grad(env['sadg'], model.Extractor.parameters(), create_graph=True,retain_graph=True) for env in envs]
        all_flatten_grads_of_norm_of_grad = [self._flatten_grad(grad_of_norm_of_grad) for grad_of_norm_of_grad in grads_of_norm_of_grad]

        hessian_grad = [torch.mul(envs[k]['sadg'],f_grad) for k, f_grad in enumerate(all_flatten_grads_of_norm_of_grad)]
        
        if len(envs) > 0:
            for i in range(len(all_flatten_grads)):
                sadg_penalty_list.append(self.penalty_alpha *(hessian_grad[i] - mean_hessian_grad.detach()).pow(2).sum() + self.penalty_beta * (all_flatten_grads[i] - flatten_mean_grad.detach()).pow(2).sum() )

            N = len(sadg_penalty_list)
            sadg_penalty = torch.stack(sadg_penalty_list).sum()/len(envs)
        else:
             sadg_penalty = torch.stack([self.penalty_alpha * torch.flatten(hessian_grad[0]).pow(2).sum(),self.penalty_beta* envs[0]['sadg']]).sum()


        loss = sadg_penalty + loss
        pdb.set_trace()
        
        return train_nll, sadg_penalty, loss
    
    def compute_sadg_penalty(self, model, pre_map, gt):
        gradient_norm=[]
        numels=[]
        loss = F.mse_loss(pre_map, gt)
        grads = autograd.grad(loss, model.parameters(), create_graph=True,retain_graph=True)
        for grad in grads:
            grad = grad + 1e-16
            gradient_norm.append(torch.norm(grad, p=2))
            numels.append(torch.numel(grad))
        gradient_loss = torch.norm(torch.stack(gradient_norm), p=2)
        return  gradient_loss, grads

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad
        



