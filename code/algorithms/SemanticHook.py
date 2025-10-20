# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy


class SemanticHook():
    def __init__(self, noise_scale, lambda_0, alpha, beta, total_iter):
        super(SemanticHook, self).__init__()
        self.visit_time = 0
        self.lambda_0, self.alpha, self.beta, self.total_iter = lambda_0, alpha, beta, total_iter
        self.noise_scale = noise_scale
  
    def update_lambda(self, update=True):
        if update:
            self.visit_time += 1
        return self.lambda_0 * (1 - (1 + self.alpha * (self.visit_time / self.total_iter))**(-self.beta))

    def forward(self, img, features, last_features, gt_map, model):
        noise = (torch.randn_like(img) * self.noise_scale).to(img.device)
        aug_img = deepcopy(img.detach()) + noise
        aug_feature, _, _ = model(aug_img, None, OnlyEnc=True)
        hook_feature = (aug_feature - self.update_lambda(False) * features) * (1 - self.update_lambda(True))
        _, pre_map = model(hook_feature, gt_map, last_feature=last_features, apply_loss=True, OnlyDec=True)
        
        penalty = model.loss
        return penalty, pre_map
        