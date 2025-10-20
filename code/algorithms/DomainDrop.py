import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def mask_selection(scores, percent, wrs_flag):
    # input: scores: BxN
    batch_size = scores.shape[0]
    num_neurons = scores.shape[1]
    drop_num = int(num_neurons * percent)

    if wrs_flag == 0:
        # according to scores
        threshold = torch.sort(scores, dim=1, descending=True)[0][:, drop_num]
        threshold_expand = threshold.view(batch_size, 1).expand(batch_size, num_neurons)
        mask_filters = torch.where(scores > threshold_expand, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
    else:
        # add random modules
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_min = scores.min(dim=1, keepdim=True)[0]
        scores = (scores - score_min) / (score_max - score_min)
        
        r = torch.rand(scores.shape).cuda()  # BxC
        key = r.pow(1. / scores)
        threshold = torch.sort(key, dim=1, descending=True)[0][:, drop_num]
        threshold_expand = threshold.view(batch_size, 1).expand(batch_size, num_neurons)
        mask_filters = torch.where(key > threshold_expand, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())

    mask_filters = 1 - mask_filters  # BxN
    return mask_filters


def filter_dropout_channel(scores, percent, wrs_flag):
    # scores: BxCxHxW
    batch_size, channel_num, H, W = scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
    channel_scores = nn.AdaptiveAvgPool2d((1, 1))(scores).view(batch_size, channel_num)
    # channel_scores = channel_scores / channel_scores.sum(dim=1, keepdim=True)
    mask = mask_selection(channel_scores, percent, wrs_flag)   # BxC
    mask_filters = mask.view(batch_size, channel_num, 1, 1)
    return mask_filters


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None


def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)


class LayerDiscriminator(nn.Module):
    def __init__(self, num_channels, num_classes, grl=True, reverse=True, lambd=0.0, wrs_flag=1):
        super(LayerDiscriminator, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = nn.Linear(num_channels, num_classes)
        self.softmax = nn.Softmax(0)
        self.num_channels = num_channels

        self.grl = grl
        self.reverse = reverse
        self.lambd = lambd

        self.wrs_flag = wrs_flag

    def scores_dropout(self, scores, percent):
        mask_filters = filter_dropout_channel(scores=scores, percent=percent, wrs_flag=self.wrs_flag)
        mask_filters = mask_filters.cuda()  # BxCx1x1
        return mask_filters

    def norm_scores(self, scores):
        score_max = scores.max(dim=1, keepdim=True)[0]
        score_min = scores.min(dim=1, keepdim=True)[0]
        scores_norm = (scores - score_min) / (score_max - score_min)
        return scores_norm

    def get_scores(self, feature, labels, percent=0.33):
        weights = self.model.weight.clone().detach()  # num_domains x C
        domain_num, channel_num = weights.shape[0], weights.shape[1]
        batch_size, _, H, W = feature.shape[0], feature.shape[1], feature.shape[2], feature.shape[3]

        weight = weights[labels].view(batch_size, channel_num, 1).expand(batch_size, channel_num, H * W)\
            .view(batch_size, channel_num, H, W)

        right_score = torch.mul(feature, weight)
        right_score = self.norm_scores(right_score)

        # right_score_masks: BxCxHxW
        right_score_masks = self.scores_dropout(right_score, percent=percent)
        return right_score_masks

    def forward(self, x, labels, percent=0.33):
        if self.grl:
            x = grad_reverse(x, self.lambd, self.reverse)

        feature = x.clone().detach()  # BxCxHxW
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # BxC
        y = self.model(x)

        # This step is to compute the 0-1 mask, which indicate the location of the domain-related information.
        # mask_filters: {0 / 1} BxCxHxW
        mask_filters = self.get_scores(feature, labels, percent=percent)
        return y, mask_filters
    

class DomainDrop():
    def __init__(self, discriminator_layers, num_sample_per_domain=3):
        super(DomainDrop, self).__init__()
        self.num_sample_per_domain=num_sample_per_domain
        self.discriminator_layers = discriminator_layers
        self.domain_criterion = nn.CrossEntropyLoss()
        self.consis_criterion = nn.MSELoss()
    
    def select_layers(self, layer_wise_prob):
        layer_index = np.random.randint(len(self.discriminator_layers), size=1)[0]
        layer_select = self.discriminator_layers[layer_index]
        layer_drop_flag = [0, 0, 0, 0]
        if random.random() <= layer_wise_prob:
            layer_drop_flag[layer_select - 1] = 1
        return layer_drop_flag

    def forward(self, domain_logits, domain_gt, pre_map):
        penalty = 0
        nmb = len(domain_gt) 
        domain_losses = []
        for i, logit in enumerate(domain_logits):
            domain_loss = self.domain_criterion(logit, domain_gt)
            domain_losses.append(domain_loss)
        penalty = sum(domain_losses)
        pre_map_1, pre_map_2 = pre_map[: int(nmb/2)], pre_map[int(nmb/2): ]
        consis_penalty = ( self.consis_criterion(pre_map_1, pre_map_2.detach()) + self.consis_criterion(pre_map_2, pre_map_1.detach()) ) / 2
        return penalty, consis_penalty
        


