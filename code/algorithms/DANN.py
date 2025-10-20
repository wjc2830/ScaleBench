import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Function

from torch import optim
import torch.nn as nn


class FCDiscriminator(nn.Module):
    def __init__(self, input_dim, num_domain, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, ndf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(ndf*4, num_domain, kernel_size=3, stride=1, padding=1)
		

    def forward(self, x):
        x = self.conv1(x)
        # x = self.leaky_relu(x)
        x = self.conv2(x)
        # x = self.leaky_relu(x)
        x = self.conv3(x)
        # x = self.leaky_relu(x)
        x = self.conv4(x)
        # x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x) 

        return x


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN():
    def __init__(self, num_sample_per_domain, feat_dim, disc_alpha, num_domain=3):
        super(DANN, self).__init__()
        self.discriminator = FCDiscriminator(feat_dim, num_domain).cuda()
        self.optimizer = optim.Adam([{'params':self.discriminator.parameters(), 'lr':1e-3, 'weight_decay':1e-5}])
        self.d_loss = nn.CrossEntropyLoss().cuda()
        self.disc_alpha = disc_alpha
        self.num_sample_per_domain = num_sample_per_domain


    @property
    def optim(self):
         return self.optimizer


    def forward(self, features):
        discri_inport = ReverseLayerF.apply(features, self.disc_alpha)
        discri_label = torch.cat([
             torch.full( (self.num_sample_per_domain, 1, features.size(2), features.size(3)),  i, dtype=torch.int64, device='cuda') 
                for i in range(int(features.size(0)/self.num_sample_per_domain))
        ], dim=0)
        self.optimizer.zero_grad()
        discri_outport = self.discriminator(discri_inport)
        # print(discri_outport.size(), features.size())
        discri_outport_flattened = discri_outport.view(-1, discri_outport.size(1), discri_outport.size(2), discri_outport.size(3))
        discri_label_flattened = discri_label.view(-1, discri_label.size(2), discri_label.size(3)).squeeze(1)

        loss = self.d_loss(discri_outport_flattened, discri_label_flattened)       
        return loss