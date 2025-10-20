import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from einops import rearrange
# from misc.utils import *

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)
        out_feats = []

        for layer in self.encoder_layers:
            out = layer(out)
            out_feats.append(out)

        out = self.norm(out)
        return out, out_feats


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        
        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        self.deeper_feat = nn.Sequential(
            nn.Conv2d(
                in_channels=768,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )

        self.de_pred = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.Sigmoid()
        )
        # initialize_weights(self.de_pred)
        # initialize_weights(self.deeper_feat)


    def forward(self, x):
        emb = self.embedding(x)     # (n, c, gh, gw)
        emb = rearrange(emb, "b c h w -> b (h w) c")
        
        feat, out_feats = self.transformer(emb)
        feat = rearrange(feat, "b (h w) c -> b c h w", h=int(feat.size(1)**.5))
        deeper_feat = self.deeper_feat(feat)
        pre_map = self.de_pred(deeper_feat)
        
        # classifier
        return feat, pre_map

def get_ViT_Base(vit_dir):
    model = VisionTransformer(
        image_size=(512, 512),
        patch_size=(16, 16),
        emb_dim = 768,
        mlp_dim = 3072,
        num_heads = 12,
        num_layers = 12,
        attn_dropout_rate = 0.0,
        dropout_rate = 0.0
    )
    model_dict = model.state_dict()
    model_keys = list(model_dict.keys())
    pre_trained_dict = torch.load(vit_dir)
    new_dict = {}
    num_not_exists, num_not_matched, num_matched = 0, 0, 0
    for k, v in pre_trained_dict['state_dict'].items():
        if k not in model_keys:
            print('not exists', k)
            num_not_exists += 1
        else:
            new_dict[k] = model_dict[k]
            if v.size() != model_dict[k].size():
                print(' not matched', k)
                num_not_matched += 1
            else:
                num_matched += 1
    print(num_not_exists, num_not_matched, num_matched)
    model.load_state_dict(new_dict, strict=False)
    return model
    





