import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from algorithms.DomainDrop import *

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class DDUNet(nn.Module):
    def __init__(self, n_class, args):
        super().__init__()
        
        self.base_model = torchvision.models.resnet18(True)
        
        layer_channels = [64, 64, 128, 256, 512]
            
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]

        self.domain_discriminator_flag = 1
        self.drop_percent = args.DomainDrop_drop_percent
        self.dropout_mode = 0
        
        self.recover_flag = 1
        self.layer_wise_flag = 1
        self.domain_discriminators = nn.ModuleList([
            LayerDiscriminator(
                num_channels=layer_channels[layer],
                num_classes=args.num_domains,
                grl=1,
                reverse=True,
                lambd=args.DomainDrop_lambd,
                wrs_flag=1,
                )
            for i, layer in enumerate([0, 1, 2, 3, 4])])



        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.act_layer = nn.Sigmoid()
        # in_channels = [64, 128, 256, 512]
        # self.fpn_layer =  FPN(in_channels,256,len(in_channels))

    def perform_dropout(self, feature, domain_labels, layer_index, layer_dropout_flag):
        domain_output = None
        if self.domain_discriminator_flag and self.training:
            index = layer_index
            percent = self.drop_percent
            domain_output, domain_mask = self.domain_discriminators[index](
                feature.clone(),
                domain_labels,
                percent=percent,
            )
            if self.recover_flag:
                domain_mask = domain_mask * domain_mask.numel() / domain_mask.sum()
            if layer_dropout_flag:
                feature = feature * domain_mask
        return feature, domain_output
    
    def forward(self, input, domain_labels=None, layer_drop_flag=None):
       
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8

        domain_outputs = []
        for i, layer_out in enumerate([e2, e3, e4, f]):
            _, domain_output = self.perform_dropout(layer_out, domain_labels, layer_index=i+1,
                                                    layer_dropout_flag=layer_drop_flag[i])
            if domain_output is not None:
                domain_outputs.append(domain_output)
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        out = self.act_layer(out)
        
        return f, out, domain_outputs









class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print('outputs[{}].shape = {!r}'.format(i, outputs[i].shape))
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,in_channels,out_channels,num_outs,start_level=0,end_level=-1,
                extra_convs_on_inputs=True,bn=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d( in_channels[i], out_channels,1,bn=bn, bias=not bn,same_padding=True)

            fpn_conv = Conv2d( out_channels, out_channels,3,bn=bn, bias=not bn,same_padding=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = F.interpolate(laterals[i], size=prev_shape, mode='nearest') + laterals[i - 1]

        # build outputs
        # part 1: from original levels
        outs = [ self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) ]


        return tuple(outs)



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=False)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
if __name__ == '__main__':
    model = UNet(1)
    img = torch.rand((1, 3, 224, 224))
    out, feature = model(img)
    print(out.size(), feature.size())
    
