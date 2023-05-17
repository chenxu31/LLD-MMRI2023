# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import pdb
from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from models import util


NUM_DOWNSAMPLING = 3
NUM_FILTERS = (32, 64, 128, 256, 512, 1024)

    
class CNNBase(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=112, in_ch=3, num_classes=1000, num_rb=12, norm_type="group", act_type="relu"):
        super().__init__()

        self.num_classes = num_classes

        nets = [
            torch.nn.Conv3d(in_ch, NUM_FILTERS[0], 3, padding=1),
            util.native_norm(norm_type, NUM_FILTERS[0], dim=3),
            util.native_act(act_type),
        ]
        for i in range(NUM_DOWNSAMPLING):
            nets.append(util.DepthwiseResidualBlock(NUM_FILTERS[i], dim=3, kernel_size=3, norm_type=norm_type, act_type=act_type))
            nets.append(util.depthwise_conv(NUM_FILTERS[i], NUM_FILTERS[i + 1], dim=3, kernel_size=3, stride=2 if i == 0 else (1, 2, 2)))
            nets.append(util.native_norm(norm_type, NUM_FILTERS[i + 1], dim=3))
            nets.append(util.native_act(act_type))

        for i in range(num_rb):
            nets.append(util.DepthwiseResidualBlock(NUM_FILTERS[NUM_DOWNSAMPLING], dim=3, kernel_size=3, norm_type=norm_type, act_type=act_type))

        nets.append(util.depthwise_conv(NUM_FILTERS[NUM_DOWNSAMPLING], NUM_FILTERS[NUM_DOWNSAMPLING], dim=3, kernel_size=3))
        nets.append(util.native_norm(norm_type, NUM_FILTERS[NUM_DOWNSAMPLING], dim=3))
        nets.append(util.native_act(act_type))
        nets.append(torch.nn.AdaptiveAvgPool3d(1))
        nets.append(torch.nn.Flatten())
        nets.append(torch.nn.Linear(NUM_FILTERS[NUM_DOWNSAMPLING], num_classes))

        self.net = torch.nn.Sequential(*nets)

    def forward(self, x):
        output = self.net(x)
        return output


@register_model
def cnn_base(num_classes=2, num_phase=8, pretrained=None, **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = CNNBase(in_ch=num_phase, num_classes=num_classes)

    return model