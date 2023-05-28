import sys
from typing import Any

import torch
from torch import nn, Tensor
from model.modules.layers import Conv2dBnReLU
from model.modules.attention import Attention

__all__ = ["DeepSupervisionHead"]


class DeepSupervisionBlock(nn.Sequential):
    def __init__(self, in_channels, attention_name, num_classes, use_separable_conv=False):
        conv = Conv2dBnReLU(in_channels, in_channels // 4, 1, use_depth_wise_separable_conv=use_separable_conv)
        attention = Attention(name=attention_name, in_channels=in_channels // 4)
        cls = nn.Conv2d(in_channels // 4, num_classes, 1)
        super().__init__(conv, attention, cls)


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, attention_name, num_classes: int = 1, use_separable_conv=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for in_channel in in_channels[::-1][:3]:
            self.blocks.append(DeepSupervisionBlock(in_channel, attention_name, num_classes, use_separable_conv=use_separable_conv))

    def forward(self, features) -> list[Any]:
        outs = []
        for index, feature in enumerate(features[::-1][:3]):
            x = self.blocks[index](feature)
            outs.append(x)
        return outs
