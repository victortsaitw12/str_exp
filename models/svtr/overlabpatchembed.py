import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.transforms import transforms as T

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.active = nn.GELU()

    def forward(self, x):
        return self.active(self.bn(self.conv(x)))

class OverlapPatchEmbed(nn.Module):
    """Image to the progressive overlapping Patch Embedding.
    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
        num_layers (int, optional): Number of Conv_BN_Layer. Defaults to 2 and
            limit to [2, 3].
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self, in_channels=3, embed_dims=768, num_layers=2, init_cfg=None):    
        super(OverlapPatchEmbed, self).__init__()
        self.net = nn.Sequential()
        for num in range(num_layers, 0, -1):
            if (num == num_layers):
                _input = in_channels
            _output = embed_dims // (2**(num - 1))
            self.net.add_module(f'ConvModule{str(num_layers - num)}',
                BaseConv(
                    in_channels=_input,
                    out_channels=_output,
                    kernel_size=3,
                    stride=2,
                    padding=1)
            )
            _input = _output

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.
        Returns:
            Tensor: A tensor of shape math:`(N, HW//16, C)`.
        """
        x = self.net(x) # batch, C, H/4, W/4
        x = x.flatten(2) # batch, C, (H/4)*(W/4)
        x = x.permute(0, 2, 1) # N, HW//16, C)
        return x
