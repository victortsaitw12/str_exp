import torch.nn as nn

class MerigingBlock(nn.Module):
    """The last block of any stage, except for the last stage.
    Args:
        in_channels (int): The channels of input.
        out_channels (int): The channels of output.
        types (str, optional): Which downsample operation of ['Pool', 'Conv'].
            Defaults to 'Pool'.
        stride (Union[int, Tuple[int, int]], optional): Stride of the Conv.
            Defaults to [2, 1].
        act (bool, optional): activation function. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
          in_channels,
          out_channels,
          types='Pool',
          stride=[2, 1],
          act=None,
          init_cfg=None):
        super(MerigingBlock, self).__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1)
        self.norm = nn.LayerNorm(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H/2, W, 2C)`.
        """
        if self.types == 'Pool':
            x = (self.avgpool(x) + self.maxpool(x)) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))

        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out