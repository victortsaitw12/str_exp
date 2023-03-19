import torch.nn as nn

class ConvMixer(nn.Module):
    """The conv Mixer.
    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [3, 3].
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self, embed_dims, num_heads=8, input_shape=[8, 25], local_k=[3, 3], init_cfg=None):
        super(ConvMixer, self).__init__()
        self.input_shape = input_shape
        self.embed_dims = embed_dims
        self.local_mixer = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=local_k,
            stride=1,
            padding=(local_k[0] // 2, local_k[1] // 2),
            groups=num_heads
        )

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, HW, C)`.
        Returns:
            torch.Tensor: Tensor: A tensor of shape math:`(N, HW, C)`.
        """
        h, w = self.input_shape
        x = x.permute(0, 2, 1)
        x = x.reshape([-1, self.embed_dims, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x
