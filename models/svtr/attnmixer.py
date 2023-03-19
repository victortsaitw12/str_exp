import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class AttnMixer(nn.Module):
    """One of mixer of {'Global', 'Local'}. Defaults to Global Mixer.
    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        mixer (str, optional): The mixer type, choices are 'Global' and
            'Local'. Defaults to 'Global'.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [7, 11].
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        attn_drop (float, optional): Attn dropout probability. Defaults to 0.0.
        proj_drop (float, optional): Proj dropout layer. Defaults to 0.0.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """
    def __init__(self, embed_dims, num_heads=8, mixer='Global',
        input_shape=[8, 25], local_k=[7, 11],
        qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0,
        init_cfg=None ):
        super(AttnMixer, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_shape = input_shape
        if input_shape is not None:
            height, width = input_shape
            self.input_size = height * width
            self.embed_dims = embed_dims

        if mixer == 'Local' and input_shape is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(
                [height * width, height + hk - 1, width + wk - 1],
                dtype=torch.float32)
            for h in range(0, height):
                for w in range(0, width):
                    mask[h * width + w, h:h + hk, w:w + wk] = 0.
            mask = mask[:, hk // 2:height + hk // 2,
                    wk // 2:width + wk // 2].flatten(1)
            mask[mask >= 1] = -np.inf
            self.register_buffer('mask', mask[None, None, :, :])
        self.mixer = mixer

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        if self.input_shape is not None:
            input_size, embed_dims = self.input_size, self.embed_dims
        else:
            _, input_size, embed_dims = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, input_size, 3, self.num_heads, embed_dims // self.num_heads))
        qkv = qkv.permute((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mixer == 'Local':
            attn += self.mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn.matmul(v).permute(0, 2, 1, 3).reshape(-1, input_size, embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x