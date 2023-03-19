
import torch
import torch.nn as nn
from .mlp import MLP
from .convmixer import ConvMixer
from .attnmixer import AttnMixer

def drop_path(x, drop_prob=0., training=False):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of
  residual blocks).
  We follow the implementation
  https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
  """
  if drop_prob == 0. or not training:
    return x
  keep_prob = 1 - drop_prob
  # handle tensors with different dimensions, not just 4D tensors.
  shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
  random_tensor = keep_prob + torch.rand(
      shape, dtype=x.dtype, device=x.device)
  output = x.div(keep_prob) * random_tensor.floor()
  return output


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample  (when applied in main path of
  residual blocks).
  We follow the implementation
  https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
  Args:
      drop_prob (float): Probability of the path to be zeroed. Default: 0.1
  """

  def __init__(self, drop_prob=0.1):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)

class MixingBlock(nn.Module):
    """The Mixing block.
    Args:
        embed_dims (int): Number of character components.
        num_heads (int): Number of heads
        mixer (str, optional): The mixer type. Defaults to 'Global'.
        window_size (Tuple[int ,int], optional): Local window size.
            Defaults to [7, 11].
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        mlp_ratio (float, optional): The ratio of hidden features to input.
            Defaults to 4.0.
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        drop (float, optional): cfg of Dropout. Defaults to 0..
        attn_drop (float, optional): cfg of Dropout. Defaults to 0.0.
        drop_path (float, optional): The probability of drop path.
            Defaults to 0.0.
        pernorm (bool, optional): Whether to place the MxingBlock before norm.
            Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
            embed_dims,
            num_heads,
            mixer='Global',
            window_size=[7, 11],
            input_shape=[8, 25],
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            prenorm=True,
            init_cfg=None):
        super(MixingBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        if mixer in {'Global', 'Local'}:
            self.mixer = AttnMixer(
                embed_dims,
                num_heads=num_heads,
                mixer=mixer,
                input_shape=input_shape,
                local_k=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                embed_dims,
                num_heads=num_heads,
                input_shape=input_shape,
                local_k=window_size)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(in_features=embed_dims, hidden_features=mlp_hidden_dim, drop=drop)
        self.prenorm = prenorm

    def forward(self, x) :
        """Forward function.
        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H*W, C)`.
        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H*W, C)`.
        """
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
