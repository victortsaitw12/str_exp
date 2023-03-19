#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn
from .positionalencoder2d import PositionalEncoder2D
from .positionalencoder1d import PositionalEncoder1D
from .shallowcnn import ShallowCNN

class TransformerEncoder(nn.Module):
  def __init__(self, input_size, dropout, device, layers=2):
    super(TransformerEncoder, self).__init__()
    # # Embedding
    # self.backbone = ShallowCNN(3, d_model)
    
    # Positional Embedding
    self.positional_encoder_2D = PositionalEncoder1D(d_model=input_size, dropout=dropout, device=device)

    # Transformer Encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=8, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)


  def forward(self, input):
    # [b, c, h, w] -> [b, w*h, c]
    n, c, h, w = input.shape
    input = input.view(n, c, -1).permute(2, 0, 1)
    # src embedding
    # embedding = self.backbone(image)
    embedding = self.positional_encoder_2D(input)
    # src = src.permute(0, 2, 3, 1).reshape(src.size(0), -1, src.size(1))
    memory = self.encoder(src=embedding)
    memory = memory.permute(1, 2, 0).view(n, c, h, w)
    return memory
