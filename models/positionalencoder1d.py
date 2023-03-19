#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import math

class PositionalEncoder1D(nn.Module):
  def __init__(self, d_model, dropout, device, max_len=5000):
    super(PositionalEncoder1D, self).__init__()
    pe = torch.zeros(max_len, d_model).to(device)
    positions = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe[:, :x.size(1)].requires_grad_(False)
