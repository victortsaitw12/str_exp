#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import math

class PositionalEncoder2D(nn.Module):
  def __init__(self, d_model, dropout, device, max_len=5000):
    super(PositionalEncoder2D, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.d_model = d_model

    pe_h = torch.zeros(max_len, d_model).to(device)
    pe_w = torch.zeros(max_len, d_model).to(device)
    div_term = torch.exp(
        torch.arange(0, 2, d_model) * -math.log(10000) / d_model
    )
    position_h = torch.arange(0, max_len).unsqueeze(1)
    position_w = torch.arange(0, max_len).unsqueeze(1)
    pe_h[:, 0::2] = torch.sin(position_h * div_term)
    pe_h[:, 1::2] = torch.cos(position_h * div_term)
    pe_w[:, 0::2] = torch.sin(position_w * div_term)
    pe_w[:, 1::2] = torch.cos(position_w * div_term)
    self.register_buffer('pe_h', pe_h)
    self.register_buffer('pe_w', pe_w)
    
    self.pool_h = nn.AdaptiveAvgPool2d((1, 1))
    self.pool_w = nn.AdaptiveAvgPool2d((1, 1))
    self.transform_h = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model),
      nn.Sigmoid()
    )
    self.transform_w = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model),
      nn.Sigmoid()
    )

  def forward(self, x):
    # tensor shape
    nbatches = x.size(0)
    pe_h = self.pe_h[:x.size(2), :].unsqueeze(0).unsqueeze(2)
    pe_w = self.pe_w[:x.size(3), :].unsqueeze(0).unsqueeze(1)
    x_h = self.pool_h(x).permute(0, 2, 3, 1)
    x_w = self.pool_w(x).permute(0, 2, 3, 1)
    alpha = self.transform_h(x_h)
    beta = self.transform_w(x_w)
    pe = alpha * pe_h + beta * pe_w
    outputs = (x.permute(0, 2, 3, 1) + pe).view(nbatches, -1, self.d_model)
    return outputs
