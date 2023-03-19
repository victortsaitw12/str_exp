#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn

class ShallowCNN(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ShallowCNN, self).__init__()
    self.feature_extractor = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )

  def forward(self, inputs):
    return self.feature_extractor(inputs)
