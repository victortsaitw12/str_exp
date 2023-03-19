#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn
from .bilstm import BidirectionalLSTM

class SeqCLRProj(nn.Module):
  def __init__(self, projection_input_size, projection_hidden_size, projection_output_size):
    super(SeqCLRProj, self).__init__()
    self.projection = BidirectionalLSTM(projection_input_size,
                      projection_hidden_size,
                      projection_output_size)
    w = 5
    self.instance_mapping_func = nn.AdaptiveAvgPool2d((w, projection_output_size))

  def forward(self, features):
    # if self.working_layer == 'backbone_feature':
    #     features = features.permute(0, 2, 3, 1).flatten(1, 2)  # (N, E, H, W) -> (N, H*W, E)
    # print('features:', features.shape)
    features = features.permute(0, 2, 3, 1).flatten(1, 2)  # (N, E, H, W) -> (N, H*W, E)
    projected_features = self.projection(features)
    projected_instances = self.instance_mapping_func(projected_features)
    return projected_instances

  # def forward(self, output, *args):
  #   if isinstance(output, (tuple, list)):
  #     return [self._single_forward(o) for o in output]
  #   else:
  #     return [self._single_forward(output)]
