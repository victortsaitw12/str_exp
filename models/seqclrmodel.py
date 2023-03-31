#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn
from .model import Encoder
from .seqclrproj import SeqCLRProj

class SeqCLRModel(nn.Module):
  def __init__(self, opt):
    super(SeqCLRModel, self).__init__()
    self.vision = Encoder(opt)
    self.instance_mapping_func = nn.AdaptiveAvgPool2d((5, opt.projection_input_channel))
    # self.seqclr_proj = SeqCLRProj(projection_input_size=opt.projection_input_channel,
    #               projection_hidden_size=opt.projection_hidden_size,
    #               projection_output_size=opt.projection_output_channel)

  def forward(self, images, *args, **kwargs):
    v_res_view0 = self.vision(images[:, 0], *args, **kwargs)
    v_res_view1 = self.vision(images[:, 1], *args, **kwargs)
    # projected_features_view0 = self.seqclr_proj(v_res_view0)
    # projected_features_view1 = self.seqclr_proj(v_res_view1)
    projected_features_view0 = self.instance_mapping_func(v_res_view0)
    projected_features_view1 = self.instance_mapping_func(v_res_view1)
    return projected_features_view0, projected_features_view1