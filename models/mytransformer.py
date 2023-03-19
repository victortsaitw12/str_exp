#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from .transformerdecoder import TransformerDecoder as MyDecoder
from .transformerencoder import TransformerEncoder as MyEncoder
import os
from collections import OrderedDict

class MyTransformer(nn.Module):
  def __init__(self, tgt_num_class, d_model, dropout=0.1):
    super(MyTransformer, self).__init__()
    # Transformer Encoder
    self.encoder = MyEncoder(d_model=d_model, dropout=dropout)
    
    # Transformer Decoder
    self.decoder = MyDecoder(num_class=tgt_num_class, d_model=d_model, dropout=dropout)

    self.projector = nn.Linear(d_model, tgt_num_class)

  def load_pretrain_encoder(self, path):
    states = torch.load(os.path.join(path, 'checkpoint.pt'))
    states = states['state_dict']
    states_dict = OrderedDict(
      {k.split('.', 1)[1]: v for k, v in states.items()
        if k.split('.', 1)[0] == 'vision'}
    )
    self.encoder.load_state_dict(states_dict)
    
    # Freeze
    for param in self.encoder.parameters():
      param.requires_grad = False

  def forward(self, src, tgt, pretrain_encoder=False):
    memory = self.encoder(src)
    out = self.decoder(tgt, memory)
    return out