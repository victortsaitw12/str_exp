#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class BCNLanguage(nn.Module):
  def __init__(self, input_channel, num_classes, max_length, eos_index):
    super(BCNLanguage, self).__init__()
    self.max_length = max_length  # additional stop token
    d_model = 512
    nhead = 8
    d_inner = 2048
    dropout = 0.1
    activation = 'relu'
    num_layers = 4
    self.eos_index = eos_index
    self.d_model = d_model
    self.detach = True
    self.loss_weight = 1.0
    self.debug = False

    # self.vproj = nn.Linear(input_channel, num_classes)

    self.proj = nn.Linear(num_classes, d_model, False)
    self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
    self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)
    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_inner,
            dropout=dropout)
    self.model = nn.TransformerDecoder(decoder_layer, num_layers)

    self.cls = nn.Linear(d_model, num_classes)

    # if config.model_language_checkpoint is not None:
    #   self.load(config.model_language_checkpoint)

  def forward(self, tokens):
    """
    Args:
        tokens: (N, T, C) where T is length, N is batch size and C is classes number
        lengths: (N,)
    """
    # v_features = self.vproj(tokens)
    if self.detach: 
      tokens = tokens.detach()
    lengths = self._get_length(tokens).clamp_(2, self.max_length)
    # print(lengths)
    embed = self.proj(tokens)  # (N, T, E)
    embed = embed.permute(1, 0, 2)  # (T, N, E)
    embed = self.token_encoder(embed)  # (T, N, E)
    
    padding_mask = self._get_padding_mask(lengths, self.max_length)
    zeros = embed.new_zeros(*embed.shape)
    qeury = self.pos_encoder(zeros)
    location_mask = self._get_location_mask(self.max_length, tokens.device)
    output = self.model(qeury, embed,
            tgt_key_padding_mask=padding_mask,
            memory_mask=location_mask,
            memory_key_padding_mask=padding_mask)  # (T, N, E)
    output = output.permute(1, 0, 2)  # (N, T, E)
    # print(output)
    logits = self.cls(output)  # (N, T, C)
    return logits
    # pt_lengths = self._get_length(logits)

    # res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
    #         'loss_weight':self.loss_weight, 'name': 'language'}
    # return res


  def _get_length(self, logit):
    """ Greed decoder to obtain length from logit"""
    out = (logit.argmax(dim=-1) == self.eos_index)
    out = self.first_nonzero(out.int()) + 1
    return out

  @staticmethod
  def first_nonzero(x):
    non_zero_mask = x != 0
    mask_max_values, mask_max_indices = torch.max(non_zero_mask.int(), dim=-1)
    mask_max_indices[mask_max_values == 0] = -1
    return mask_max_indices

  @staticmethod
  def _get_padding_mask(length, max_length):
    length = length.unsqueeze(-1)
    grid = torch.arange(0, max_length, device=length.device).unsqueeze(0)
    return grid >= length


  @staticmethod
  def _get_location_mask(sz, device=None):
    mask = torch.eye(sz, device=device)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))
    return mask