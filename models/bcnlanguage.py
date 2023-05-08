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
    d_model = 384
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
    return logits, output
    # pt_lengths = self._get_length(logits)

    # res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
    #         'loss_weight':self.loss_weight, 'name': 'language'}
    # return res

  # def forward(self, tokens, num_iter=3):
  #   tokens_list = []
  #   for i in range(num_iter):
  #     tokens = torch.softmax(tokens, dim=-1)
  #     tokens = self.forward_tokens(tokens)
  #     tokens_list.append(tokens)
  #   return tokens_list
  
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
  



class BCNAlignment(nn.Module):
  def __init__(self, input_channel, num_classes, max_length, eos_index, opt):
    super(BCNAlignment, self).__init__()
    self.max_length = max_length  # additional stop token
    self.eos_index = eos_index
    self.language = BCNLanguage(input_channel, num_classes, max_length, eos_index)
    self.w_att = nn.Linear(2 * input_channel, input_channel)
    self.cls = nn.Linear(input_channel, num_classes)

    if opt.language_module_checkpoint != 'None':
      print('load lm at ', opt.language_module_checkpoint)
      checkpoint = torch.load(opt.language_module_checkpoint, map_location=opt.device)
      self.language.load_state_dict(checkpoint['state_dict'])

  def forward_iter(self,  l_feature, v_feature):
      """
      Args:
          l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
          v_feature: (N, T, E) shape the same as l_feature 
      """
      f = torch.cat((l_feature, v_feature), dim=2)
      f_att = torch.sigmoid(self.w_att(f))
      output = f_att * v_feature + (1 - f_att) * l_feature

      logits = self.cls(output)  # (N, T, C)
      return logits, output

  def forward(self, args):
      v_features, v_tokens = args
      all_l_res, all_a_res = [], []
      for _ in range(3):
          tokens = torch.softmax(v_tokens, dim=-1)
          l_tokens, l_features = self.language(tokens)
          all_l_res.append(l_tokens)
          a_tokens, a_features = self.forward_iter(l_features, v_features)
          all_a_res.append(a_tokens)
      return all_a_res, all_l_res, v_tokens
  

class BCNEncoder(nn.Module):
    def __init__(self, input_channel, num_classes,):
      super(BCNEncoder, self).__init__()
      self.cls = nn.Linear(input_channel, num_classes)

    def forward(self, x):
      tokens = self.cls(x)
      return x, tokens
