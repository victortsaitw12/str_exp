#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from .positionalencoder1d import PositionalEncoder1D

class TransformerDecoder(nn.Module):
  def __init__(self, input_size, num_classes, layers=2, dropout=0.1, pad=2):
    super(TransformerDecoder, self).__init__()
    self.num_classes = num_classes
    self.pad = pad
    # Positional Encoder
    self.text_embedding = nn.Embedding(num_classes, input_size)
    self.positional_encoder_1D = PositionalEncoder1D(d_model=input_size, dropout=dropout)

    # Transformer Decoder
    decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=8, batch_first=True)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
    self.generator = nn.Linear(input_size, num_classes)

  def _forward_pass(self, memory, text, device):
    # mask
    text_mask = nn.Transformer.generate_square_subsequent_mask(text.size(1)).to(device)
    text_key_padding_mask = TransformerDecoder.get_key_padding_mask(text, self.pad)

    # tgt embedding
    text_embedding = self.text_embedding(text)
    text_embedding = self.positional_encoder_1D(text_embedding)

    # transformer function
    out = self.decoder(tgt=text_embedding, memory=memory, 
        tgt_mask=text_mask, tgt_key_padding_mask=text_key_padding_mask)
    return out
      
  # def forward(self, text, memory, device):
  def forward(self, memory, text, is_train=True, batch_max_length=25):
    device = text.device

    if is_train:
      out = self._forward_pass(memory, text, device)
      probs = self.generator(out)
    else:
      batch_size = memory.size(0)
      num_steps = batch_max_length - 1 #-1 for bos
      targets = torch.LongTensor(batch_size, num_steps).fill_(0).to(device)  # [GO] token
      probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

      for i in range(num_steps):
          text = targets[:, i]
          out = self._forward_pass(memory, text, device)
          probs_step = self.generator(out[:, -1])
          probs[:, i, :] = probs_step
          _, next_input = probs_step.max(1)
          targets[:, i] = next_input
    return probs  # batch_size x num_steps x num_classes

  @staticmethod
  def get_key_padding_mask(data, pad):
    return data == pad
