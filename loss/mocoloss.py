#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class MocoLoss(nn.Module):
  def __init__(self, temp=0.1, reduction="batchmean"):
    super(MocoLoss, self).__init__()
    self.reduction = reduction
    self.temp = temp

  def forward(self, q, k, queue, *args, **kwargs):
    N = q.shape[0]
    C = q.shape[1]
    
    pos_similarity_matrix = torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1)
    pos_similarity_matrix_exp = torch.exp(torch.div(pos_similarity_matrix, self.temp))

    neg_similarity_matrix= torch.mm(q.view(N,C), torch.t(queue))
    neg_similarity_matrix_exp = torch.exp(torch.div(neg_similarity_matrix, self.temp))
    neg = torch.sum(neg_similarity_matrix_exp, dim=1)

    denominator = neg + pos_similarity_matrix_exp
    loss = -torch.log(torch.div(pos_similarity_matrix_exp, denominator))
    return torch.mean(loss)
 