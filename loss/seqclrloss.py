#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqCLRLoss(nn.Module):
  def __init__(self, temp=0.1, reduction="batchmean"):
    super(SeqCLRLoss, self).__init__()
    self.reduction = reduction
    self.temp = temp

  def _seqclr_loss(self, features0, features1, n_instances_per_view, n_instances_per_image):
    instances = torch.cat((features0, features1), dim=0)
    normalized_instances = F.normalize(instances, dim=1)
    similarity_matrix = normalized_instances @ normalized_instances.T
    similarity_matrix_exp = (similarity_matrix / self.temp).exp_()
    cross_entropy_denominator = similarity_matrix_exp.sum(dim=1) - similarity_matrix_exp.diag()
    cross_entropy_nominator = torch.cat((
        similarity_matrix_exp.diagonal(offset=n_instances_per_view)[:n_instances_per_view],
        similarity_matrix_exp.diagonal(offset=-n_instances_per_view)
    ), dim=0)
    cross_entropy_similarity = cross_entropy_nominator / cross_entropy_denominator
    loss = - cross_entropy_similarity.log()

    if self.reduction == "batchmean":
        loss = loss.mean()
    elif self.reduction == "sum":
        loss = loss.sum()
    elif self.reduction == "mean_instances_per_image":
        loss = loss.sum() / n_instances_per_image
    return loss

  def forward(self, instances_view0, instances_view1, *args, **kwargs):
    features0 = torch.flatten(instances_view0, start_dim=0, end_dim=1)
    features1 = torch.flatten(instances_view1, start_dim=0, end_dim=1)
    n_instances_per_image = instances_view0.shape[1]
    n_instances_per_view = instances_view0.shape[0] * n_instances_per_image
    seqclr_loss = self._seqclr_loss(features0, features1, n_instances_per_view, n_instances_per_image)
    return seqclr_loss
