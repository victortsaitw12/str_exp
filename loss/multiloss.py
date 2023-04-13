#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class MultiLoss(nn.Module):
    def __init__(self, ignore_index):
        super(MultiLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def _ce_loss(self, out, tgt, weight=1.0):
        iter_size = out.shape[0] // tgt.shape[0]
        if iter_size > 1:
            tgt = tgt.repeat(iter_size, 1, 1)
        flat_out = out.contiguous().view(-1, out.size(-1))
        flat_gt = tgt.contiguous().view(-1)
        return self.ce(flat_out, flat_gt) * weight
    
    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res        
        return torch.cat(all_res, dim=0)
    
    def forward(self, outs, tgt):
        if isinstance(outs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outs]
            loss = sum([self._ce_loss(o, tgt) for o in outputs])
        else:
            loss = self._ce_loss(outs, tgt)
        return loss