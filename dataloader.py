#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
from dataset.lmdbdataset import LmdbDataset
from dataset.rawdataset import RawDataset
from dataset.augimagedataset import AugImageDataset

def collate_fn(batch, max_len):
  tgt = []
  src = []
  for _src, _tgt in batch:
    _tgt = [0] + _tgt + [1]
    _tgt = torch.LongTensor(_tgt)
    _tgt = F.pad(_tgt, (0, max_len - len(_tgt)), value=2)
    tgt.append(_tgt)
    src.append(_src)
  tgt = torch.stack(tgt)
  src = torch.stack(src)
  tgt_in = tgt[:, :-1]
  tgt_out = tgt[:, 1:]
  n_tokens = (tgt_out != 2).sum()
  return src, tgt_in, tgt_out, n_tokens

class MyDataLoader():
  def __init__(self, batch_size, img_w, img_h, max_len,
               raw_root='', train_path='', test_path='', seqclr_data_count='', lmdb_root=''):
    super().__init__()
    self.batch_size = batch_size
    self.raw_root = raw_root
    self.train_path = train_path
    self.seqclr_data_count = seqclr_data_count
    self.lmdb_root = lmdb_root
    self.test_path = test_path
    self.img_w = img_w
    self.img_h = img_h
    self.max_len = max_len

  def __call__(self, aug_img=False):
    if aug_img:
      return self.load_aug_image_loader()
    return self.load_train_loader()
  
  def load_aug_image_loader(self):
    ds = AugImageDataset(self.raw_root, self.train_path, count=self.seqclr_data_count)
    return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

  def load_train_loader(self):
    train_ds = self.load_train_ds()
    valid_ds_size = int(len(train_ds) * 0.05)
    train_ds_size = len(train_ds) - valid_ds_size
    train_ds, valid_ds = random_split(train_ds, [train_ds_size, valid_ds_size])

    train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                              collate_fn=lambda x: collate_fn(x, self.max_len))
    valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False, 
                              collate_fn=lambda x: collate_fn(x, self.max_len))
    return train_loader, valid_loader

  def load_train_ds(self):
    dataset_list = []
    # Get LMDB Dataset
    if self.lmdb_root:
      for dirpath, dirnames, filenames in os.walk(self.lmdb_root):
        if dirnames:
          continue
        if not filenames:
          continue
        if os.path.isdir(dirpath):
          dataset = LmdbDataset(dirpath, img_w=self.img_w, img_h=self.img_h, rgb=True)
          dataset_list.append(dataset)
    else:
      print('no LMDB')
    # Get Raw Dataset
    if self.raw_root and self.train_path:
      raw_train_ds = RawDataset(self.raw_root, self.train_path)
      dataset_list.append(raw_train_ds)

    if self.raw_root and self.test_path:
      raw_test_ds = RawDataset(self.raw_root, self.test_path)
      dataset_list.append(raw_test_ds)

    # Concat all datasets
    print(f'dataset_list count: {len(dataset_list)}')
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset