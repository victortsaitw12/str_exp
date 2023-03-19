#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import os
import logging
from pathlib import Path
# import numpy as np
# import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def img_show(images):
  if not isinstance(images, list):
    images = [images]
  _, axis = plt.subplots(ncols=len(images), squeeze=False)
  for i, image in enumerate(images):
    image  = image.detach()
    image = TF.to_pil_image(image)
    axis[0, i].imshow(np.asarray(image))
    axis[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
  plt.show()

def ctc_collate_fn(batch, max_len, pad):
  tgt = []
  src = []
  length = []
  for _src, _tgt in batch:
    length.append(len(_tgt))
    _tgt = torch.LongTensor(_tgt)
    _tgt = F.pad(_tgt, (0, max_len - len(_tgt)), value=pad)
    tgt.append(_tgt)
    src.append(_src)
  tgt = torch.stack(tgt)
  src = torch.stack(src)
  n_tokens =  torch.IntTensor(length)
  return src, tgt,  n_tokens

def attn_collate_fn(batch, max_len, bos, eos, pad):
  tgt = []
  src = []
  for _src, _tgt in batch:
    _tgt = [bos] + _tgt + [eos]
    _tgt = torch.LongTensor(_tgt)
    _tgt = F.pad(_tgt, (0, max_len - len(_tgt)), value=pad)
    tgt.append(_tgt)
    src.append(_src)
  tgt = torch.stack(tgt)
  src = torch.stack(src)
  tgt_in = tgt[:, :-1]
  tgt_out = tgt[:, 1:]
  n_tokens = (tgt_out != pad).sum()
  return src, tgt_in, tgt_out, n_tokens


class MyLogger(object):
  def __init__(self, log_path, mode='debug'):
    super().__init__()
    self.log_path = log_path
    self._switch_print = True
    self.mode = mode
    self.remove_log_file()
    logger = logging.getLogger('exp')
    logging.basicConfig(
        filename=log_path, # write to this file
        filemode='a', # open in append mode
        format='[%(asctime)s,%(msecs)d %(levelname)s] %(message)s',
        force=True,
        level=logging.DEBUG
    )
    self.writer = SummaryWriter()
    
  def __call__(self, msg):
    if self._switch_print:
      print(msg)
    logging.info(msg)

  def remove_log_file(self):
    if os.path.isfile(self.log_path):
      os.remove(self.log_path)
  
  def debug(self, *args):
    msg = ''.join([str(t) for t in args])
    if self.mode == 'debug':
      logging.debug(msg)

  def print_all(self):
    print(Path(self.log_path).read_text())

  def switch_print(self, switch=False):
    self._switch_print = switch

  def set_mode(self, mode):
    self.mode = mode

  def add_scalar(self, tag, *args):
    self.writer.add_scalar(tag, *args)
    self.writer.flush()

  def flush(self):
    return self.writer.flush()
  
  def close(self):
    return self.writer.close()
# log = MyLogger(LOG_PATH)

def save_ckp(model_state, optimizer_state, epoch, step, ckp_path,
             scheduler_step_count=0, scheduler_last_epoch=0
             ):
  if not os.path.exists(ckp_path):
    os.mkdir(ckp_path)
  f_path = os.path.join(ckp_path, 'checkpoint.pt')
  torch.save({
      'epoch': epoch,
      'step': step,
      'state_dict': model_state,
      'optimizer': optimizer_state,
      'scheduler_last_epoch': scheduler_last_epoch,
      'scheduler_step_count': scheduler_step_count
  }, f_path)

def check_checkpoints(root):
    f_path = os.path.join(root, 'checkpoint.pt')
    return os.path.exists(f_path)

def load_encoder_ckp(model, ckp_path, device):
    f_path = os.path.join(ckp_path, 'checkpoint.pt')

    if not os.path.exists(f_path):
        raise ValueError(f'{f_path} not found')
    
    checkpoint = torch.load(f_path, map_location=device)
    states = checkpoint['state_dict']
    states_dict = OrderedDict(
    {k.split('.', 1)[1]: v for k, v in states.items()
        if k.split('.', 1)[0] == 'vision'}
    )
    
    model.load_encoder_state(states_dict)
    return model

def load_ckp(model, optimizer, ckp_path, device, scheduler=None):
  f_path = os.path.join(ckp_path, 'checkpoint.pt')
  if not os.path.exists(f_path):
    print(f'{f_path} is not found')
    return model, optimizer, 0, 0, scheduler
  
  checkpoint = torch.load(f_path, map_location=device)
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  epoch = checkpoint['epoch']
  step = checkpoint['step']
  if scheduler:
    scheduler._step_count = checkpoint['scheduler_step_count']
    scheduler.last_epoch = checkpoint['scheduler_last_epoch']

  return model, optimizer, epoch, step, scheduler


class Charset(object):
  def __init__(self, char_path, specials=[]):
    super().__init__()
    with open(char_path, 'r', encoding='utf-8') as f:
      line = f.readline()
    self.chars = specials + list(line)
    self.dict = dict([(i, self.chars[i]) for i in range(len(self.chars))])
    self.default_index = -1
    self.pad_index =-1
    self.eos_index = -1
    self.bos_index = -1

  def lookup_indices(self, chars):
    idx_list = []
    for c in chars:
      try:
        idx = self.chars.index(c)
        idx_list.append(idx)
      except ValueError:
        idx_list.append(self.default_index)
    return idx_list

  def lookup_token(self, idx):
    return self.dict[idx]

  def lookup_tokens(self, indices):
    return [self.dict[idx] for idx in indices]

  def set_default_index(self, idx):
    self.default_index = idx

  def set_pad_index(self, idx):
    self.pad_index = idx

  def get_pad_index(self):
    return self.pad_index
  
  def set_bos_index(self, index):
    self.bos_index = index

  def get_bos_index(self):
    return self.bos_index
  
  def set_eos_index(self, index):
    self.eos_index = index
  
  def get_eos_index(self):
    return self.eos_index
  
  def __len__(self):
    return len(self.chars)
