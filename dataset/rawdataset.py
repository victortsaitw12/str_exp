#!/usr/bin/python
# -*- coding: UTF-8 -*-

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import os

class RawDataset(Dataset):
  def __init__(self, root, file_path, charset, img_w, img_h, rgb=True):
    super(RawDataset, self).__init__()
    self.root = root
    self.img_w = img_w
    self.img_h = img_h
    self.rgb = rgb
    self.imgs, self.labels = self.read(file_path)
    self.charset = charset
    self.charset.set_default_index(3)
    self.transform = transforms.Compose([
        transforms.ToTensor()
    ])

  def read(self, file_path):
    abs_file_path = os.path.join(self.root, file_path)
    imgs = []
    labels = []
    with open(abs_file_path, 'r', encoding='utf-8') as f:
      for line in f:
        img_path, label = line.split()
        imgs.append(img_path)
        labels.append(label)
    return imgs, labels

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img = self.imgs[idx]
    label = self.labels[idx]
    img_path = os.path.join(self.root, img)
    if self.rgb:
        img = Image.open(img_path).convert('RGB')
    else:
        img = Image.open(img_path).convert('L')
    img = img.resize((self.img_w, self.img_h))
    img = self.transform(img)
    label = self.charset.lookup_indices(list(label))
    return (img, label)
