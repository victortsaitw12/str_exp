#!/usr/bin/python
# -*- coding: UTF-8 -*-

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import os
import numpy as np
from .augment.seqclraug import get_augmentations

class AugImageDataset(Dataset):
  def __init__(self, root, file_path, img_w, img_h, count=-1):
    super(AugImageDataset, self).__init__()
    self.root = root
    self.count = count
    self.img_w = img_w
    self.img_h = img_h
    self.imgs = self.read(file_path)
    self.augment = get_augmentations().augment_image
    self.transform = transforms.Compose([
        transforms.ToTensor()
    ])

  def read(self, file_path):
    abs_file_path = os.path.join(self.root, file_path)
    imgs = []
    with open(abs_file_path, 'r', encoding='utf-8') as f:
      for line in f:
        img_path, _ = line.split()
        imgs.append(img_path)

    if self.count > 0:
      imgs = imgs[:self.count]

    return imgs

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img = self.imgs[idx]
    img = Image.open(os.path.join(self.root, img)).convert('RGB')
    img = img.resize((self.img_w, self.img_h))
    img = np.array(img)
    image_views = []
    for _ in range(2):
      img = self.augment(img)
      image_views.append(self.transform(img))
    return np.stack(image_views, axis=0)