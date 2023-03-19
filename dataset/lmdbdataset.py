#!/usr/bin/python
# -*- coding: UTF-8 -*-

import six
import lmdb
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from .augment.seqclraug import get_augmentations
from .augment.colorjitter import ColorJitter
from .augment.geometry import Geometry
from .augment.deterioration import Deterioration

class LmdbDataset(Dataset):
  def __init__(self, root, img_w, img_h, charset=None, rgb=True, pretrain=False, limit=1):
    super(LmdbDataset, self).__init__()
    self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, 
                         readahead=False, meminit=False)
    if not self.env:
      print('cannot open lmdb from %s' % (root))
      return
    
    self.root = root
    self.img_w = img_w
    self.img_h = img_h
    self.rgb = rgb
    self.pretrain = pretrain
    self.charset = charset
    if charset:
      self.charset.set_default_index(3)
    
    if pretrain:
      self.augment = get_augmentations().augment_image
    else:
      self.augment = transforms.Compose([
        Geometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
        Deterioration(var=20, degrees=6, factor=4, p=0.25),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
      ])

    self.transform = transforms.Compose([
        transforms.ToTensor()
    ])  

    with self.env.begin(write=False) as txn:
      nSamples = int(txn.get('num-samples'.encode()))
      self.nSamples = nSamples
      if pretrain:
        index_list = [index + 1 for index in range(self.nSamples)
                                    if (index % 10) > 4]
      else:
        index_list = [index + 1 for index in range(self.nSamples) if (index % 10) < 5]
        
      data_limit = int(len(index_list) * limit)
      self.filtered_index_list = index_list[:data_limit]
      # self.filtered_index_list = self.filtered_index_list[384000:]
  def __len__(self):
    return len(self.filtered_index_list)

  def _readitem(self, img_key, label_key=None):
    with self.env.begin(write=False) as txn:
      if label_key:
        label = txn.get(label_key).decode('utf-8')
      else:
        label = '[dummy_label]'
      imgbuf = txn.get(img_key)

      buf = six.BytesIO()
      buf.write(imgbuf)
      buf.seek(0)

      try:
        if self.rgb:
          img = Image.open(buf).convert('RGB')
        else:
          img = Image.open(buf).convert('L')
      except IOError:
        print(f'Corrupted image for {img_key}')
        # make dummy image and label for corrupted image.
        if self.rgb:
          img = Image.new('RGB', (self.img_w, self.img_h))
        else:
          img = Image.new('L', (self.img_w, self.img_h))
        label = '[dummy_label]'
    return img, label


  def _pretrainitem(self, index):
    assert index <= len(self), 'index range error'
    index = self.filtered_index_list[index]
    img_key = 'image-%09d'.encode() % index
    img, _ = self._readitem(img_key=img_key)
    img = img.resize((self.img_w, self.img_h))
    img = np.array(img)
    image_views = []
    for _ in range(2):
      img = self.augment(img)
      image_views.append(self.transform(img))
    return np.stack(image_views, axis=0)

  def _trainitem(self, index):
    assert index <= len(self), 'index range error'
    index = self.filtered_index_list[index]
    label_key = 'label-%09d'.encode() % index
    img_key = 'image-%09d'.encode() % index
    img, label = self._readitem(img_key=img_key, label_key=label_key)
    img = self.augment(img)
    img = img.resize((self.img_w, self.img_h))
    img = self.transform(img)
    label = self.charset.lookup_indices(list(label))
    return (img, label)
  
  def __getitem__(self, index):
    if self.pretrain:
      return self._pretrainitem(index=index)
    return self._trainitem(index=index)
  

if __name__ == '__main__':
  print('test')