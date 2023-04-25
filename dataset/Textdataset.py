import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import lmdb
import random
import numpy as np
import pandas as pd

class SpellingMutation(object):
    def __init__(self, pn0=0.5, pn1=0.85, pn2=0.95, pt0=0.7, pt1=0.85, charset=None):
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1 = pt0, pt1
        self.charset = charset

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in '0123456789' for t in text])
        if digit_num / length < ratio: 
          return False
        return True

    def is_unk_char(self, char):
        # return char == self.charset.unk_char
        return char == '<unk>'
        # return (char not in '0123456789') and (char == '<unk>')

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0:
            num_to_modify = 0
        elif prob < self.pn1:
            num_to_modify = 1
        elif prob < self.pn2:
            num_to_modify = 2
        else:
            num_to_modify = 3

        if length <= 1:
            num_to_modify = 0
        elif length >= 2 and length <= 4:
            num_to_modify = min(num_to_modify, 1)
        else:
            num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify



    def __call__(self, text, debug=False):
        if self.is_digit(text): 
          return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: 
          return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: 
          self.index = index
        for i, t in enumerate(text):
            if i not in index:
                chars.append(t)

            elif self.is_unk_char(t):
                chars.append(t)

            else:
                prob = random.random()
                if prob < self.pt0:  # replace
                    chars.append(random.choice(self.charset.chars))

                elif prob < self.pt1:  # insert
                    chars.append(random.choice(self.charset.chars))
                    chars.append(t)

                else:  # delete
                    continue
        new_text = ''.join(chars)
        return new_text if len(new_text) >= 1 else text

class TextDataset(Dataset):
  def __init__(self, root, charset=None, max_length=50, limit=1):
    super(TextDataset, self).__init__()
    self.charset = charset
    self.max_len = max_length
    self.eos = charset.get_eos_index()
    self.sm = SpellingMutation(charset=self.charset)

    self.env = lmdb.open(root, 
                max_readers=32, 
                readonly=True, 
                lock=False, 
                readahead=False, meminit=False)
    if not self.env:
      print('cannot open lmdb from %s' % (root))
      return

    with self.env.begin(write=False) as txn:
      nSamples = int(txn.get('num-samples'.encode()))
      self.nSamples = nSamples
      index_list = [index + 1 for index in range(self.nSamples)
                if (index % 10) > 4]
        
      data_limit = int(len(index_list) * limit)
      self.filtered_index_list = index_list[:data_limit]

  def __len__(self):
    return len(self.filtered_index_list)

  def _readitem(self, label_key):
    with self.env.begin(write=False) as txn:
      if label_key:
        label = txn.get(label_key).decode('utf-8')
      else:
        label = '<unk>'
    return label

  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    index = self.filtered_index_list[index]
    label_key = 'label-%09d'.encode() % index
    label_y = self._readitem(label_key=label_key)
    label_x = self.sm(label_y)

    label_x = self.charset.lookup_indices(list(label_x))
    label_x = label_x + [self.eos]
    label_x = torch.LongTensor(label_x)
    label_x = F.pad(label_x, (0, self.max_len - len(label_x)), value=self.eos)
    label_x = onehot(label_x, len(self.charset))

    label_y = self.charset.lookup_indices(list(label_y))
    label_y = label_y + [self.eos]
    label_y = torch.LongTensor(label_y)
    label_y = F.pad(label_y, (0, self.max_len - len(label_y)), value=self.eos)
    return label_x, label_y
  
def onehot(label, depth, device=None):
  if not isinstance(label, torch.Tensor):
    label = torch.tensor(label, device=device)
  onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
  onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)
  return onehot