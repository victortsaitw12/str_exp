
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
from nltk.metrics.distance import edit_distance

def validation(loader, model, criterion, optimizer, opt):
  # Validation
  model.eval()
  n_correct = 0
  tot_loss = 0
  tot_loss_count = 0
  norm_ED = 0
  length_of_data = 0
  with torch.no_grad():
    for i, batch in enumerate(loader):
      length_of_data = length_of_data + opt.batch_size

      if opt.decoder == 'CTC':
        src, tgt, n_tokens = batch
        src, tgt = src.to(opt.device), tgt.to(opt.device)
        out = model(src, tgt)
        out_size = torch.IntTensor([out.size(1)] * opt.batch_size)
        loss = criterion(out.log_softmax(2).permute(1, 0, 2), tgt, out_size, n_tokens)
        _, preds_index = out.max(2)
        preds_str = []
        for index, l in enumerate(out_size.data):
          t = preds_index[index,:].tolist()
          char_list = [t[i] for i in range(l) 
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i]))
          ]  # removing repeated characters and blank.
          preds_str.append(''.join(opt.charset.lookup_tokens(char_list)))

        labels = []
        for index, l in enumerate(n_tokens):
          t = tgt[index, :].tolist()
          char_list = [t[i] for i in range(l)]
          labels.append(''.join(opt.charset.lookup_tokens(char_list)))

      else: # Attn || Transformer
        tgt = torch.zeros(opt.batch_size, opt.max_len).to(opt.device)
        src, _, tgt_y, _ = batch
        src, tgt_y = src.to(opt.device), tgt_y.to(opt.device)
        
        out = model(src, tgt, is_train=False)
        loss = criterion(
          out.contiguous().view(-1, out.size(-1)),
          tgt_y.contiguous().view(-1)
        )
        _, preds_index = torch.max(out, dim=2)
        # Load Predictions
        # preds_str = []
        # length = torch.IntTensor([opt.max_length] * opt.batch_size).to(opt.device)
        # for index, l in enumerate(length):
        #     text = ''.join(opt.charset.lookup_tokens(preds_index[index, :])
        #     preds_str.append(text)
        # Load Labels
        labels = []
        for i in range(opt.batch_size):
            label = tgt_y[i].tolist()
            label = label[:label.index(opt.charset.get_eos_index())]
            labels.append(''.join(opt.charset.lookup_tokens(label)))
            
            pred_str = preds_index[i].tolist()
            eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
            if eos_res[0].any():
              eos_index = eos_res[0][0]
              pred_str = pred_str[:eos_index]
            preds_str = ''.join(opt.charset.lookup_tokens(pred_str))

      tot_loss += loss.data.sum()
      tot_loss_count += loss.data.numel()

      for gt, pred in zip(labels, preds_str):
        if gt == pred:
          n_correct += 1

        if len(gt) == 0 or len(pred) == 0:
          norm_ED += 0
        elif len(gt) > len(pred):
          norm_ED += 1 - edit_distance(pred, gt) / len(gt)
        else:
          norm_ED += 1 - edit_distance(pred, gt) / len(pred)

  accuracy = n_correct / float(length_of_data) * 100
  norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

  return tot_loss / float(tot_loss_count), accuracy, norm_ED
