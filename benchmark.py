#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from nltk.metrics.distance import edit_distance
from models.model import Model
from dataset.rawdataset import RawDataset

def benchmark(opt, log):
    log('benchmark')
    test_dataset = RawDataset(root=opt.raw_root, file_path=opt.test_path,
                            img_w=opt.img_w, img_h=opt.img_h, 
                            charset=opt.charset, rgb=opt.rgb)
    
    # Load saved model
    model=Model(opt).to(opt.device)
    f_path = os.path.join(opt.save_path, 'checkpoint.pt')
    log(f'loading pretrained model from {f_path}')
    if not os.path.exists(f_path):
        raise Exception(f'Checkpoints not found at {f_path}')
    checkpoint = torch.load(f_path, map_location=opt.device)
    model.load_state_dict(checkpoint['state_dict'])

    # benchmark
    length_of_data = len(test_dataset)
    model.eval()
    with torch.no_grad():
        labels = []
        preds_str = []
        n_correct = 0
        norm_ED = 0
        for i, (img, label) in enumerate(test_dataset):
            img = img.to(opt.device)
            img = img.unsqueeze(0)
            label = ''.join(opt.charset.lookup_tokens(label))
            labels.append(label)
            if opt.decoder == 'CTC':
                tgt = torch.LongTensor(opt.batch_size, opt.max_len)
                out = model(img, tgt)
                _, preds_index = out.max(2)
                t = preds_index.squeeze().tolist()
                char_list = [t[i] for i in range(len(t)) 
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i]))
                ]  # removing repeated characters and blank.
                pred = ''.join(opt.charset.lookup_tokens(char_list))
            elif opt.decoder == 'LM':
                tgt = torch.LongTensor(opt.batch_size, opt.max_len)
                out = model(img, tgt)
                _, preds_index = torch.max(out[0][-1], dim=2)

                for index in range(opt.batch_size):
                    pred_str = preds_index[index, :].tolist()
                    eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
                    if eos_res[0].any():
                        eos_index = eos_res[0][0]
                        pred_str = pred_str[:eos_index]
                    pred = ''.join(opt.charset.lookup_tokens(pred_str))
            else:
                tgt = torch.LongTensor(opt.batch_size, opt.max_len)
                tgt.fill_(opt.charset.get_bos_index())
                tgt = tgt.to(opt.device)
                out = model(img, tgt)
                _, preds_index = torch.max(out, dim=2)

                for i in range(opt.batch_size):        
                    pred_str = preds_index[i].tolist()
                    eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
                    if eos_res[0].any():
                        eos_index = eos_res[0][0]
                        pred_str = pred_str[:eos_index]
                    pred = ''.join(opt.charset.lookup_tokens(pred_str))

            # log(f'{i}/{length_of_data}, {label}, {pred}')
            preds_str.append(pred)

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

    log(f'acc:{accuracy}, norm_ED:{norm_ED}')
