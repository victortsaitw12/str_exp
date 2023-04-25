#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.Textdataset import TextDataset
from models.bcnlanguage import BCNLanguage
from loss.multiloss import MultiLoss
from utility import  save_ckp, load_ckp
from nltk.metrics.distance import edit_distance
import numpy as np
from torch.utils.data import random_split

def pretrainBCN(opt, log):
    pretrain_dataset = TextDataset(root=opt.lmdb_root, 
                                    charset=opt.charset,
                                    max_length=opt.max_len,
                                    limit=opt.data_limit)
    pretrain_len = len(pretrain_dataset)
    valid_len = int(pretrain_len * 0.1)
    pretrain_len = pretrain_len - valid_len
    pretrain_ds, valid_ds = random_split(pretrain_dataset, [pretrain_len, valid_len])
    pretrain_loader = DataLoader(pretrain_ds, batch_size=opt.batch_size,
                            shuffle=True,
                            pin_memory=True, 
                            drop_last=True,
                            num_workers=opt.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=opt.batch_size,
                            shuffle=False,
                            pin_memory=True, 
                            drop_last=True,
                            num_workers=opt.num_workers)
    log('[Pretrain LM]pretrain_loader:{}, dataset:{}'.format(
        len(pretrain_loader), len(pretrain_loader.dataset)
    ))
    model=BCNLanguage( input_channel=opt.input_channel,
              num_classes=opt.num_class,
              max_length=opt.max_len,
              eos_index=opt.charset.get_eos_index(),).to(opt.device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    criterion = MultiLoss(ignore_index=opt.charset.get_pad_index(),
                          one_hot=False)
    model, optimizer, start_epoch, step, _ = load_ckp(model, optimizer, 
                                            opt.language_module_checkpoint, opt.device)
    log('[Pretrain LM]start epoch:{}/{}'.format(start_epoch, opt.epochs))
    for epoch in range(start_epoch, opt.epochs):
        tot_loss = 0
        tot_loss_count = 0
        for loss in pretrainByBatch(pretrain_loader, model, criterion, optimizer, opt):
            tot_loss += loss.data.sum()
            tot_loss_count += loss.data.numel()

            if step != 0 and step % opt.save_step == 0:
                valid_loss, accuracy, norm_ED = \
                    validationByBatch(valid_loader, model, criterion, opt)
                log(f'[Pretrain LM Valid]epoch:{epoch}, step:{step}, Loss:{valid_loss:0.5f}')
                # Write to the tensorboard
                log.add_scalar('Pretrain LM Valid/Loss', valid_loss, step)
                log.add_scalar('Pretrain LM Valid/Accuracy', accuracy, step)
                log.add_scalar('Pretrain LM Valid/Norm_ED', norm_ED, step)


                # save checkpoints
                save_ckp(model.state_dict(), optimizer.state_dict(),
                        epoch, step, opt.language_module_checkpoint)
                log(f'[Pretrain LM]epoch:{epoch}, step:{step}, Loss:{tot_loss/float(tot_loss_count):0.5f}')
                # Write to the tensorboard
                log.add_scalar('Pretrain/Loss', tot_loss/float(tot_loss_count), step)
                tot_loss = 0
                tot_loss_count = 0

            step += 1
        
        log.add_scalar('Pretrain/LM', scheduler.get_last_lr()[0], epoch)

    log.close()
    save_ckp(model.state_dict(), optimizer.state_dict(),
            epoch, step, opt.language_module_checkpoint)        
    log('[Pretrain LM]Done Training!!!!!!!!!!!!!!!')

def pretrainByBatch(loader, model, criterion, optimizer, opt):
    for i, (x, y) in enumerate(loader):
        if not model.training: 
            model.train()
        label_x = x.to(opt.device)
        label_y = y.to(opt.device)
        outs, _ = model(label_x)
        loss = criterion(outs, label_y)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        yield loss


def validationByBatch(loader, model, criterion, opt):
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
      
    src, tgt = batch
    src, tgt = src.to(opt.device), tgt.to(opt.device)
    out, _ = model(src)
    loss = criterion(out, tgt)

    _, preds_index = torch.max(out, dim=2)
    # _, src_index = torch.max(src, dim=2)
    # sources = []
    preds_str = []
    labels = []
    for index in range(opt.batch_size):
        pred_str = preds_index[index, :].tolist()
        eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
        if eos_res[0].any():
            eos_index = eos_res[0][0]
            pred_str = pred_str[:eos_index]
        preds_str.append(''.join(opt.charset.lookup_tokens(pred_str)))

        t = tgt[index, :].tolist()
        eos_res = np.where(np.equal(t, opt.charset.get_eos_index()))
        if eos_res[0].any():
            eos_index = eos_res[0][0]
        t = t[:eos_index]
        labels.append(''.join(opt.charset.lookup_tokens(t)))
       
      
        # s = src_index[index, :].tolist()
        # eos_res = np.where(np.equal(s, opt.charset.get_eos_index()))
        # if eos_res[0].any():
        #     eos_index = eos_res[0][0]
        # s = s[:eos_index]
        # sources.append(''.join(opt.charset.lookup_tokens(s)))

        tot_loss += loss.data.sum()
        tot_loss_count += loss.data.numel()

        # for src, gt, pred in zip(sources, labels, preds_str):
        for gt, pred in zip(labels, preds_str):
            # print(src, gt, pred)
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
