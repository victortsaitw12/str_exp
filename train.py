#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.lmdbdataset import LmdbDataset
from dataset.rawdataset import RawDataset
from models.model import Model
from utility import MyLogger, save_ckp, load_ckp, load_encoder_ckp, img_show
from utility import ctc_collate_fn, attn_collate_fn, lm_collate_fn, onehot_collate_fn, vit_collate_fn
from validation import validation
from loss.multiloss import MultiLoss

def train(opt, log):
    # === Load dataset ===
    train_dataset = LmdbDataset(root=opt.lmdb_root, 
                                img_w=opt.img_w, img_h=opt.img_h, 
                                charset=opt.charset, rgb=opt.rgb, 
                                pretrain=False, limit=opt.data_limit)
    # train_dataset = RawDataset(root=opt.raw_root, file_path=opt.validation_path,
    #                             img_w=opt.img_w, img_h=opt.img_h, 
    #                             charset=opt.charset, rgb=opt.rgb)
    valid_dataset = RawDataset(root=opt.raw_root, file_path=opt.validation_path,
                                img_w=opt.img_w, img_h=opt.img_h, 
                                charset=opt.charset, rgb=opt.rgb)
  
    if opt.decoder == 'CTC':
        collate_fn = lambda batch: ctc_collate_fn(batch, max_len=opt.max_len, 
            pad=opt.charset.get_pad_index())
        
    elif opt.decoder == 'LM':
        # collate_fn = lambda batch: lm_collate_fn(batch, max_len=opt.max_len,
        #     eos=opt.charset.get_eos_index(), 
        #     pad=opt.charset.get_pad_index())
        collate_fn = lambda batch: onehot_collate_fn(batch, 
            max_len=opt.max_len,
            charset=opt.charset,
            device=opt.device)
    else:
        if opt.encoder == 'ViTSTR':
             collate_fn = lambda batch: vit_collate_fn(batch=batch, max_len=opt.max_len,
              bos=opt.charset.get_bos_index(), 
              eos=opt.charset.get_eos_index(),
              pad=opt.charset.get_pad_index()
          )
        else:
          collate_fn = lambda batch: attn_collate_fn(batch=batch, max_len=opt.max_len,
              bos=opt.charset.get_bos_index(), 
              eos=opt.charset.get_eos_index(),
              pad=opt.charset.get_pad_index()
          )
      
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True,
                              num_workers=opt.num_workers)
    
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True,
                              num_workers=opt.num_workers)
    
    log('[Train]train_loader:{}, dataset:{}'.format(
        len(train_loader), len(train_loader.dataset)
    ))
    log('[Train]valid_loader:{}, dataset:{}'.format(
        len(valid_loader), len(valid_loader.dataset)
    ))

    # === Defome Model, Criterion and Optimizer ===
    model=Model(opt).to(opt.device)
    if opt.optimizer == 'adamW':
      optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    else:
      optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

    if opt.decoder == 'CTC':
      criterion = nn.CTCLoss(zero_infinity=True)
    elif opt.decoder == 'LM':
      criterion = MultiLoss(ignore_index=opt.charset.get_pad_index())
    else:
      criterion = nn.CrossEntropyLoss(ignore_index=opt.charset.get_pad_index())
  
    # === Load previous status ===
    start_epoch = 0
    step = 0
    if opt.action == 'finetune':
       model = load_encoder_ckp(model, opt.encoder_path, opt.device)
    else:
       model, optimizer, start_epoch, step, _ = load_ckp(model, optimizer, opt.save_path, opt.device)
    if opt.freeze_encoder:
       model.freeze_encoder()
    log('[Train]start epoch:{}/{}'.format(start_epoch, opt.epochs))

    # === Train ===
    for epoch in range(start_epoch, opt.epochs):
        tot_loss = 0
        tot_loss_count = 0
        for loss in train_one_batch(train_loader, model, criterion, optimizer, opt):
          tot_loss += loss.data.sum()
          tot_loss_count += loss.data.numel()
          
          if step != 0 and step % opt.save_step == 0:
            valid_loss, accuracy, norm_ED = \
              validation(valid_loader, model, criterion, optimizer, opt)
            log(f'[Valid]epoch:{epoch}, step:{step}, Loss:{valid_loss:0.5f}')
            # Write to the tensorboard
            log.add_scalar('Valid/Loss', valid_loss, step)
            log.add_scalar('Valid/Accuracy', accuracy, step)
            log.add_scalar('Valid/Norm_ED', norm_ED, step)

            # save checkpoints
            save_ckp(model.state_dict(), optimizer.state_dict(), epoch, step, opt.save_path)
            log(f'[Train]epoch:{epoch}, step:{step}, Loss:{tot_loss/float(tot_loss_count):0.5f}')
            # Write to the tensorboard
            log.add_scalar('Train/Loss', tot_loss/float(tot_loss_count), step)
            tot_loss = 0
            tot_loss_count = 0

          
          step += 1
    log.close()
    save_ckp(model.state_dict(), optimizer.state_dict(), epoch, step, opt.save_path)
    log('[Train]Done Training!!!!!!!!!!!!!!!')

def train_one_batch(loader, model, criterion, optimizer, opt):
  # model.train()
  # Train
  for i, batch in enumerate(loader):
    if not model.training: 
       model.train()
    if opt.decoder == 'CTC':
      src, tgt, n_tokens = batch
      src, tgt = src.to(opt.device), tgt.to(opt.device)
      out = model(src, tgt)
      out_size = torch.IntTensor([out.size(1)] * opt.batch_size)
      out = out.log_softmax(2).permute(1, 0, 2)
      loss = criterion(out, tgt, out_size, n_tokens)

    elif opt.decoder == 'LM':
      src, tgt = batch
      src, tgt = src.to(opt.device), tgt.to(opt.device)
      out = model(src, tgt)
      loss = criterion(out, tgt)
      # l_out, v_out = model(src, tgt)
      # l_loss = criterion(
      #   l_out.contiguous().view(-1, l_out.size(-1)),
      #   tgt.contiguous().view(-1)
      # )
      # v_loss = criterion(
      #   v_out.contiguous().view(-1, v_out.size(-1)),
      #   tgt.contiguous().view(-1)
      # )
      # loss = sum([l_loss * 0.5, v_loss * 0.5])

    else:
      src, tgt, tgt_y, n_tokens = batch
      src, tgt, tgt_y = src.to(opt.device), tgt.to(opt.device), tgt_y.to(opt.device)
      out = model(src, tgt)
      # print('out:', out.shape)
    # out = model.projector(out)
      loss = criterion(
        out.contiguous().view(-1, out.size(-1)),
        tgt_y.contiguous().view(-1)
      )
    optimizer.zero_grad()
    # a = list(model.parameters())[0].clone()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
    optimizer.step()
    # b = list(model.parameters())[0].clone()
    # print(torch.equal(a.data, b.data))
    yield loss
