#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.lmdbdataset import LmdbDataset
from models.seqclrmodel import SeqCLRModel
from loss.seqclrloss import SeqCLRLoss
from utility import MyLogger, save_ckp, load_ckp

def pretrain(opt, log):
    pretrain_dataset = LmdbDataset(root=opt.lmdb_root, 
                                    img_w=opt.img_w, img_h=opt.img_h, 
                                    charset=None, rgb=opt.rgb, 
                                    pretrain=True, limit=opt.data_limit)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=opt.batch_size, shuffle=False,
                                 pin_memory=True, num_workers=opt.num_workers)
    log('[SeqCLR]pretrain_loader:{}, dataset:{}'.format(
        len(pretrain_loader), len(pretrain_loader.dataset)
    ))
    model=SeqCLRModel(opt).to(opt.device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,17], gamma=0.1)
    criterion = SeqCLRLoss()
    model, optimizer, start_epoch, step, scheduler = load_ckp(model, optimizer, opt.encoder_path, opt.device, scheduler)
    log('[SeqCLR]start epoch:{}/{}'.format(start_epoch, opt.epochs))
    for epoch in range(start_epoch, opt.epochs):
        tot_loss = 0
        tot_loss_count = 0
        for loss in pretrainByBatch(pretrain_loader, model, criterion, optimizer, opt):
            tot_loss += loss.data.sum()
            tot_loss_count += loss.data.numel()

            if step != 0 and step % opt.save_step == 0:
                # save checkpoints
                save_ckp(model.state_dict(), optimizer.state_dict(),
                        epoch, step, opt.encoder_path,
                        scheduler_last_epoch=scheduler.last_epoch,
                        scheduler_step_count=scheduler._step_count)
                log(f'[Train]epoch:{epoch}, step:{step}, Loss:{tot_loss/float(tot_loss_count):0.5f}')
                # Write to the tensorboard
                log.add_scalar('Pretrain/Loss', tot_loss/float(tot_loss_count), step)
                tot_loss = 0
                tot_loss_count = 0

            step += 1
        
        scheduler.step()
        log.add_scalar('Pretrain/SeqCLCR_LR', scheduler.get_last_lr()[0], epoch)

    log.close()
    save_ckp(model.state_dict(), optimizer.state_dict(),
            epoch, step, opt.encoder_path,
            scheduler_last_epoch=scheduler.last_epoch,
            scheduler_step_count=scheduler._step_count)        
    log('[SeqCLR]Done Training!!!!!!!!!!!!!!!')

def pretrainByBatch(loader, model, criterion, optimizer, opt):
    for i, images in enumerate(loader):
        if not model.training: 
            model.train()
        images = images.to(opt.device)
        features_view0, features_view1 = model(images)
        loss = criterion(features_view0, features_view1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        yield loss
