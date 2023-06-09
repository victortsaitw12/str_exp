#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import numpy as np
from nltk.metrics.distance import edit_distance
from models.model import Model
import os
from PIL import Image
from torchvision.transforms import transforms
from torchsummary import summary
import time

def predict(opt, log):
    print(f'predict {opt.predict_img}')
    if opt.rgb:
        img = Image.open(opt.predict_img).convert('RGB')
    else:
        img = Image.open(opt.predict_img).convert('L')
    img = img.resize((opt.img_w, opt.img_h))
    # img.show()
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(opt.device)
    # Load saved model
    model=Model(opt).to(opt.device)
    f_path = os.path.join(opt.save_path, 'checkpoint.pt')
    print(f'loading pretrained model from {f_path}')
    if not os.path.exists(f_path):
        raise Exception(f'Checkpoints not found at {f_path}')
    checkpoint = torch.load(f_path, map_location=opt.device)
    model.load_state_dict(checkpoint['state_dict'])

    # Inference
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        if opt.decoder == 'CTC':
            tgt = torch.LongTensor(opt.batch_size, opt.max_len)
            # summary(model, [(3, 224, 224), (1, 50)])
            summary(model, [(3, 32, 128), (50,)], dtypes=[torch.float, torch.long])
            out = model(img, tgt)
            print('Duration:', time.time() - start_time)
            _, preds_index = out.max(2)
            t = preds_index.squeeze().tolist()
            char_list = [t[i] for i in range(len(t)) 
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i]))
            ]  # removing repeated characters and blank.
            preds_str = ''.join(opt.charset.lookup_tokens(char_list))
        elif opt.decoder == 'LM':
            tgt = torch.LongTensor(opt.batch_size, opt.max_len)
            # summary(model.encoder, [(3, 32, 128)], dtypes=[torch.float])
            summary(model, [(3, 32, 128), (50,)], dtypes=[torch.float, torch.long])
            out = model(img, tgt)
            print('Duration:', time.time() - start_time)
            _, preds_index = torch.max(out[0][-1], dim=2)

            for index in range(opt.batch_size):
                pred_str = preds_index[index, :].tolist()
                eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
                if eos_res[0].any():
                    eos_index = eos_res[0][0]
                    pred_str = pred_str[:eos_index]
                preds_str = ''.join(opt.charset.lookup_tokens(pred_str))

        elif opt.encoder == 'ViTSTR':
            tgt = torch.zeros(opt.batch_size, opt.max_len).to(opt.device)
            summary(model, [(3, 224, 224), (50,)], dtypes=[torch.float, torch.long])
            out = model(img, tgt, is_train=False)
            print('Duration:', time.time() - start_time)
            _, preds_index = torch.max(out, dim=2)
            
            
            for i in range(opt.batch_size):
                pred_str = preds_index[i].tolist()
                eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
                if eos_res[0].any():
                    eos_index = eos_res[0][0]
                    pred_str = pred_str[1:eos_index]
                preds_str = ''.join(opt.charset.lookup_tokens(pred_str))
        else:
            tgt = torch.LongTensor(opt.batch_size, opt.max_len)
            tgt.fill_(opt.charset.get_bos_index())
            tgt = tgt.to(opt.device)
            summary(model, [(3, 32, 128), (50,)], dtypes=[torch.float, torch.long])
            # summary(model.encoder, (3, 32, 128))
            # summary(model.SequenceModeling, (32, 384))
            # summary(model.decoder, [(32, 256), (50,)], dtypes=[torch.float, torch.long])
            out = model(img, tgt)
            print('Duration:', time.time() - start_time)
            _, preds_index = torch.max(out, dim=2)

            for i in range(opt.batch_size):        
                pred_str = preds_index[i].tolist()
                eos_res = np.where(np.equal(pred_str, opt.charset.get_eos_index()))
                if eos_res[0].any():
                    eos_index = eos_res[0][0]
                    pred_str = pred_str[:eos_index]
                preds_str = ''.join(opt.charset.lookup_tokens(pred_str))
        print('predict result:', preds_str)

    