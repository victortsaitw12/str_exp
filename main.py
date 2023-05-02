#!/usr/bin/python
# -*- coding: UTF-8 -*-

from utility import Charset
import torch
import argparse
from utility import MyLogger, check_checkpoints
from pretrain import pretrain
from pretrainbcn import pretrainBCN
from train import train
from predict import predict
from benchmark import benchmark
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # utility
    parser.add_argument('--action', default="predict", 
                        help='predict || train || pretrain || finetune || benchmark')
    parser.add_argument('--debug_mode', default=False, help='debug mode')
    parser.add_argument('--log_path', default='info.log')
    parser.add_argument('--save_path', default=r'C:\Users\victor\Desktop\experiment\checkpoints')
    parser.add_argument('--encoder_path', default=r'C:\Users\victor\Desktop\experiment\checkpoints\seqclr')
    parser.add_argument('--save_step', type=int, default=200)

    # dataset
    parser.add_argument('--lmdb_root', default=r'C:\Users\victor\Desktop\experiment\datasets\TCSynth')
    parser.add_argument('--raw_root', default=r'C:\Users\victor\Desktop\experiment\datasets\TC-STR')
    parser.add_argument('--validation_path', default='train_labels.txt')
    parser.add_argument('--test_path', default='test_labels.txt')
    parser.add_argument('--char_path', default=r'C:\Users\victor\Desktop\experiment\datasets\cht_tra_characters.txt')
    parser.add_argument('--data_limit', default=0.05)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--img_h', type=int, default=32)
    parser.add_argument('--rgb', action='store_true', default=True, help='rgb')
    parser.add_argument('--max_len', type=int, default=50)
    # Training
    # parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64,
                        help="train:64 || pretrain:304")
    parser.add_argument('--optimizer', default='adam', help='adam || adamW')
    parser.add_argument('--lr', default=1e-4, help='1e-4')
    parser.add_argument('--wd', default=0, help="0 || 0.05")
    parser.add_argument('--grad_clip', type=int, default=5, 
                        help="20 || 5")
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_fiducial', default=20, help="for TPS")
    # architecture
    parser.add_argument('--freeze_encoder', default=False)
    parser.add_argument('--trans', default="None", 
                        help='TPS || None') 
    parser.add_argument('--encoder', default="ViTSTR", 
                        help='VGG || ResNet || GRCNN || SVTR_L || SVTR_T || ViTSTR || None') 
    parser.add_argument('--encoder_with_transformer', default=False)
    parser.add_argument('--SequenceModeling', default="None",
                        help="BiLSTM || Attn | Position Attn || None")
    parser.add_argument('--decoder', default="None", 
                        help='CTC || SeqAttn || Transformer || LM || None')
    parser.add_argument('--language_module', default="None", 
                        help='BCN || None')
    parser.add_argument('--language_module_checkpoint', default=r"C:\Users\victor\Desktop\experiment\checkpoints\lm", 
                        help='path || None')
    
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=384,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, 
                        help='the size of the LSTM hidden state')
    parser.add_argument('--transformation_encoder_ln', type=int, default=3, 
                        help='transformation_encoder_layers')
    parser.add_argument('--transformation_decoder_ln', type=int, default=3, 
                        help='transformation_decoder_layers')
    # Archetecture for SeqCLR
    parser.add_argument('--projection_input_channel', type=int, default=3,
                        help='the number of input channel of SeqCLR')
    parser.add_argument('--projection_output_channel', type=int, default=512,
                        help='the number of output channel of SeqCLR')
    parser.add_argument('--projection_hidden_size', type=int, default=256, 
                        help='the size of the SeqCLR hidden state')
    # Predict
    parser.add_argument('--predict_img', default=r'C:\Users\victor\Desktop\experiment\datasets\TC-STR\images\poster_03690_438_菸灰缸.jpg')

    opt = parser.parse_args()
    log = MyLogger(opt.log_path)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('device: {}'.format(opt.device))


    if opt.decoder == 'CTC':
      charset = Charset(char_path=opt.char_path, specials=['<blk>', '<unk>'])
      charset.set_default_index(1)
      charset.set_pad_index(0)
      charset.set_bos_index(0)
      charset.set_eos_index(0)

    elif opt.decoder == 'LM':
      charset = Charset(char_path=opt.char_path, specials=['</s>', '<pad>', '<unk>'])
      charset.set_default_index(2)
      charset.set_pad_index(1)
      charset.set_eos_index(0)
      charset.set_bos_index(0)
      
    # elif opt.decoder == 'Attn':
    #   charset = Charset(char_path=opt.char_path, specials=['<s>', '</s>', '<unk>'])
    #   charset.set_default_index(2)
    #   charset.set_pad_index(0)
    #   charset.set_bos_index(0)
    #   charset.set_eos_index(1)

    else:
      charset = Charset(char_path=opt.char_path, specials=['<s>', '</s>', '<pad>', '<unk>'])
      charset.set_default_index(3)
      charset.set_pad_index(2)
      charset.set_bos_index(0)
      charset.set_eos_index(1)

    opt.num_class = len(charset)
    opt.charset = charset

    if opt.action == 'pretrain' and opt.encoder != 'None':
      opt.projection_input_channel = opt.output_channel
      pretrain(opt, log)
    elif opt.action == 'pretrain' and opt.decoder == 'LM':
      pretrainBCN(opt, log)
    elif opt.action == 'predict':
      opt.batch_size = 1
      predict(opt, log)
    elif opt.action == 'benchmark':
      opt.batch_size = 1
      benchmark(opt, log)
    else:
      if opt.action == 'finetune' and check_checkpoints(opt.save_path):
        print(f'In finetune, there is {opt.save_path}.')
        print('please remove it or change to train mode to load it!!')
        sys.exit(0)

      if opt.freeze_encoder:
        check_freeze = input('The encoder will be frozen, do you want to continue?[y/n]')
        if check_freeze != 'y':
          sys.exit(0)
      
      log(f'action:{opt.action}, freeze_encoder:{opt.freeze_encoder}')
      train(opt, log)
