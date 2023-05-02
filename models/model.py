import torch
import torch.nn as nn
from models.bilstm import BidirectionalLSTM
from models.vgg import VGG
from models.resnet import ResNet50
from models.resnet45 import ResNet45
from models.grcnn import GRCNN
from models.seqattentiondecoder import SeqAttention
from models.transformerencoder import TransformerEncoder
from models.transformerdecoder import TransformerDecoder
from models.tps import TPS_SpatialTransformerNetwork
from models.svtr import large_svtr, tiny_svtr
from models.bcnlanguage import BCNEncoder, BCNAlignment
from models.vitstr import ViTSTR

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.stages = {
            'Trans': opt.trans,
            'encoder': opt.encoder,
            'encoder_with_transformer': opt.encoder_with_transformer
        }
        
        """ Transformation """
        if opt.trans == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.img_h, opt.img_w), I_r_size=(opt.img_h, opt.img_w), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.encoder == 'VGG':
            self.encoder = VGG(opt.input_channel, opt.output_channel)

        elif opt.encoder == 'GRCNN':
            self.encoder = GRCNN(opt.input_channel, opt.output_channel)

        elif opt.encoder == 'ResNet':
            self.encoder = ResNet45(opt.input_channel, opt.output_channel)
        
        elif opt.encoder == 'SVTR_L':
            self.encoder = large_svtr(img_size=[opt.img_h, opt.img_w], 
                                      max_seq_len=opt.max_len,
                                      out_channels=opt.output_channel)
        elif opt.encoder == 'SVTR_T':
            self.encoder = tiny_svtr(img_size=[opt.img_h, opt.img_w], 
                                      max_seq_len=opt.max_len,
                                      out_channels=opt.output_channel)
        elif opt.encoder == 'ViTSTR':
              self.encoder = ViTSTR(max_len=opt.max_len,
                                    emb_size=opt.output_channel,
                                    num_class=opt.num_class)
        else:
            raise Exception('No FeatureExtraction module specified')

        if opt.encoder_with_transformer:
            self.transformerEncoder = TransformerEncoder(input_size=opt.output_channel, 
                                                         dropout=opt.dropout, 
                                                         device=opt.device,
                                                         layers=opt.transformation_encoder_ln)

    def forward(self, input):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        # ResNet45[N,C,8,32] || ResNet, GRCNN[N,C,1,33] || VGG[N,C,1,31]
        visual_feature = self.encoder(input) 
        
        if self.stages['encoder_with_transformer']: # do not change shape
            visual_feature = self.transformerEncoder(visual_feature)
        return visual_feature


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {
            'Trans': opt.trans, 
            'encoder': opt.encoder,
            'encoder_with_transformer': opt.encoder_with_transformer,
            'Seq': opt.SequenceModeling, 
            'decoder': opt.decoder,
            'LM': opt.language_module
        }
        self.encoder = Encoder(opt)
        """ Transformation """
        # if opt.Transformation == 'TPS':
        #     self.Transformation = TPS_SpatialTransformerNetwork(
        #         F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        # else:
        #     print('No Transformation module specified')

        """ FeatureExtraction """
        # if opt.encoder == 'VGG':
        #     self.encoder = VGG(opt.input_channel, opt.output_channel)

        # elif opt.encoder == 'GRCNN':
        #     self.encoder = GRCNN(opt.input_channel, opt.output_channel)

        # elif opt.encoder == 'ResNet':
        #     self.encoder = ResNet45(opt.input_channel, opt.output_channel)

        # else:
        #     raise Exception('No FeatureExtraction module specified')
        
        # if opt.encoder_with_transformer:
        #     self.transformerEncoder = TransformerEncoder(opt.output_channel, opt.dropout, opt.device)

        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size

        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.decoder == 'CTC':
            self.decoder = nn.Linear(self.SequenceModeling_output, opt.num_class)

        elif opt.decoder == 'LM':
            self.decoder = BCNEncoder(self.SequenceModeling_output, opt.num_class)
            # self.decoder = nn.Linear(self.SequenceModeling_output, opt.num_class)

        elif opt.decoder == 'SeqAttn':
            self.decoder = SeqAttention(self.SequenceModeling_output, opt.hidden_size, 
                                     opt.num_class)
            
        elif opt.decoder == 'Transformer':
            self.decoder = TransformerDecoder(input_size=self.SequenceModeling_output, 
                                              num_classes=opt.num_class,
                                              layers=opt.transformation_decoder_ln,
                                              dropout=opt.dropout, 
                                              pad=opt.charset.get_pad_index())

        else:
            self.decoder = nn.Identity()
            # raise Exception('Prediction is neither CTC or Attn')
        
        if opt.decoder == 'LM' and opt.language_module != 'None':
            self.language_module = BCNAlignment(
              input_channel=self.SequenceModeling_output,
              num_classes=opt.num_class,
              max_length=opt.max_len,
              eos_index=opt.charset.get_eos_index(),
              opt=opt
            )
        else:
            print('No Language module specified')
        
    def load_encoder_state(self, states_dict):
        print('load encoder state')
        self.encoder.load_state_dict(states_dict)

    def freeze_encoder(self):
        print('freeze encoder')
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input, text, is_train=True):
        # """ Transformation stage """
        # # if not self.stages['Trans'] == "None":
        # #     input = self.Transformation(input)

        # """ Feature extraction stage """
        # # ResNet45[N,C,8,32] || ResNet, GRCNN[N,C,1,33] || VGG[N,C,1,31]
        # visual_feature = self.encoder(input) 
        
        # if self.stages['encoder_with_transformer']: # do not change shape
        #     visual_feature = self.transformerEncoder(visual_feature)

        visual_feature = self.encoder(input)
        if len(visual_feature.shape) == 4:
            # ResNet45[N,32,C,1] || ResNet, GRCNN[N,33,C,1] || VGG[N,31,C,1]
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            
            # ResNet45[N,32,C] || ResNet, GRCNN[N,33,C] || VGG[N,31,C]
            visual_feature = visual_feature.squeeze(3)


        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            # VGG-BiLSTM[N, 31, H]
            contextual_feature = self.SequenceModeling(visual_feature) 

        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['decoder'] == 'CTC':
            prediction = self.decoder(contextual_feature.contiguous())

        elif self.stages['decoder'] == 'LM':
            prediction = self.decoder(contextual_feature.contiguous())
            
        elif self.stages['decoder'] == 'SeqAttn':
            prediction = self.decoder(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.max_len)
        else:
            prediction = contextual_feature

        if self.stages['decoder'] == 'LM' and self.stages['LM'] != 'None':
            # tokens = torch.softmax(prediction, dim=-1)
            prediction = self.language_module(prediction)
            # prediction = (l_prediction, prediction)

        return prediction