# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:56:49 2022

@author: lenovo
"""
import transformers
from transformers import AutoModel,AutoTokenizer, DataCollatorWithPadding
import torch
from torch import nn
import torch.nn.functional as F
import sys
import pandas as pd
from dataset import CustomDataset
import torch.utils.data as Data

class Bert(nn.Module):
    def __init__(self,checkpoint='bert-base-cased',pretrained_freeze = False,dropout_rate = 0.3,hidden_size = 768,num_classes = 6,uniform_init=0.1):
        super(Bert,self).__init__()
        #输入参数的存储
        self.checkpoint = checkpoint
        self.pretrained_freeze = pretrained_freeze
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.uniform_init= uniform_init
        
        #预训练模型
        self.bert = AutoModel.from_pretrained(checkpoint)
        if pretrained_freeze:#如果需要进行freeze
            for param in self.bert.parameters():
                param.requires_grad = False
        #模型head部分
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes)
            )
        #head部分的初始化
        print('initializing the head of Bert...')
        for param in self.fc.parameters():
            param.data.uniform_(-uniform_init, uniform_init)

    def forward(self,input_ids,attention_mask,token_type_id):
        output = self.bert(input_ids,attention_mask,token_type_id ,return_dict  = True)
        cls_h = output.last_hidden_state[:,0,:]#本身last_hidden_state的shape为(batch,seq_len,hidden_size)
        logits = self.fc(cls_h)#logits的shape为(batch,num_classes)
        return logits

    @staticmethod
    def load(model_path):#模型的加载
        print('loading best model from',model_path)
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = Bert(**args)# **表示传入的是字典
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path):
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(checkpoint=self.checkpoint,
                         pretrained_freeze = self.pretrained_freeze,
                         dropout_rate = self.dropout_rate,
                         hidden_size = self.hidden_size,
                         num_classes = self.num_classes,
                         uniform_init=self.uniform_init),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

class BertCNN(nn.Module):
    def __init__(self,checkpoint='bert-base-cased',pretrained_freeze=False,
                 hidden_size = 768,dropout_rate = 0.3,num_classes = 6,
                 uniform_init=0.1,num_kernels=64,kernel_size=[2,3,4]):#max_len扔掉了cls和sep
        super(BertCNN,self).__init__()
        #args保存
        self.checkpoint = checkpoint
        self.pretrained_freeze = pretrained_freeze
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.uniform_init= uniform_init    
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        #模型
        self.bert = AutoModel.from_pretrained(checkpoint)
        if not pretrained_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes)
            )
        
        self.conv_head_2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, 
                      out_channels=num_kernels, 
                      kernel_size=kernel_size[0]),#输入数据必须为(batch,channel,seq_len)
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
            )
        self.conv_head_3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, 
                      out_channels=num_kernels, 
                      kernel_size=kernel_size[1]),#输入数据必须为(batch,channel,seq_len)
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
            )
        self.conv_head_4 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, 
                      out_channels=num_kernels, 
                      kernel_size=kernel_size[2]),#输入数据必须为(batch,channel,seq_len)
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
            )
        self.conv_fc = nn.Sequential(
            nn.Linear(len(kernel_size)*num_kernels,len(kernel_size)*num_kernels),
            nn.Dropout(p=dropout_rate),
            nn.Linear(len(kernel_size)*num_kernels, num_classes)
            )
        
        #head部分的初始化
        print('initializing the head of Bert...')
        for param in self.fc.parameters():
            param.data.uniform_(-uniform_init, uniform_init)  
        for head in [self.conv_head_2,self.conv_head_3,self.conv_head_4]:
            for param in head.parameters():
                param.data.uniform_(-uniform_init, uniform_init)
        for param in self.conv_fc.parameters():
            param.data.uniform_(-uniform_init, uniform_init)
 
    def forward(self,input_ids,attention_mask,token_type_id):
        #cls部分的fc
        output = self.bert(input_ids,attention_mask,token_type_id ,return_dict  = True)
        cls_h = output.last_hidden_state[:,0,:]#本身last_hidden_state的shape为(batch,seq_len,hidden_size)
        logits = self.fc(cls_h)#logits的shape为(batch,num_classes)
        #text_cnn
        seq_h = output.last_hidden_state[:,1:-1,:]#batch,seq_len-2,hidden_state
        seq_h = seq_h.permute(0,2,1)#batch,hidden_state,seq_len-2 
        feature_vector = torch.cat((self.conv_head_2(seq_h).squeeze(2),
                                    self.conv_head_3(seq_h).squeeze(2),
                                    self.conv_head_4(seq_h).squeeze(2)),1)#feature_vector的shape为batch,3*num_kernels
        feature_vector = self.conv_fc(feature_vector)                                    
        return logits+feature_vector

    @staticmethod
    def load(model_path):#模型的加载
        print('loading best model from',model_path)
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BertCNN(**args)# **表示传入的是字典
        model.load_state_dict(params['state_dict'])
        return model

    def save(self, path):
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(checkpoint=self.checkpoint,
                         pretrained_freeze = self.pretrained_freeze,
                         dropout_rate = self.dropout_rate,
                         hidden_size = self.hidden_size,
                         num_classes = self.num_classes,
                         uniform_init=self.uniform_init,
                         num_kernels = self.num_kernels,
                         kernel_size = self.kernel_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

if __name__=='__main__':
    model=BertCNN()
    
    max_len = 256
    model_name = 'bert-base-cased'
    data_df = pd.read_csv(r'E:\0kaggle\preprocess\train_url.csv')[:7]
    train_batch_size = 8
    
    #检查这个dataset有没有问题
    train = CustomDataset(data_df,max_len=max_len,model_name=model_name)
    
    data_collator = DataCollatorWithPadding(tokenizer = train.tokenizer,padding = True)
    
    train_loader = Data.DataLoader(train,batch_size=train_batch_size,collate_fn=data_collator,shuffle = True,num_workers=0)
    
    for batch in train_loader:
        output = model(input_ids = batch['input_ids'],
                       attention_mask = batch['attention_mask'],
                       token_type_id = batch['token_type_ids'])
    
    