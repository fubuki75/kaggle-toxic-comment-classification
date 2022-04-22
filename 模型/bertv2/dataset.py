# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import pandas as pd
import torch.utils.data as Data
import numpy as np
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding

class CustomDataset(Data.Dataset):
    def __init__(self,data,max_len,with_labels = True,model_name = 'bert-base-cased',transform=None,target_transform=None):
        self.data = data# dataFrame
        # initialization of tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.with_labels = with_labels
        #最前面的预处理
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        sample={}
        #selection of sentence(s)
        sent = str(self.data.loc[index,'comment_text'])
        #transform
        if self.transform != None:
            sent = self.transform(sent)
        #encoding
        encoded_sent = self.tokenizer(sent,
                                      truncation = True,
                                      max_length = self.max_len, 
                                      return_tensors = 'pt')#返回input_ids,attention_mask,token_type_ids
        
        #sample['idx'] = str(self.data.loc[index,'id'])
        #sample['sentence'] = sent 要么在这里就不含这两个str，要么在建立dataloader前得去掉
        
        sample['input_ids'] = encoded_sent['input_ids'].squeeze(0)#将list转化为一维
        sample['attention_mask'] = encoded_sent['attention_mask'].squeeze(0)
        sample['token_type_ids'] = encoded_sent['token_type_ids'].squeeze(0)
        if self.with_labels:
            label = torch.Tensor(self.data.iloc[index,-6:].values.astype(float)).type(torch.float)
            if self.target_transform!=None:
                label = self.target_transform(label)
            sample['labels'] = label
        return sample
    
if __name__ == '__main__':
    #参数
    max_len = 512
    model_name = 'bert-base-cased'
    data_df = pd.read_csv(r'D:\0kaggle\preprocess\train_url.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_batch_size = 8
    
    #检查这个dataset有没有问题
    train = CustomDataset(data_df,max_len=max_len,model_name=model_name)
    
    data_collator = DataCollatorWithPadding(tokenizer = train.tokenizer,padding = True)
    
    train_loader = Data.DataLoader(train,batch_size=train_batch_size,collate_fn=data_collator,shuffle = True,num_workers=0)
    
    i=0
    for sample in train_loader:
        if i<10:
            print( sample['attention_mask'].shape)
            i+=1
        else:
            break
    
    
    
        
    