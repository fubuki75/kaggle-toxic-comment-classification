# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:33:09 2022

@author: lenovo
"""
import torch
import numpy as np
import os
import random
import pandas as pd
import torch.utils.data as Data
import transformers
import time
from tqdm import tqdm
import sys
from transformers import AutoTokenizer, DataCollatorWithPadding
from dataset import CustomDataset
from model import Bertv2
from transformers import AdamW
from sklearn.metrics import roc_auc_score

    
##参数
max_len = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_batch_size = 32
val_batch_size = 32
#test相关
test_batch_size = 32
output_path = r'submission.csv'#test set结果的输出
#训练相关
learning_rate = 1e-5
weight_decay = 1e-2
log_every = 20
valid_niter = 1200
model_save_path = 'model.bin'
max_patience = 5
max_num_trial = 5
lr_decay = 0.5
max_epoch =100
#模型相关
model_name = 'bert-base-cased'
pretrained_freeze = False
dropout_rate = 0.3
hidden_size = 768
num_classes = 6
init_value = 0.1
#num_kernels=256
#kernel_size=[2,3,4]
pooler_depth=4

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_val_split(df,val_size = 0.2,shuffle = False, random_state = None):
    if shuffle:
        df_shuffle = df.sample(frac=1,random_state = random_state).reset_index(drop = True)
    train_set = df_shuffle.loc[int(val_size*len(df)):].reset_index(drop = True)
    val_set = df_shuffle.loc[:int(val_size*len(df))].reset_index(drop = True)
    return train_set,val_set

def loss_fn(output,target):
    return torch.nn.BCEWithLogitsLoss()(output,target)

def evaluate_auc(model, valloader):
    was_training = model.training#根据是否在训练返回bool型变量
    model.eval()
    #记录预测结果与ground truth
    fin_targets = []
    fin_outputs = []
    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():#被with torch.no_grad()的代码不会track反向梯度，仍保持之前的梯度情况，适合于dev
        for batch in valloader:#验证，就来一遍就行，还是一个batch一个batch来，不然可能装不下
            batch = batch.to(device)
            output = model(input_ids = batch['input_ids'],
                           attention_mask = batch['attention_mask'],
                           token_type_id = batch['token_type_ids'])

            targets = batch['labels']
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist()) #output需要进行sigmoid转化为概率       

    if was_training:
        model.train()

    return roc_auc_score(fin_targets, fin_outputs,average = 'weighted')

def train(model,trainloader,valloader,optimizer,loss_fn,log_every,valid_niter,model_save_path,
          max_patience,max_num_trial,lr_decay,max_epoch):
    #训练标记
    model.train()
    print('use device: %s' % device)
    model = model.to(device)#模型传输至device
    
    num_trial = 0
    train_iter = patience = 0
    epoch = valid_num = 0
    
    #这里有意记录valid的分数
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin training...')

    while True:#early stop为终止
        epoch += 1
        for batch in trainloader:
            train_iter += 1 #循环计数
            batch = batch.to(device)#将数据传入device
            #套路
            optimizer.zero_grad()
            output = model(input_ids = batch['input_ids'],
                           attention_mask = batch['attention_mask'],
                           token_type_id = batch['token_type_ids'])
            loss = loss_fn(output,batch['labels'])
            loss.backward()
            optimizer.step()
            
            #每多少个循环输出一次日志
            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, time elapsed %.2f sec' % (epoch,
                                                                                    train_iter,
                                                                                    loss.item(),
                                                                                    time.time() - begin_time), file=sys.stderr)

            # perform validation 这里很重要，基于validation set判断是否该停止
            #没多少循环进行一次验证
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, avg. loss %.2f' % (epoch, train_iter,loss.item()), file=sys.stderr)
                valid_num += 1
                print('begin validation ...', file=sys.stderr)
                
                # 用验证集计算AUC
                valid_metric = evaluate_auc(model, valloader)   
                print('validation: iter %d, AUC %f' % (train_iter, valid_metric), file=sys.stderr)
                
                #判断是不是最好的结果
                #首次验证：is_better直接是true
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
                
            #整体的逻辑：
            #若目前的最优，保存下来
            #结果不是最优时，累积一定次数（patience）decay一次学习率且记录num_trial，decay一定次数后，再停止
                
            #目前最好的结果
                if is_better:
                    #目前最好的结果，保存模型，且patience清0（控制学习率衰减）
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    #同时保存模型和优化器的状态
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
            #当前结果不比之前所有的结果更好
                elif patience < max_patience:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    
                    if patience == max_patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        #decay一定次数之后，就停止
                        if num_trial == max_num_trial:
                            print('early stop!', file=sys.stderr)
                            #程序直接终止
                            exit(0)

                        # 衰减学习的写法！不仅要decay还要恢复到原来的model和优化器
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
                        #这种参数加载方法，读model，但加载是model的参数
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        #这里map_location=lambda storage,loc:storage的含义是将GPU上的模型加载到CPU上；如果没有这些参数就是同种设备上的加载
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)
                        
                        #load优化器
                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == max_epoch:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)
    

def main(reload_weight = False):
    # 可重复性
    set_seed(1)
    #读取数据文件，划分训练验证集
    raw_dataset = pd.read_csv(r'train_url.csv',encoding='utf-8')#
    train_set,val_set = train_val_split(raw_dataset,shuffle = True,random_state = 1)#作为例子暂时只读100
    print(len(train_set))
    print(len(val_set))
    #dataloader_train
    print('reading training data...')
    train_data = CustomDataset(train_set,max_len,
                               with_labels = True,
                               model_name = model_name)
    data_collator_train = DataCollatorWithPadding(tokenizer = train_data.tokenizer,padding = True)   
    train_loader = Data.DataLoader(train_data,
                                   batch_size=train_batch_size,
                                   collate_fn=data_collator_train,
                                   shuffle = True,
                                   num_workers=0)
    
    #dataloader_val
    print('reading validation data...')
    val_data = CustomDataset(val_set, max_len,
                             with_labels = True,
                             model_name = model_name)
    data_collator_val = DataCollatorWithPadding(tokenizer = val_data.tokenizer,padding = True)   
    val_loader = Data.DataLoader(val_data,
                                 batch_size=val_batch_size,
                                 collate_fn=data_collator_val,
                                 shuffle = True,
                                 num_workers=0)
    
    #model
    model = Bertv2(checkpoint=model_name,
                    pretrained_freeze = pretrained_freeze,
                    dropout_rate = dropout_rate,
                    hidden_size = hidden_size,
                    num_classes = num_classes,
                    uniform_init=init_value,
                    pooler_depth=pooler_depth)
    
    
    #优化器
    optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    #google colab是否断连
    if reload_weight:#遇到了断连需要重新载入模型
        print('load previously best model...', file=sys.stderr)
        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
        #这里map_location=lambda storage,loc:storage的含义是将GPU上的模型加载到CPU上；如果没有这些参数就是同种设备上的加载
        model.load_state_dict(params['state_dict'])
        model = model.to(device)
        #load优化器
        print('restore parameters of the optimizers...', file=sys.stderr)
        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
        print('and the learning rate is [%s]' %(optimizer.param_groups[0]['lr']), file=sys.stderr)
                            
    #train_loop
    train(model,train_loader,val_loader,optimizer,loss_fn,log_every,valid_niter,model_save_path,
          max_patience,max_num_trial,lr_decay,max_epoch)
    
def test(model_save_path,test_loader):
    #加载模型
    best_model = Bertv2.load(model_save_path)
    best_model = best_model.to(device)
    #预测
    print('tests begin...')
    best_model.eval()
    #记录预测结果
    fin_outputs = []
    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():#被with torch.no_grad()的代码不会track反向梯度，仍保持之前的梯度情况，适合于dev
        for batch in test_loader:#验证，就来一遍就行，还是一个batch一个batch来，不然可能装不下
            batch = batch.to(device)
            output = best_model(input_ids = batch['input_ids'],
                                attention_mask = batch['attention_mask'],
                                token_type_id = batch['token_type_ids'])

            #targets = batch['labels']
            #fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist()) #output需要进行sigmoid转化为概率       
    #结果返回
    return pd.DataFrame(fin_outputs).rename(columns={0:'toxic',1:'severe_toxic',2:'obscene',
                                                     3:'threat',4:'insult',5:'identity_hate'})

if __name__ == '__main__':
    main(reload_weight=True)
    
    '''
    #testing
    print('reading test data...')
    test_set = pd.read_csv(r'test_url.csv',encoding = 'utf-8')
    test_data=CustomDataset(test_set, max_len,with_labels=False,model_name=model_name)
    data_collator_test = DataCollatorWithPadding(tokenizer = test_data.tokenizer,padding = True)   
    test_loader = Data.DataLoader(test_data,
                                  batch_size=test_batch_size,
                                  collate_fn=data_collator_test,
                                  shuffle = False,
                                  num_workers = 0
                                  )

    result_df = test(model_save_path,test_loader)#传入最优模型地址，测试数据loader
    #横向拼接
    submit = pd.concat([test_set,result_df],axis=1).drop(columns=['comment_text'])
    submit.to_csv(output_path,index=False)
    print('test finished!')
    '''