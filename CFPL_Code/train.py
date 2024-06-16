import os
import sys
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertConfig
import tensorflow_hub as hub
import tensorflow as tf
import copy as cp

from losses import Loss_fn
from bert_atten import MyModel
from parser_util import get_parser
from data_loader import FewShotRawDataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def init_dataloader(args, mode):
    filePath = os.path.join(args.dataFile, mode + '.json')
    data_loader = FewShotRawDataLoader()
    examples, few_shot_batches, max_support_size = data_loader.load_data(filePath)
    return few_shot_batches

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def init_model(args):
    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = MyModel(args).to(device)
    return model

def init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    return optimizer

def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    
    t_total = args.epochs * args.episodeTrain
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler

def deal_label(episode_labels, support_labels, query_labels):
    label2ids = defaultdict(int)
    for i, label in enumerate(episode_labels):
        label2ids[label] = i
    support_ids, query_ids = [], []
    for labels in support_labels:
        tmp = []
        for t in episode_labels:
            if t in labels:
                tmp.append(1)
            else:
                tmp.append(0)
        support_ids.append(tmp)
    
    for labels in query_labels:
        tmp = []
        for t in episode_labels:
            if t in labels:
                tmp.append(1)
            else:
                tmp.append(0)
        query_ids.append(tmp)

    return support_ids, query_ids, label2ids

def transfer(text_list, label=None):
    text = ""
    for c in text_list:
        text += (" " + c)
    if label:
        text += "[SEP] "
        text += label
    return text[1:]

def deal_data(batch, labeldict, extenddict): 
    episode = {}
    episode['query_size'] = len(batch)
    episode_labels = []
    extend_labels = {}

    # obtain support set, update labels, add labels to text
    support_text, support_labels = [], []
    t_support_text = []
    for item in batch[0].support_data_items:
        for l in item.label:
            el = extenddict[l]
            l = labeldict[l]
            if l not in episode_labels:
                episode_labels.append(l)
                extend_labels[l] = el
        for each_label in item.label:
            each_label = labeldict[each_label]
            support_labels.append([each_label])
            support_text.append(transfer(item.seq_in, each_label))
            t_support_text.append(' '.join(item.seq_in))
      
    # obtain query set
    query_text, query_labels = [], []
    for item in batch:
        labels = item.test_data_item.label
        new_labels = []
        for each_label in labels:
            new_labels.append(labeldict[each_label])
        query_labels.append(new_labels)
        query_text.append(transfer(item.test_data_item.seq_in))

    support_ids, query_ids, label2ids = deal_label(episode_labels, support_labels, query_labels)
    episode['support_size'] = len(support_text)
    episode['episode_labels'] = episode_labels
    episode['extend_labels'] = extend_labels
    episode['support_text'] = support_text
    episode['t_support_text'] = t_support_text
    episode['support_labels'] = support_ids
    episode['query_text'] = query_text
    episode['query_labels'] = query_ids
    episode['label2ids'] = label2ids
    # print(t_support_text) 
    # print(f'episode_labels:{episode_labels}\nsupport_labels:{support_labels}\nquery_labels:{query_labels}')
    return episode


def train(args, tr_dataloader, model,  optim, lr_scheduler, val_dataloader=None, test_dataloader=None):
    
    if val_dataloader is None:
        acc_best_state = None
        f1_best_state = None 
    test_dataloader = init_dataloader(args, 'test')
    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    train_p, epoch_train_p = [], []
    train_r, epoch_train_r = [], []
    train_f1, epoch_train_f1 = [], []
    train_auc, epoch_train_auc = [], []
    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    val_p, epoch_val_p = [], []
    val_r, epoch_val_r = [], []
    val_f1, epoch_val_f1 = [], []
    val_auc, epoch_val_auc = [], []
    best_p = 0
    best_r = 0
    best_f1 = 0
    best_acc = 0
    best_auc = 0
    loss_fn = Loss_fn(args)
    
    
    p_best_model_path = os.path.join(args.fileModelSave, 'p_best_model.pth')
    r_best_model_path = os.path.join(args.fileModelSave, 'r_best_model.pth')
    f1_best_model_path = os.path.join(args.fileModelSave, 'f1_best_model.pth')
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    auc_best_model_path = os.path.join(args.fileModelSave, 'auc_best_model.pth')

    with open(args.fileLabel, 'r') as src:
        labeldict = json.load(src)
        
    with open(args.extendLabel, 'r') as f:
        extenddict = json.load(f)
    
    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
         
        random.shuffle(tr_dataloader)
        # len(tr_dataloader):200

        for  i, batch in tqdm(enumerate(tr_dataloader[:args.episodeTrain])):
            optim.zero_grad()
            episode = deal_data(batch, labeldict, extenddict)

            model_outputs = model(episode)
            loss, p, r, f, acc, c_acc= loss_fn(model_outputs, episode)
            loss.backward()
            optim.step()
            lr_scheduler.step()
            
            train_loss.append(loss.item())
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f)
            train_acc.append(acc)
            train_auc.append(c_acc)

        avg_loss = np.mean(train_loss[-args.episodeTrain:])
        avg_acc = np.mean(train_acc[-args.episodeTrain:])
        avg_p = np.mean(train_p[-args.episodeTrain:])
        avg_r = np.mean(train_r[-args.episodeTrain:])
        avg_f1 = np.mean(train_f1[-args.episodeTrain:])
        avg_auc = np.mean(train_auc[-args.episodeTrain:])

        print('Avg Train Loss: {}, Avg Train p: {}, Avg Train r: {}, Avg Train f1: {}, Avg Train acc: {}, Avg Train c_acc: {}'.format(avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc))
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r)
        epoch_train_f1.append(avg_f1)
        epoch_train_auc.append(avg_auc)

        if val_dataloader is None:
            continue
        with torch.no_grad():
            model.eval()
            
            for batch in tqdm(val_dataloader):
                episode = deal_data(batch, labeldict, extenddict)
                model_outputs = model(episode)
                loss, p, r, f, acc, c_acc= loss_fn(model_outputs, episode)
                val_loss.append(loss.item())
                val_acc.append(acc)
                val_p.append(p)
                val_r.append(r)
                val_f1.append(f)
                val_auc.append(c_acc)
                
            avg_loss = np.mean(val_loss[-args.episodeVal:])
            avg_acc = np.mean(val_acc[-args.episodeVal:])
            avg_p = np.mean(val_p[-args.episodeVal:])
            avg_r = np.mean(val_r[-args.episodeVal:])
            avg_f1 = np.mean(val_f1[-args.episodeVal:])
            avg_auc = np.mean(val_auc[-args.episodeVal:])

            epoch_val_loss.append(avg_loss)
            epoch_val_acc.append(avg_acc)
            epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r)
            epoch_val_f1.append(avg_f1)
            epoch_val_auc.append(avg_auc)


        postfix = ' (Best)' if avg_p >= best_p else ' (Best: {})'.format(
            best_p)
        r_prefix = ' (Best)' if avg_r >= best_r else ' (Best: {})'.format(
            best_r)
        f1_prefix = ' (Best)' if avg_f1 >= best_f1 else ' (Best: {})'.format(
            best_f1)
        acc_prefix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
       
        print('Avg Val Loss: {}, Avg Val p: {}{}, Avg Val r: {}{}, Avg Val f1: {}{}, Avg Val acc: {}{}, Avg Val c_acc: {}'.format(
            avg_loss, avg_p, postfix, avg_r, r_prefix, avg_f1, f1_prefix, avg_acc, acc_prefix, avg_auc))
   
        if avg_p >= best_p:
            torch.save(model.state_dict(), p_best_model_path)
            best_p = avg_p
            p_best_state = model.state_dict()

        if avg_r >= best_r:
            torch.save(model.state_dict(), r_best_model_path)
            best_r = avg_r
            r_best_state = model.state_dict()

        if avg_f1 >= best_f1:
            torch.save(model.state_dict(), f1_best_model_path)
            best_f1 = avg_f1
            f1_best_state = model.state_dict()

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            acc_best_state = model.state_dict()

    for name in ['epoch_train_loss', 'epoch_train_p', 'epoch_train_r', 'epoch_train_f1', 'epoch_train_acc', 'epoch_train_auc', 'epoch_val_loss', 'epoch_val_p', 'epoch_val_r', 'epoch_val_f1', 'epoch_val_acc', 'epoch_val_auc']:
        save_list_to_file(os.path.join(args.fileModelSave,
                                       name + '.txt'), locals()[name])

    return p_best_state, f1_best_state
        


def test(args, test_dataloader, model):
   
    val_p = []
    val_r = []
    val_loss = []
    val_f1 = []
    val_acc = []
    val_auc = []
    loss_fn = Loss_fn(args)
    with open(args.fileLabel, 'r') as src:
        labeldict = json.load(src)
        
    with open(args.extendLabel, 'r') as f:
        extenddict = json.load(f)

    with torch.no_grad():
        model.eval()
        
        for batch in tqdm(test_dataloader):
            episode = deal_data(batch, labeldict, extenddict)
            model_outputs = model(episode)
            loss, p, r, f, acc, c_acc= loss_fn(model_outputs, episode)
            val_loss.append(loss.item())
            val_acc.append(acc)
            val_p.append(p)
            val_r.append(r)
            val_f1.append(f)
            val_auc.append(c_acc)

        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        avg_p = np.mean(val_p)
        avg_r = np.mean(val_r)
        avg_f1 = np.mean(val_f1)
        avg_auc = np.mean(val_auc)
        


        print('Test p: {}'.format(avg_p))
        print('Test r: {}'.format(avg_r))
        print('Test f1: {}'.format(avg_f1))
        print('Test acc: {}'.format(avg_acc))
        print('Test c_acc: {}'.format(avg_auc))
        print('Test Loss: {}'.format(avg_loss))

        path = args.fileModelSave + "/test_score.json"
        with open(path, "a+") as fout:
            tmp = {"p": avg_p, "r": avg_r, "f1": avg_f1, "acc": avg_acc, "c_acc": avg_auc, "Loss": avg_loss}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))


def write_args_to_json(args):
    path = args.fileModelSave + "/config.json"
    args = vars(args)
    json_str = json.dumps(args, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_str)


def main():
    args = get_parser().parse_args()
    
    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    write_args_to_json(args)
    
    model = init_model(args)
    # print(model)

    

    tr_dataloader = init_dataloader(args, 'train')
    val_dataloader = init_dataloader(args, 'dev')
    test_dataloader = init_dataloader(args, 'test')


    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)
    results = train(args=args,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler,
                    test_dataloader=test_dataloader)
    
    
    print('Testing with last best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(torch.load(args.fileModelSave + "/p_best_model.pth"))
    print('Testing with p best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(torch.load(args.fileModelSave + "/r_best_model.pth"))
    print('Testing with r best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(torch.load(args.fileModelSave + "/f1_best_model.pth"))
    print('Testing with f1 best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)
    
    model.load_state_dict(torch.load(args.fileModelSave + "/acc_best_model.pth"))
    print('Testing with acc best model..')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)
    

   
if __name__ == '__main__':
    main()


