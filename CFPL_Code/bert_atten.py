import math
from math import sqrt
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import BertConfig, BertModel, AutoModel, AutoTokenizer
from collections import defaultdict
import copy
import json
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text

class BertEncoder(nn.Module):
    
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        
        student_model_name = args.studentModel
        teacher_model_name = args.teacherModel
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.bert = BertModel.from_pretrained(student_model_name)
        
        self.teacher = AutoModel.from_pretrained(teacher_model_name)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)

        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)
        
        # freeze the parameters of the teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
  

    #(optional:)freeze the parameters of the student model
    def freeze_layers(self, numFreeze):
        print("Freeze Layers: ", numFreeze)
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def forward(self, text):

        tokenizer = self.tokenizer(
        text,
        padding = True,
        truncation = True,
        max_length = 250,
        return_tensors='pt'
        )
        input_ids = tokenizer['input_ids'].cuda()
        token_type_ids = tokenizer['token_type_ids'].cuda()
        attention_mask = tokenizer['attention_mask'].cuda()
        outputs = self.bert(
              input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
              )

        last_hidden_states = outputs[0]
        # we use the mean pool result to represent the whole utterance or label name
        embeddings = last_hidden_states.mean(dim=1)
        return embeddings
    
    # calculate the teacher models's prototypes network
    def teach(self, utterances, labels, episode):
        
        support_labels = torch.tensor(episode['support_labels'])
        # handle the utterances
        tokenizer = self.teacher_tokenizer(
        utterances,
        padding = True,
        truncation = True,
        max_length = 250,
        return_tensors='pt'
        )
        
        input_ids = tokenizer['input_ids'].cuda()
        attention_mask = tokenizer['attention_mask'].cuda()
        outputs = self.teacher(
              input_ids,
              attention_mask=attention_mask,
              output_hidden_states=True
              )
        
        # we use the [CLS] token's last hidden state to represent the whole utterance
        teacher_utterances_hiddens = outputs.last_hidden_state[:, 0, :]
        
        # handle the label names
        tokenizer = self.teacher_tokenizer(
        labels,
        padding = True,
        truncation = True,
        max_length = 250,
        return_tensors='pt'
        )
        
        input_ids = tokenizer['input_ids'].cuda()
        attention_mask = tokenizer['attention_mask'].cuda()
        outputs = self.teacher(
              input_ids,
              attention_mask=attention_mask,
              output_hidden_states=True
              )
        
        teacher_labels_hiddens = outputs.last_hidden_state[:, 0, :]
        
        # we use the [CLS] token's last hidden state to represent the whole label name 
        teacher_cls_hiddens = torch.cat([teacher_utterances_hiddens, teacher_labels_hiddens], dim=-1)

        # calculate prototypes
        support_size = support_labels.shape[0]
        
        prototypes_dict = defaultdict(list)
        for i, each in enumerate(support_labels):
            for j, item in enumerate(each):
                if item == 1:
                    prototypes_dict[j].append(teacher_cls_hiddens[i])
        label_size = support_labels.shape[1]
        prototypes = []
        for i in range(label_size):
            i_proto = torch.stack(prototypes_dict[i]).mean(dim=0)
            prototypes.append(i_proto)
        prototypes = torch.stack(prototypes)
        return prototypes
        

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.bert = BertEncoder(args)
        self.scalar = args.balanced_scalar
        self.k = args.k
        tf.config.set_visible_devices([], 'GPU')
        use_path = 'your multilinguish Universal Sentence Encoder path'
        with tf.device('/cpu:0'):
            self.use_model = hub.load(use_path)
        
        self.thresholder = Thresholder()
        config = BertConfig.from_json_file('config/config.json')
        self.selfattention = TransformerEncoderLayer(config=config)
        self.crossattention = TransformerEncoderLayer(config=config, is_crossattention=True)
        # the hidden dimension of student's cross attention is different from prototypes network's
        config_guidance = copy.deepcopy(config)
        config_guidance.hidden_size = config.hidden_size * 2
        self.guidance_crossattention = TransformerEncoderLayer(config=config_guidance, is_crossattention=True)
        self.train_state_dict = None
        

    def get_logits(self, embeddings, support_label_emb, episode):
        
        # we use prototypes network to generate the similarity between query samples and prototypes
        support_labels = torch.tensor(episode['support_labels'])
        query_labels = torch.tensor(episode['query_labels'])

        query_label_emb = torch.zeros(query_labels.shape[0], support_label_emb.shape[1]).cuda()
        label_emb = torch.cat([support_label_emb, query_label_emb], 0)
        label_emb = label_emb.float()
        
        embeddings = self.selfattention(embeddings)
        logits = self.crossattention(embeddings, label_emb)

        embeddings = torch.cat([embeddings, logits], 1)

        # calculate prototypes
        support_size = support_labels.shape[0]
        query_emb = embeddings[support_size:]
        
        prototypes_dict = defaultdict(list)
        for i, each in enumerate(support_labels):
            for j, item in enumerate(each):
                if item == 1:
                    prototypes_dict[j].append(embeddings[i])
        label_size = support_labels.shape[1]
        prototypes = []
        for i in range(label_size):
            i_proto = torch.stack(prototypes_dict[i]).mean(dim=0)
            prototypes.append(i_proto)
        prototypes = torch.stack(prototypes)
        return prototypes, query_emb
    

    def get_support_labels(self, episode):
        # get labels by ids
        support_labels = episode['support_labels']
        episode_labels = episode['episode_labels']
        extend_labels = episode['extend_labels']
        labels = []
        for each_label in support_labels:
            for i, item in enumerate(each_label):
                if item == 1:
                    labels.append(episode_labels[i])
        return labels
    
    def forward(self, episode):
        # obtain episode support labels list
        support_label_list = self.get_support_labels(episode)
        # print(support_label_list)

        support_text_embedding = self.bert(episode['support_text'])
        query_text_embedding = self.bert(episode['query_text'])

        if self.k != 0:
            extend_support_label_list = [episode['extend_labels'][i] for i in support_label_list]
            support_text = episode['support_text']
            extend_list = []
            for i, text in enumerate(support_text):
                member = extend_support_label_list[i]
                tmp = []
                for j in member:
                    embs = self.use_model([text, j]).numpy()
                    norm = np.linalg.norm(embs, axis=1)
                    embs = embs / norm[:, None]
                    bs = (embs[:1] * embs[1:]).sum(axis=1)[0]
                    tmp.append(bs)
                extend_list.append(tmp)
            extend_score = torch.tensor(extend_list).cuda()
            _, extend_k = torch.topk(extend_score, self.k, dim=-1)
            extend_k = extend_k.cpu().tolist()
            extend_support_label_list = [[extend_support_label_list[i][j] for j in sublist] for i, sublist in enumerate(extend_k)]
            extend_support_label_list = [' [SEP] '.join(i) for i in extend_support_label_list]
            extend_embedding = self.bert(extend_support_label_list)
            support_text_embedding = self.scalar * support_text_embedding + (1 - self.scalar) * extend_embedding
        
        text_embedding = torch.cat([support_text_embedding, query_text_embedding], dim=0)

        teacher_prototypes = self.bert.teach(episode['support_text'], support_label_list, episode)
        
        support_label_emb  = self.bert(support_label_list)
        
        prototypes, query_emb = self.get_logits(text_embedding, support_label_emb, episode)
        
        guidance_prototypes = self.guidance_crossattention(prototypes, teacher_prototypes)
        
        prototypes = self.args.alpha * prototypes + (1 - self.args.alpha) * guidance_prototypes
        
        thresholds = self.thresholder(prototypes)
        
        return prototypes, query_emb, thresholds


def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(0, 1)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, is_crossattention = False):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.is_crossattention = is_crossattention

    def forward(self, hidden_state, encoder_hidden_state=None):
        if not self.is_crossattention:
            attn_outputs = scaled_dot_product_attention(
                self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        else:
            attn_outputs = scaled_dot_product_attention(
                self.q(hidden_state), self.k(encoder_hidden_state), self.v(encoder_hidden_state)
            )
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_crossattention = False):
        super().__init__()
        self.is_crossattention = is_crossattention
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim, is_crossattention=is_crossattention) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state, y=None):
        if self.is_crossattention:
            x = torch.cat([h(hidden_state, y) for h in self.heads], dim=-1)
        else:
            x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, is_crossattention = False):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.is_crossattention = is_crossattention
        if self.is_crossattention:
            self.crossattention = MultiHeadAttention(config=config, is_crossattention=is_crossattention)

    def forward(self, x, y=None):
        if self.is_crossattention:
            # Apply layer normalization and then copy input into query, key, value
            hidden_state = self.layer_norm_1(y)
            y = y + self.attention(hidden_state)
            # Apply attention with a skip connection
            # Apply feed-forward layer with a skip connection
        
            hidden_state = self.layer_norm_2(y)
            y = y + self.crossattention(x, hidden_state)
            y = y + self.feed_forward(self.layer_norm_3(y))
            return y
        else:
            # Apply layer normalization and then copy input into query, key, value
            hidden_state = self.layer_norm_1(x)
            x = x + self.attention(hidden_state)
            # Apply attention with a skip connection
            # Apply feed-forward layer with a skip connection
            x = x + self.feed_forward(self.layer_norm_2(x))
            return x

class Thresholder(nn.Module):
    def __init__(self):
        super(Thresholder, self).__init__()
        self.linear1 = nn.Linear(1536, 300)
        # self.linear1 = nn.Linear(512, 300)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, prototypes):
        output = self.linear1(prototypes)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output

        