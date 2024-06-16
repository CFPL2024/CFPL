import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Loss_fn(torch.nn.Module):
    def __init__(self, args):
        super(Loss_fn, self).__init__()
        self.args = args
        self.max_count = args.numCount
        self.loss_fn = CrossEntropyLoss()
        self.lamda = args.lamda
        self.beta = args.beta
        self.circle_loss = Circle_Loss(args.temperature)
        print("Lamda: ", self.lamda)
        
    def count(self, label_ids):
        counts = []
        for label in label_ids:
            t = 0
            for l in label:
                if l == 1:
                    t += 1
            counts.append(t)
        return counts

    def forward(self, model_outputs, episode):
        # prototypes: C, 768; query_emb: Q, 768

        prototypes, query_emb, thresholds = model_outputs

        query_labels = episode['query_labels']
        support_labels = episode['support_labels']
        
        query_counts = self.count(query_labels)
        
        query_counts = torch.tensor(query_counts).cuda()
       
        # multi-label 
        query_labels = torch.tensor(episode['query_labels']).float().cuda()
        dists = euclidean_dist(query_emb, prototypes)
        p_y = F.softmax(-dists + 1e-5, dim=1) #avoid nan
        thresholds = thresholds.view(-1)
        log_p_y = F.log_softmax(-dists, dim=1) # num_query x num_class
        loss = - query_labels * log_p_y
        loss = loss.mean() 

        circle_loss = self.circle_loss(p_y, query_labels, thresholds)
        loss = loss + self.lamda * circle_loss

        y_pred = (p_y - thresholds >= 0).long()
        count_pred = torch.sum(y_pred, dim=-1, keepdim=True).long()
        query_counts = query_counts.cpu().detach()
        count_pred = count_pred.cpu().detach()
        c_acc = accuracy_score(query_counts, count_pred) #
        
        target_mode = 'micro'

        query_labels = query_labels.cpu().detach()
        y_pred = y_pred.cpu().detach()
        p = precision_score(query_labels, y_pred, average=target_mode, zero_division=0)
        r = recall_score(query_labels, y_pred, average=target_mode, zero_division=0)
        f = f1_score(query_labels, y_pred, average=target_mode, zero_division=0)
        acc = accuracy_score(query_labels, y_pred)

        return loss, p, r, f, acc, c_acc
       

class Circle_Loss(torch.nn.Module):
    def __init__(self, temperature=0.03):
        super(Circle_Loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, logits, labels, thresholds):
        delta_thresholds = logits - thresholds
        positive = labels * logits
        negative = (1 - labels) * logits
        loss = torch.log(torch.exp(self.temperature * thresholds) 
                         + torch.sum(torch.exp(self.temperature * negative), dim=0) + 1e-10)\
               + torch.log(torch.exp(-self.temperature * thresholds)
                         + torch.sum(torch.exp(-self.temperature * positive), dim=0) + 1e-10)
        loss = loss.mean()
        
        return loss
