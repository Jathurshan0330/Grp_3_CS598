import os
import random
import numpy as np
import torch
import math



def seed_everything(seed=5):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def class_wise_f1_score(output, label, n_classes):

    preds = torch.argmax(output, 1)

    conf_matrix = torch.zeros(n_classes, n_classes)
    F1_list = []

    for p, t in zip(preds, label):
        if torch.is_tensor(p):
          p = p.item()
          t = int(t.item())
        conf_matrix[p, t] += 1
    

    TP = conf_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()



        if ((2*TP[c]) + (FN + FP)) !=0:
            F1_score = (2*TP[c])/((2*TP[c]) + (FN + FP))
        else:
            F1_score = 0
        
    
        F1_list.append(float(F1_score))
        
    return F1_list



def read_hyp_params(hyp_path):
    with open(hyp_path, 'r') as f:
        hyp = f.readlines()
    hyp = [i.strip() for i in hyp]
    hyp = [i.split(':') for i in hyp]
    hyp = {i[0].strip():i[1].strip() for i in hyp}
    
    for key in hyp.keys():
        if hyp[key].lower() == 'true':
            hyp[key] = True
        elif hyp[key].lower() == 'false':
            hyp[key] = False
        else:
            try:
                hyp[key] = int(hyp[key])
            except:
                try:
                    hyp[key] = float(hyp[key])
                except:
                    try:
                        hyp[key] = eval(hyp[key])
                    except:
                        continue 
    # convert the dict to a object
    class Hyp():
        def __init__(self, **entries):
            self.__dict__.update(entries)
    hyp = Hyp(**hyp)                    
    return hyp
        
        

def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()
        
        
# define focal loss on binary classification for CHB-MIT
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()