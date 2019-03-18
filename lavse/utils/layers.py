import torch 
import torch.nn as nn


def default_initializer(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)    
    elif type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)
        

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def tensor_to_numpy(x):
    return x.data.cpu().numpy()