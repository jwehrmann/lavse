
import torch.optim.lr_scheduler


_scheduler = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'exp': torch.optim.lr_scheduler.ExponentialLR,
    'step': torch.optim.lr_scheduler.StepLR,
}

def get_scheduler(name, optimizer, **kwargs):
    return _scheduler[name](optimizer=optimizer, **kwargs)
