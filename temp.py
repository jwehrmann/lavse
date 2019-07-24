import torch
from torch.optim import lr_scheduler
import numpy as np

def update_lr(curr_lr, gamma=0.25):
    return curr_lr * gamma

iters = [16000, 80000, 10000]
it = range(*iters)
print(it)

gamma = 0.25
curr_lr = 0.002
lrs = {}
for i in range(22*4000):
    if i not in it:
        continue
    curr_lr = update_lr(curr_lr, gamma=gamma)
    lrs[curr_lr] = i / 4000


epochs = np.array(it) / 4000.
print(epochs)
print(lrs)
