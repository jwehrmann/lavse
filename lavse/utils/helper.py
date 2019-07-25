import os
import torch
from tensorboardX import SummaryWriter


def save_checkpoint(
        outpath, model, optimizer=None,
        is_best=False, save_all=False, **kwargs
    ):

    if hasattr(model, 'module'):
        model = model.module

    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    state_dict.update(**kwargs)

    if not save_all:
        epoch = -1

    torch.save(
        obj=state_dict,
        f=os.path.join(outpath, f'checkpoint_{epoch}.pkl'),
    )

    if is_best:
        import shutil
        shutil.copy(
            os.path.join(outpath, f'checkpoint_{epoch}.pkl'),
            os.path.join(outpath, 'best_model.pkl'),
        )


def restore_checkpoint(path, model=None, optimizer=False):
    state_dict = torch.load(
        path,  map_location=lambda storage, loc: storage
    )
    new_state = {}
    for k, v in state_dict['model'].items():
        new_state[k.replace('module.', '')] = v

    if model is None:
        from ..model.model import LAVSE
        model_params = state_dict['args']['model_args']
        model = LAVSE(**model_params)

    model.load_state_dict(new_state)
    state_dict['model'] = model

    if optimizer:
        optimizer = state_dict['optimizer']
        state_dict['optimizer'] = None

    return state_dict


# def adjust_learning_rate(
#     optimizer, epoch, initial_lr,
#     interval=1, decay=0.
# ):

#     lr = initial_lr * (decay ** (epoch // interval))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#         if 'name' in param_group:
#             param_group['lr'] = lr

#     return lr


def get_tb_writer(logger_path):

    if logger_path == 'runs/':
        tb_writer = SummaryWriter()
        logger_path = tb_writer.file_writer.get_logdir()
    else:
        tb_writer = SummaryWriter(logger_path)

    return tb_writer


def get_device(gpu_id):

    if gpu_id >= 0:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')

    return device


def reset_pbar(pbar):
    from time import time
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.start_t = time()
    pbar.last_print_t = time()
    pbar.update()
    return pbar


def print_tensor_dict(tensor_dict, print_fn):
    line = []
    for k, v in sorted(tensor_dict.items()):
        try:
            v = v.item()
        except AttributeError:
            pass
        line.append(f'{k.title()}: {v:10.6f}')
    print_fn(', '.join(line))
