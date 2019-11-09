import argparse
from pathlib import Path

import torch
from addict import Dict
from tqdm import tqdm

import params
from lavse.data.loaders import get_loader, get_loaders
from lavse.model import imgenc, loss, model
from lavse.model import txtenc, data_parallel
from lavse.train import train
from lavse.utils.logger import create_logger
from lavse.utils import helper
from lavse.utils.file_utils import load_yaml_opts, parse_loader_name
# from lavse.utils import options
import yaml
import os

import torch.multiprocessing as mp

import random

torch.manual_seed(0)
random.seed(0, version=2)


def init_distributed_mode(opt):
    opt.distributed = True

    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=opt.ngpu,
        rank=opt.local_rank,
    )
    setup_for_distributed(opt.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == '__main__':
    # mp.set_start_method('spawn')

    # loader_name = 'precomp'
    from lavse.utils import options

    # args = params.get_train_params()
    # opt = load_yaml_opts(args.options)
    opt = options.Options()
    opt = Dict(vars(opt)).options

    if opt.misc.distributed:
        init_distributed_mode(opt)

    logger = create_logger(
        level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used options: \n{opt}')

    # torch.cuda.set_device(args.local_rank)

    # loader_name = args.loader_name

    # if args.local_rank != 0:
    #     logger.propagate = False

    train_data = opt.dataset.train

    if 'DATA_PATH' not in os.environ:
        data_path = opt.dataset.data_path
    else:
        data_path = os.environ['DATA_PATH']

    ngpu = opt.ngpu

    data_name, lang = parse_loader_name(opt.dataset.train.data)
    train_loader = get_loader(
        data_split='train',
        data_path=data_path,
        data_name=data_name,
        loader_name=opt.dataset.loader_name,
        local_rank=opt.local_rank,
        lang=lang,
        text_repr=opt.dataset.text_repr,
        vocab_paths=opt.dataset.vocab_paths,
        tokenizer_name='default' if 'tokenizer_name' not in opt.dataset else opt.dataset.tokenizer_name,
        ngpu=ngpu,
        cnn=opt.model.params.cnn,
        tokenizer_params=opt.dataset.tokenizer_params,
        **opt.dataset.train,
        # vocab_path=args.vocab_path,
        # batch_size=args.batch_size,
        # workers=args.workers,
        # text_repr=args.text_repr,
    )

    val_loaders = []
    for val_data in opt.dataset.val.data:
        data_name, lang = parse_loader_name(val_data)
        val_loaders.append(
            get_loader(
                data_split='dev',
                data_path=data_path,
                data_name=data_name,
                loader_name=opt.dataset.loader_name,
                local_rank=opt.local_rank,
                lang=lang,
                text_repr=opt.dataset.text_repr,
                tokenizer_name='default' if 'tokenizer_name' not in opt.dataset else opt.dataset.tokenizer_name,
                vocab_paths=opt.dataset.vocab_paths,
                ngpu=1,
                cnn=opt.model.params.cnn,
                tokenizer_params=opt.dataset.tokenizer_params,
                **opt.dataset.val,
            )
        )

    assert len(val_loaders) > 0

    adapt_loaders = []
    for adapt_data in opt.dataset.adapt.data:
        data_name, lang = parse_loader_name(adapt_data)
        adapt_loaders.append(
            get_loader(
                data_split='train',
                data_path=data_path,
                data_name=data_name,
                loader_name='lang',
                local_rank=opt.local_rank,
                lang=lang,
                text_repr=opt.dataset.text_repr,
                vocab_paths=opt.dataset.vocab_paths,
                ngpu=1,
                tokenizer_params=opt.dataset.tokenizer_params,
                **opt.dataset.adapt,
            )
        )

    logger.info(f'Adapt loaders: {len(adapt_loaders)}')

    tokenizers = train_loader.dataset.tokenizers
    if type(tokenizers) != list:
        tokenizers = [tokenizers]

    model = model.LAVSE(**opt.model, tokenizers=tokenizers)#.to(device)
    if opt.misc.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(model)

    if opt.exp.resume is not None:
        logger.info(f'Resuming checkpoint: {opt.resume}')
        checkpoint = helper.restore_checkpoint(
            path=opt.resume,
            model=model,
        )
        model = checkpoint['model']
        logger.info((
            f"Loaded checkpoint. Iteration: {checkpoint['iteration']}, "
            f"rsum: {checkpoint['rsum']}, "
            f"keys: {checkpoint.keys()}"
        ))

    # Distributed data parallel training
    if opt.misc.distributed:
        device = torch.device('cuda:{}'.format(opt.local_rank))
        model = model.to(device)
        model = data_parallel.DistributedDataParallel(
               model, device_ids=[opt.local_rank],
               output_device=opt.local_rank,
        )
        model.set_device(device)
        # model = data_parallel.DistributedDataParallel(model)
    # Standard Data parallel + Single gpu
    else:
        device = torch.device('cuda')
        nb_devices = torch.cuda.device_count()
        if nb_devices > 1:
            logger.info(f'Found {nb_devices} devices. Using DataParallel.')
            # model.img_enc = data_parallel.DataParallel(model.img_enc)
            # model.txt_enc = data_parallel.DataParallel(model.txt_enc)
            model.set_device(device)
        elif nb_devices == 0:
            device = torch.device('cpu')
        print(device)
        model = model.to(device)

    is_master = True
    model.master = is_master # FIXME: Replace "if print" by built_in print
    print_fn = (lambda x: x) if not is_master else tqdm.write

    trainer = train.Trainer(
        model=model,
        args=opt,
        sysoutlog=print_fn,
        device=device
    )

    trainer.setup_optim(
        lr=opt.optimizer.lr,
        lr_scheduler=opt.optimizer.lr_scheduler,
        clip_grad=opt.optimizer.grad_clip,
        log_grad_norm=False,
        log_histograms=False,
        optimizer=opt.optimizer,
        freeze_modules=opt.model.freeze_modules,
        early_stop=opt.engine.early_stop,
        save_all=opt.engine.save_all,
        val_metric=opt.engine.val_metric if opt.engine.val_metric else 'rsum'
    )

    if opt.engine.eval_before_training:
        result, rs = trainer.evaluate_loaders(
            val_loaders
        )

    trainer.fit(
        train_loader=train_loader,
        valid_loaders=val_loaders,
        lang_loaders=adapt_loaders,
        nb_epochs=opt.engine.nb_epochs,
        path=opt.exp.outpath,
        valid_interval=opt.engine.valid_interval,
        log_interval=opt.engine.print_freq,
        world_size=1 # TODO,
    )
