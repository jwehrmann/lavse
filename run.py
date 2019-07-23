import argparse
from pathlib import Path

import torch
from addict import Dict
from tqdm import tqdm

import params
import profiles
from lavse.data.loaders import get_loader, get_loaders
from lavse.model import imgenc, loss, model
from lavse.model import txtenc, data_parallel
from lavse.train import train
from lavse.utils.logger import create_logger
from lavse.utils import helper
# from lavse.utils import options
import yaml
import os

import torch.multiprocessing as mp


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


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


def load_yaml_options(path):
    result = {}
    with open(path, 'r') as yaml_file:
        options_yaml = Dict(yaml.safe_load(yaml_file))
    return options_yaml


def parse_loader_name(data_name):
    if '.' in data_name:
        name, lang = data_name.split('.')
        return name, lang
    else:
        return data_name, None


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # loader_name = 'precomp'
    args = params.get_train_params()
    opt = load_yaml_options(args.options)
    # init_distributed_mode(args)s

    logger = create_logger(
        level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    # torch.cuda.set_device(args.local_rank)

    # loader_name = args.loader_name

    # if args.local_rank != 0:
    #     logger.propagate = False

    train_data = opt.dataset.train_data

    if 'DATA_PATH' not in os.environ:
        data_path = opt.dataset.data_path
    else:
        data_path = os.environ['DATA_PATH']

    ngpu = torch.cuda.device_count()

    data_name, lang = parse_loader_name(opt.dataset.train.data)
    train_loader = get_loader(
        data_split='train',
        data_path=data_path,
        data_name=data_name,
        loader_name=opt.dataset.loader_name,
        local_rank=args.local_rank,
        lang=lang,
        text_repr=opt.dataset.text_repr,
        vocab_path=opt.dataset.vocab_path,
        ngpu=ngpu,
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
                local_rank=args.local_rank,
                lang=lang,
                text_repr=opt.dataset.text_repr,
                vocab_path=opt.dataset.vocab_path,
                ngpu=1,
                **opt.dataset.val,
            )
        )

    assert len(val_loaders) > 0

    adapt_loaders = []
    if args.adapt_data is not None:
        for adapt_data in args.adapt_data:
            data_name, lang = adapt_data.split('.')
            adapt_loaders.append(
                get_loader(
                    data_path=args.data_path,
                    data_name=data_name,
                    loader_name='lang',
                    vocab_path=args.vocab_path,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    text_repr=args.text_repr,
                    data_split='train',
                    lang=lang,
                    ngpu=args.ngpu,
                    local_rank=args.local_rank,
                )
            )

    # if args.device != 'cpu':
    #     device = torch.device('cuda:{}'.format(args.local_rank))

    # model_params = Dict(
    #     imgenc_name=args.image_encoder,
    #     txtenc_name=args.text_encoder,
    #     latent_size=args.latent_size,
    #     num_embeddings=len(train_loader.dataset.tokenizer),
    #     embed_dim=args.embed_dim,
    #     txt_pooling=args.text_pooling,
    #     img_pooling=args.image_pooling,
    #     similarity_name=args.sim,
    # )


    tokenizer = train_loader.dataset.tokenizer
    model = model.LAVSE(**opt.model, tokenizer=tokenizer)#.to(device)
    logger.info(model)

    if opt.exp.resume is not None:
        logger.info(f'Resuming checkpoint: {args.resume}')
        checkpoint = helper.restore_checkpoint(
            path=args.resume,
            model=model,
        )
        model = checkpoint['model']
        logger.info((
            f"Loaded checkpoint. Iteration: {checkpoint['iteration']}, "
            f"rsum: {checkpoint['rsum']}, "
            f"keys: {checkpoint.keys()}"
        ))

    model.set_devices_(
        txt_devices=opt.model.txt_enc.devices,
        img_devices=opt.model.img_enc.devices,
        loss_device=opt.model.similarity.device,
    )

    # Distribute the same process in GPUs
    # This is used when a single model cannot fit the memory
    # if args.data_parallel:
    #     import torch.nn as nn
    #     model.img_enc = nn.DataParallel(model.img_enc, device_ids=[0,1,2], output_device=0)
    is_master = True
    model.master = is_master # FIXME: Replace "if print" by built_in print
    print_fn = (lambda x: x) if not is_master else tqdm.write

    trainer = train.Trainer(
        model=model,
        args={'args': args, 'model_args': opt.model},
        sysoutlog=print_fn,
    )

    multimodal_criterion = loss.ContrastiveLoss(
        **opt.criterion
    )

    multilanguage_criterion = loss.ContrastiveLoss(
        **opt.criterion
    )

    # TODO: improve
    model.mm_criterion = multimodal_criterion

    trainer.setup_optim(
        lr=opt.optimizer.lr,
        lr_scheduler=opt.optimizer.lr_scheduler,
        clip_grad=opt.optimizer.grad_clip,
        mm_criterion=multimodal_criterion,
        ml_criterion=multilanguage_criterion,
        log_grad_norm=False,
        log_histograms=False,
        optimizer=opt.optimizer,
        freeze_modules=opt.model.freeze_modules
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
        world_size=1 # TODO
    )
