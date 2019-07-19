import argparse
from pathlib import Path

import torch
from addict import Dict
from tqdm import tqdm

import params
import profiles
from lavse.data.loaders import get_loader, get_loaders
from lavse.model import imgenc, loss, model, txtenc
from lavse.train import train
from lavse.utils.logger import create_logger
from lavse.utils import helper
import torch.multiprocessing as mp


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # loader_name = 'precomp'
    args = params.get_train_params()

    torch.cuda.set_device(args.local_rank)

    loader_name = args.loader_name

    logger = create_logger(level=args.log_level)
    if args.local_rank != 0:
        logger.propagate = False

    if args.profile is not None:
        profile_args = profiles.get_profile(args.profile)
        args.update(profile_args)
        logger.info(f'Using profile {args.profile}')

    logger.info(f'Used args: \n{args}')

    train_data = args.train_data
    data_name, lang = train_data.split('.')
    train_loader = get_loader(
        data_path=args.data_path,
        data_name=data_name,
        loader_name=loader_name,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        workers=args.workers,
        text_repr=args.text_repr,
        data_split='train',
        lang=lang,
        ngpu=args.ngpu,
        local_rank=args.local_rank,
    )

    val_loaders = []
    for val_data in args.val_data:
        data_name, lang = val_data.split('.')
        val_loaders.append(
            get_loader(
                data_path=args.data_path,
                data_name=data_name,
                loader_name=loader_name,
                vocab_path=args.vocab_path,
                batch_size=args.batch_size//4,
                workers=args.workers,
                text_repr=args.text_repr,
                data_split='dev',
                lang=lang,
                ngpu=1,
                local_rank=args.local_rank,
            )
        )

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

    model_params = Dict(
        imgenc_name=args.image_encoder,
        txtenc_name=args.text_encoder,
        latent_size=args.latent_size,
        num_embeddings=len(train_loader.dataset.tokenizer),
        embed_dim=args.embed_dim,
        txt_pooling=args.text_pooling,
        img_pooling=args.image_pooling,
        similarity_name=args.sim,
        loss_device='cuda'
    )

    model = model.LAVSE(**model_params)#.to(device)
    logger.info(model)

    is_master = (args.local_rank == 0)

    # Distribute across distinct process on various GPUS
    # This is used when a single model can fit the memory
    # and the user wants to speed up the training
    world_size = args.ngpu
    if world_size > 1:
        model = model.to('cuda')
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=world_size,
            rank=args.local_rank,
        )
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
        model.module.set_master_(is_master)
        model.master = is_master
        model.module.set_devices_(['cuda'], ['cuda'], 'cuda')
    else:
        model.set_master_(is_master)
        model.set_devices_(['cuda'], ['cuda'], 'cuda')

    if args.resume is not None:
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


    if args.data_parallel:
        model.set_devices_(
            ['cuda:3'], ['cuda:0','cuda:1','cuda:2'], 'cuda:3'
        )

    # Distribute the same process in GPUs
    # This is used when a single model cannot fit the memory
    # if args.data_parallel:
    #     import torch.nn as nn
    #     model.img_enc = nn.DataParallel(model.img_enc, device_ids=[0,1,2], output_device=0)

    if hasattr(model, 'module'):
        model.get_sim_matrix = model.module.get_sim_matrix
        model.get_sim_matrix_shared = model.module.get_sim_matrix_shared
        model.master = is_master

    print_fn = (lambda x: x) if not is_master else tqdm.write

    trainer = train.Trainer(
        device=torch.device('cuda'),
        model=model,
        args={'args': args, 'model_args': model_params},
        sysoutlog=print_fn,
        master=is_master,
    )

    multimodal_criterion = loss.ContrastiveLoss(
        margin=args.margin,
        max_violation=args.max_violation,
        weight=1.,
        beta=args.beta,
    )

    multilanguage_criterion = loss.ContrastiveLoss(
        margin=args.margin,
        max_violation=args.max_violation,
        weight=1.,
        beta=args.beta,
    )

    # TODO: improve
    model.mm_criterion = multimodal_criterion

    trainer.setup_optim(
        lr=args.lr,
        mm_criterion=multimodal_criterion,
        ml_criterion=multilanguage_criterion,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_interval=args.lr_decay_interval,
        cnn_lr_factor=1.,
        clip_grad=args.grad_clip,
        early_stop=args.early_stop,
        log_grad_norm=False,
        log_histograms=False,
        save_all=args.save_all,
        finetune_convnet=args.finetune,
        optimizer=torch.optim.Adam,
    )

    if args.eval_before_training:
        result, rs = trainer.evaluate_loaders(
            val_loaders
        )

    trainer.fit(
        train_loader=train_loader,
        # train_loader=val_loaders[0],
        valid_loaders=val_loaders,
        lang_loaders=adapt_loaders,
        nb_epochs=args.nb_epochs,
        path=args.outpath,
        valid_interval=args.valid_interval,
        world_size=world_size
    )
