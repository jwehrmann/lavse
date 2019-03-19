import argparse 
from lavse.data import get_loaders, get_loader
from pathlib import Path
from lavse.utils.logger import create_logger
from lavse.model import LAVSE
from addict import Dict
from lavse import loss
from lavse import train
from lavse import txtenc, imgenc

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
    )
    parser.add_argument(
        '--train_data', default='f30k_precomp.en',
        help=(
            'Data used to align images and captions.'
            'Eg.: f30k_precomp.en'
        ),
    )
    parser.add_argument(
        '--val_data', default=['f30k_precomp.en'], nargs='+',
        help=(
            'Data used for evaluation during training.'
            'Eg.: [f30k_precomp.en,m30k_precomp.de]'
        ),
    )
    parser.add_argument(
        '--adapt_data', default=None, nargs='+',
        help=(
            'Data used for training joint language space.'
            'Eg.: [m30k_precomp.en-de,jap_precomp.en-jt]'
        ),
    )
    parser.add_argument(
        '--vocab_path', default='./vocab/complete.json',
        help='Path to saved vocabulary json files.',
    )
    parser.add_argument(
        '--margin', default=0.2, type=float,
        help='Rank loss margin.',
    )
    parser.add_argument(
        '--num_epochs', default=30, type=int,
        help='Number of training epochs.',
    )
    parser.add_argument(
        '--device', default='cuda:0', type=str,
        help='Device to run the model.',
    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='Size of a training mini-batch.',
    )
    parser.add_argument(
        '--embed_dim', default=300, type=int,
        help='Dimensionality of the word embedding.',
    )
    parser.add_argument(
        '--latent_size', default=1024, type=int,
        help='Dimensionality of the joint embedding.',
    )
    parser.add_argument(
        '--grad_clip', default=2., type=float,
        help='Gradient clipping threshold.',
    )
    parser.add_argument(
        '--outpath',
        help='Path to save logs and models.',
    )
    parser.add_argument(
        '--text_encoder', default='gru',
        choices=txtenc.get_available_txtenc(),
        help='Path to save logs and models.',
    )
    parser.add_argument(
        '--text_pooling', default='lens',
        choices=['mean', 'max', 'lens'],
        help='Path to save logs and models.',
    )
    parser.add_argument(
        '--image_encoder', default='scan',
        choices=imgenc.get_available_imgenc(),
        help='Path to save logs and models.',
    )
    parser.add_argument(
        '--text_repr', 
        default='word',
        help='Path to save logs and models.',
    )
    parser.add_argument(
        '--lr', default=.0002, type=float,
        help='Initial learning rate.',
    )
    parser.add_argument(
        '--lr_decay_interval', default=15, type=int,
        help='Number of epochs to update the learning rate.',
    )
    parser.add_argument(
        '--lr_decay_rate', default=0.1, type=int,
        help='Number of epochs to update the learning rate.',
    )
    parser.add_argument(
        '--workers', default=10, type=int,
        help='Number of data loader workers.',
    )
    parser.add_argument(
        '--log_step', default=10, type=int,
        help='Number of steps to print and record the log.',
    )
    parser.add_argument(
        '--val_step', default=500, type=int,
        help='Number of steps to run validation.',
    )
    parser.add_argument(
        '--max_violation', action='store_true',
        help='Use max instead of sum in the rank loss (i.e., k=1)',
    )
    parser.add_argument(
        '--increase_k', default=.4, type=float,
        help='Rate for linear increase of k hyper-parameter (used when not --max_violation). ',
    )
    parser.add_argument(
        '--initial_k', default=1., type=float,
        help='Initial value for k hyper-parameter (used when not --max_violation)',
    )
    parser.add_argument(
        '--log_level', default='info',
        choices=['debug', 'info'],
        help='Log/verbosity level.',
    )
    args = parser.parse_args()
    
    logger = create_logger(level=args.log_level)

    train_data = args.train_data
    data_name, lang = train_data.split('.')
    train_loader = get_loader(
        data_path=args.data_path,
        data_name=data_name, 
        loader_name='precomp',
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        workers=args.workers,
        text_repr=args.text_repr,
        data_split='train', 
        lang=lang,
    )

    val_loaders = []
    for val_data in args.val_data:
        data_name, lang = val_data.split('.')
        val_loaders.append(
            get_loader(
                data_path=args.data_path, 
                data_name=data_name,
                loader_name='precomp',
                vocab_path=args.vocab_path,
                batch_size=args.batch_size,
                workers=args.workers,
                text_repr=args.text_repr,
                data_split='dev',
                lang=lang,
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
                )
            )

    device = torch.device(args.device)

    model_params = Dict(
        imgenc_name=args.image_encoder,
        txtenc_name=args.text_encoder,
        latent_size=args.latent_size,
        img_dim=train_loader.dataset.get_img_dim(),
        num_embeddings=train_loader.dataset.tokenizer.get_nb_words(),
        embed_dim=args.embed_dim,
        txt_pooling=args.text_pooling,
    )

    model = LAVSE(**model_params).to(device)
    print(model)

    trainer = train.Trainer(
        model=model,
        device=device,
        args={'args': args, 'model_args': model_params},
    )

    multimodal_criterion = loss.ContrastiveLoss(
        device=device,
        margin=args.margin,
        max_violation=args.max_violation,
        weight=1., 
        initial_k=args.initial_k, 
        increase_k=args.increase_k,
    )

    multilanguage_criterion = loss.ContrastiveLoss(
        device=device,
        margin=args.margin,
        max_violation=args.max_violation,
        weight=1., 
        initial_k=args.initial_k, 
    )

    trainer.setup_optim(
        lr=args.lr, 
        mm_criterion=multimodal_criterion, 
        ml_criterion=multilanguage_criterion,
        lr_decay_rate=args.lr_decay_rate, 
        lr_decay_interval=args.lr_decay_interval,
    )

    result, rs = trainer.evaluate_loaders(
        val_loaders
    )

    trainer.fit(
        train_loader=train_loader, 
        valid_loaders=val_loaders, 
        lang_loaders=adapt_loaders,
        nb_epochs=45, 
        early_stop=5,
    )

