import argparse
from pathlib import Path

import torch

import profiles
from addict import Dict
from lavse import imgenc, loss, train, txtenc
from lavse.data import get_loader, get_loaders
from lavse.model import LAVSE
from lavse.utils.logger import create_logger
from lavse.utils import helper, file_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
    )
    parser.add_argument(
        '--data_path',
    )
    parser.add_argument(
        '--split',
        default='test',
    )
    parser.add_argument(
        '--val_data', default=['f30k_precomp.en'], nargs='+',
        help=(
            'Data used for evaluation during training.'
            'Eg.: [f30k_precomp.en,m30k_precomp.de]'
        ),
    )
    parser.add_argument(
        '--vocab_path', default='./vocab/complete.json',
        help='Path to saved vocabulary json files.',
    )
    parser.add_argument(
        '--text_repr', default='word', type=str,
        help='Device to ', 
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
        '--outpath',
        help='Path to save logs and models.',
    )
    parser.add_argument(
        '--workers', default=10, type=int,
        help='Number of data loader workers.',
    )
    parser.add_argument(
        '--log_level', default='info',
        choices=['debug', 'info'],
        help='Log/verbosity level.',
    )


    args = parser.parse_args()
    args = Dict(vars(args))

    logger = create_logger(level=args.log_level)

    logger.info(f'Used args: \n{args}')

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
                data_split=args.split,
                lang=lang,
            )
        )

    device = torch.device(args.device)

    checkpoint = helper.restore_checkpoint(args.model_path)
    model_params = checkpoint['args']['model_args']

    model = LAVSE(**model_params).to(device)
    model.load_state_dict(checkpoint['model'])
    print(model)

    trainer = train.Trainer(
        model=model,
        device=device,
        args={'args': args, 'model_args': model_params},
    )

    result, rs = trainer.evaluate_loaders(
        val_loaders
    )
    print(result)

    file_utils.save_json(
        path=args.outpath,
        obj=result,
    )
