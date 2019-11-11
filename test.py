import os
import sys
from pathlib import Path

import torch

import params
from lavse.data.loaders import get_loader
from lavse.model import model
from lavse.train.train import Trainer
from lavse.utils import file_utils, helper
from lavse.utils.logger import create_logger
from run import load_yaml_opts, parse_loader_name
from tqdm import tqdm


if __name__ == '__main__':
    # mp.set_start_method('spawn')

    # loader_name = 'precomp'
    args = params.get_test_params()
    opt = load_yaml_opts(args.options)
    # init_distributed_mode(args)s

    logger = create_logger(
        level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    if 'DATA_PATH' not in os.environ:
        data_path = opt.dataset.data_path
    else:
        data_path = os.environ['DATA_PATH']

    ngpu = torch.cuda.device_count()

    loaders = []
    for data_name in opt.dataset.val.data:
        data_name, lang = parse_loader_name(data_name)

        loader = get_loader(
            data_split=args.data_split,
            data_path=data_path,
            data_name=data_name,
            loader_name=opt.dataset.loader_name,
            local_rank=args.local_rank,
            lang=lang,
            text_repr=opt.dataset.text_repr,
            vocab_paths=opt.dataset.vocab_paths,
            tokenizer_name='default' if 'tokenizer_name' not in opt.dataset else opt.dataset.tokenizer_name,
            cnn=opt.model.params.cnn,
            tokenizer_params=opt.dataset.tokenizer_params,
            ngpu=ngpu,
            **opt.dataset.val,
        )
        loaders.append(loader)

    device = torch.device(args.device)

    loader = loaders[0]
    tokenizers = loader.dataset.tokenizers
    if type(tokenizers) != list:
        tokenizers = [tokenizers]

    model = model.LAVSE(**opt.model, tokenizers=tokenizers)#.to(device)
    checkpoint = helper.restore_checkpoint(
        path=Path(opt.exp.outpath) / 'best_model.pkl',
        model=model,
    )

    model = checkpoint['model']
    logger.info((
        f"Loaded checkpoint. Iteration: {checkpoint['iteration']}, "
        f"rsum: {checkpoint['rsum']}, "
        f"keys: {checkpoint.keys()}"
    ))

    model.set_device('cuda')
    model.to('cuda')

    is_master = True
    model.master = is_master # FIXME: Replace "if print" by built_in print
    print_fn = (lambda x: x) if not is_master else tqdm.write

    trainer = Trainer(
        model=model,
        args={'args': args, 'model_args': opt.model},
        sysoutlog=print_fn,
    )

    result, rs = trainer.evaluate_loaders(
        loaders
    )
    result = {k: float(v) for k, v in result.items()}

    if args.outpath is not None:
        outpath = args.outpath
    else:
        filename = f'{data_name}.{lang}:{args.data_split}:results.json'
        outpath = Path(opt.exp.outpath) / filename

    print('Saving into', outpath)
    file_utils.save_json(
        path=outpath,
        obj=result,
    )
