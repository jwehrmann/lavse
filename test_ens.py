import argparse
import os
import sys
from pathlib import Path

import torch

import params
from addict import Dict
from lavse.data.loaders import get_loader
from lavse.model import model
from lavse.train import evaluation
from lavse.train.train import Trainer
from lavse.utils import file_utils, helper
from lavse.utils.logger import create_logger
from run import load_yaml_opts, parse_loader_name
from tqdm import tqdm


def get_test_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'options',
    )
    parser.add_argument(
        'options2',
    )
    parser.add_argument(
        '--device', default='cuda',
    )
    parser.add_argument(
        '--data_split', '-s', default='dev'
    )
    parser.add_argument(
        '--outpath', '-o', default=None
    )

    args = parser.parse_args()
    args = Dict(vars(args))
    return args


def load_model(opt, tokenizers):

    _model = model.LAVSE(**opt.model, tokenizers=tokenizers)#.to(device)

    # import pickle
    # with open(Path(opt.exp.outpath) / 'best_model.pkl', 'rb', 0) as f:
    #     pickle.load(f)

    print(_model)
    checkpoint = helper.restore_checkpoint(
        path=Path(opt.exp.outpath) / 'best_model.pkl',
        model=_model,
    )

    _model = checkpoint['model']
    logger.info((
        f"Loaded checkpoint. Iteration: {checkpoint['iteration']}, "
        f"rsum: {checkpoint['rsum']}, "
        f"keys: {checkpoint.keys()}"
    ))

    _model.set_devices_(
        txt_devices=opt.model.txt_enc.devices,
        img_devices=opt.model.img_enc.devices,
        loss_device=opt.model.similarity.device,
    )
    _model.master = True
    return _model

if __name__ == '__main__':
    # mp.set_start_method('spawn')

    # loader_name = 'precomp'
    args = get_test_params()
    opt = load_yaml_opts(args.options)
    opt2 = load_yaml_opts(args.options2)
    # init_distributed_mode(args)s

    logger = create_logger(
        level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    train_data = opt.dataset.train_data

    if 'DATA_PATH' not in os.environ:
        data_path = opt.dataset.data_path
    else:
        data_path = os.environ['DATA_PATH']

    ngpu = torch.cuda.device_count()

    data_name, lang = parse_loader_name(opt.dataset.train.data)

    loader = get_loader(
        data_split=args.data_split,
        data_path=data_path,
        data_name=data_name,
        loader_name=opt.dataset.loader_name,
        local_rank=args.local_rank,
        lang=lang,
        text_repr=opt.dataset.text_repr,
        vocab_paths=opt.dataset.vocab_paths,
        ngpu=ngpu,
        **opt.dataset.val,
    )

    device = torch.device(args.device)

    tokenizers = loader.dataset.tokenizers
    if type(tokenizers) != list:
        tokenizers = [tokenizers]

    model_a = load_model(opt, tokenizers)
    model_b = load_model(opt2, tokenizers)

    device = torch.device('cuda')

    img_emb, cap_emb, lens = evaluation.predict_loader(model_a, loader, device)
    m, sims_a = evaluation.evaluate(model_a, img_emb, cap_emb, lens, device, return_sims=True)

    img_emb, cap_emb, lens = evaluation.predict_loader(model_b, loader, device)
    m, sims_b = evaluation.evaluate(model_b, img_emb, cap_emb, lens, device, return_sims=True)

    sims = (sims_a + sims_b) / 2

    import numpy as np

    i2t_metrics = evaluation.i2t(sims)
    t2i_metrics = evaluation.t2i(sims)

    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

    metrics = {}

    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}

    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum
    print(metrics)

    if args.outpath is not None:
        outpath = args.outpath
    else:
        filename = f'{data_name}.{lang}:{args.data_split}:ens_results.json'
        outpath = Path(opt.exp.outpath) / filename

    print('Saving into', outpath)
    file_utils.save_json(
        path=outpath,
        obj=metrics,
    )
