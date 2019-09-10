import argparse
import os
import sys
from pathlib import Path

import torch

import params
from addict import Dict
from lavse.data.collate_fns import default_padding, liwe_padding
from lavse.data.loaders import get_loader
from lavse.model import model
from lavse.train import evaluation
from lavse.train.train import Trainer
from lavse.utils import file_utils, helper
from lavse.utils.logger import create_logger
from run import load_yaml_opts, parse_loader_name
from tqdm import tqdm
from lavse.model.similarity.measure import cosine_sim, l2norm


def load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    import pickle

    with open(path, 'wb') as f:
        pickle.dump(obj, path)

def get_test_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'options',
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

    tokenizer = loader.dataset.tokenizers[0]

    path = '/opt/jonatas/datasets/douglas/coco/val/captions.pickle'
    outpath_file = Path('/opt/jonatas/datasets/coco_embeddings/adapt/embed_val.pickle')
    outpath_folder = Path('/opt/jonatas/datasets/coco_embeddings/adapt/embed_val/')
    file = load_pickle(path)
    model = load_model(opt, [tokenizer])
    device = torch.device('cuda')
    model.eval()

    from tqdm import tqdm
    with torch.no_grad():
        outfile = {}
        for k, v in tqdm(file.items(), total=len(file)):
            tv, l = default_padding([tokenizer(x) for x in v])
            # tv, l = liwe_padding([tokenizer(x) for x in v])
            batch = {
                'caption': (tv, l)
            }
            cap = l2norm(model.embed_captions(batch).cpu(), dim=-1)

            torch.save(cap, outpath_folder / f'{k}.pkl')
            outfile[k] = cap.cpu()
        torch.save(outfile, outpath_file)
