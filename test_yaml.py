from run import load_yaml_opts, parse_loader_name
import params
from lavse.utils.logger import create_logger
from lavse.data.loaders import get_loader
from lavse.train.train import Trainer
from lavse.model import model
from lavse.utils import helper, file_utils
from pathlib import Path
import os
import torch
from tqdm import tqdm
import sys



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

    model = model.LAVSE(**opt.model, tokenizers=tokenizers)#.to(device)

    # import pickle
    # with open(Path(opt.exp.outpath) / 'best_model.pkl', 'rb', 0) as f:
    #     pickle.load(f)

    print(model)
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

    model.set_devices_(
        txt_devices=opt.model.txt_enc.devices,
        img_devices=opt.model.img_enc.devices,
        loss_device=opt.model.similarity.device,
    )

    is_master = True
    model.master = is_master # FIXME: Replace "if print" by built_in print
    print_fn = (lambda x: x) if not is_master else tqdm.write

    trainer = Trainer(
        model=model,
        args={'args': args, 'model_args': opt.model},
        sysoutlog=print_fn,
    )

    result, rs = trainer.evaluate_loaders(
        [loader]
    )
    print(result)

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
