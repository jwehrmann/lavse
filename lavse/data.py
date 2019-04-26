from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import collate_fns
from .loaders import CrossLanguageLoader, PrecompDataset, DummyDataset
from .tokenizer import Tokenizer
from .utils.file_utils import read_txt
from .utils.logger import get_logger


logger = get_logger()


def prepare_ml_data(instance, device):
    targ_a, lens_a, targ_b, lens_b, ids = instance
    targ_a = targ_a.to(device).long()
    targ_b = targ_b.to(device).long()
    return targ_a, lens_a, targ_b, lens_b, ids


def prepare_mm_data(instance, device):
    images, captions, lengths, ids = instance
    images = images.to(device)
    captions = captions.to(device).long()
    return images, captions, lengths, ids


class DataIterator:

    def __init__(self, loader, device, non_stop=False):
        self.data_iter = iter(loader)
        self.loader = loader
        self.non_stop = non_stop
        self.device = device

        self.cross_language = isinstance(loader.dataset, CrossLanguageLoader)

    def __str__(self):
        return f'{self.loader.dataset.data_name}.{self.loader.dataset.data_split}'

    def next(self):
        try:
            instance = next(self.data_iter)

            if self.cross_language:
                targ_a, lens_a, targ_b, lens_b, ids = prepare_ml_data(instance, self.device)
                logger.debug((
                    f'DataIter - CrossLang - Images: {targ_a.shape} '
                    f'DataIter - CrossLang - Target: {targ_a.shape} '
                    f'DataIter - CrossLang - Ids: {ids[:10]}\n'
                ))
                return targ_a, lens_a, targ_b, lens_b, ids

            images, captions, lengths, ids = prepare_mm_data(instance, self.device)
            #  = prepare_ml_data(instance, self.device)
            logger.debug((
                f'DataIter - Precomp - Images: {images.shape} {images.mean()} '
                f'DataIter - Precomp - Target: {captions.shape} '
                f'DataIter - Precomp - Lens: {np.sum(lengths)} '
                f'DataIter - Precomp - Ids: {ids[:10]}\n'
            ))
            return images, captions, lengths, ids

        except StopIteration:
            if self.non_stop:
                self.data_iter = iter(self.loader)
                return self.next()
            else:
                raise StopIteration(
                    'The data iterator has finished its job.'
                )


def get_loader(
    loader_name, data_path, data_name, data_split,
    batch_size, vocab_path, text_repr,
    lang='en', workers=4
):

    logger.debug('Get loader')
    dataset_class = get_dataset_class(loader_name)
    logger.debug(f'Dataset class is {dataset_class}')

    collate_fn, tokenizer = get_text_processing_objects(
        text_repr=text_repr,
        vocab_path=vocab_path,
        lang_adapt=('-' in lang)
    )

    logger.debug(f'Tokenizer built: {tokenizer}')

    dataset = dataset_class(
        data_path=data_path,
        data_name=data_name,
        data_split=data_split,
        tokenizer=tokenizer,
        lang=lang,
    )
    logger.debug(f'Dataset built: {dataset}')

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(data_split == 'train'),
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    logger.debug(f'Loader built: {loader}')

    return loader


def get_loaders(
        data_path, loader_name, data_name,
        vocab_path, batch_size,
        workers, text_repr,
        splits=['train', 'val', 'test'],
        langs=['en', 'en', 'en'],
    ):

    loaders = []
    loader_class = get_dataset_class(loader_name)
    for split, lang in zip(splits, langs):
        logger.debug(f'Getting loader {loader_class}/  {split} / Lang {lang}')
        loader = get_loader(
            loader_name=loader_name,
            data_path=data_path,
            data_name=data_name,
            batch_size=batch_size,
            workers=workers,
            text_repr=text_repr,
            data_split=split,
            lang=lang,
            vocab_path=vocab_path,
        )
        loaders.append(loader)

    return tuple(loaders)


# __datasets__ = {
#     'f30k_precomp': {
#         'lang': ['en', 'en'],
#     },
#     'm30k_precomp': {
#         'lang': ['en', 'de'],
#     },
#     'coco_precomp': {
#         'lang': ['en', 'de'],
#     },
#     'jap_precomp': {
#         'lang': ['en', 'jt'],
#     },
#     'jap_precomp': {
#         'lang': ['en', 'jt'],
#     },
# }

__loaders__ = {
    'dummy': {
        'class': DummyDataset,
    },
    'precomp': {
        'class': PrecompDataset,
    },
    'tensor': {
        'class': PrecompDataset,
    },
    'lang': {
        'class': CrossLanguageLoader,
    }
}


__text_representation__ = {
    'word': {
        'collate_fn': collate_fns.collate_fn_word,
        'collate_fn_lang': collate_fns.collate_lang_word,
        'tokenizer_args': {'char_level': False},
    },
    'char': {
        'collate_fn': collate_fns.collate_lang_word,
        'collate_fn_lang': collate_fns.collate_lang_word,
        'tokenizer_args': {'char_level': True},
    },
    'liwe': {
        'collate_fn': collate_fns.collate_fn_liwe,
        'collate_fn_lang': collate_fns.collate_lang_liwe,
        'tokenizer_args': {'char_level': True},
    },
}


def get_text_processing_objects(
    text_repr, vocab_path=None, lang_adapt=False,
):

    logger.debug(
        f'Getting text preprocessing {(text_repr, vocab_path)} '
    )

    settings = __text_representation__[text_repr]
    logger.debug(f'Settings {settings}')

    if lang_adapt:
        collate_fn = settings['collate_fn_lang']
    else:
        collate_fn = settings['collate_fn']

    tokenizer_args = settings['tokenizer_args']
    tokenizer_args.update({'vocab_path': vocab_path})
    logger.debug(f'Collate fn {collate_fn}')
    logger.debug(f'Tokenizer args {tokenizer_args}')

    tokenizer = Tokenizer(**tokenizer_args)
    logger.debug(f'Tokenizer {tokenizer}')

    return collate_fn, tokenizer


def get_dataset_class(loader_name):
    loader = __loaders__[loader_name]
    return loader['class']
