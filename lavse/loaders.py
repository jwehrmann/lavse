from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer
from .utils.logger import get_logger
from .utils.file_utils import read_txt

from . import collate_fns

logger = get_logger()


class PrecompDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name, 
        data_split, tokenizer, lang='en'
    ):  
        logger.debug(f'Precomp dataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer
        self.lang = lang 
        self.data_split = '.'.join([data_split, lang])
        self.data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name
        # Load Captions 
        caption_file = self.full_path / f'{data_split}_caps.{lang}.txt'
        self.captions = read_txt(caption_file)
        logger.debug(f'Read captions. Found: {len(self.captions)}')

        # Load Image features
        img_features_file = self.full_path / f'{data_split}_ims.npy'
        self.images = np.load(img_features_file)
        self.length = len(self.captions)

        logger.debug(f'Read feature file. Shape: {len(self.images.shape)}')

        # Each image must have five captions 
        assert (
            self.images.shape[0] == self.length 
            or self.images.shape[0]*5 == self.length
        ) 
        # Kiros code has redundancy in img feat
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        
        logger.info((
            f'Loaded PrecompDataset {self.data_name}/{self.data_split} with '
            f'images: {self.images.shape} and captions: {self.length}.'
        ))

    def get_img_dim(self):
        return self.images.shape[-1]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = self.images[img_id]
        image = torch.FloatTensor(image)

        caption = self.captions[index]
        words, chars = self.tokenizer(caption)

        return image, words, chars, index, img_id

    def __len__(self):
        return self.length
    
    def __repr__(self):
        return f'PrecompDataset.{self.data_name}.{self.data_split}'
    
    def __str__(self):
        return f'{self.data_name}.{self.data_split}' 


class CrossLanguageLoader(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name, data_split, 
        tokenizer, lang='en-de',
    ):  
        logger.debug((
            'CrossLanguageLoader dataset\n '
            f'{[data_path, data_split, tokenizer, lang]}'
        ))

        self.data_path = Path(data_path) 
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name
        self.data_split = '.'.join([data_split, lang])
        
        self.lang = lang
        self.tokenizer = tokenizer

        lang_base, lang_target = lang.split('-')
        base_filename = f'{data_split}_caps.{lang_base}.txt'
        target_filename = f'{data_split}_caps.{lang_target}.txt'

        base_file = self.full_path / base_filename
        target_file = self.full_path / target_filename
        
        logger.debug(f'Base: {base_file} - Target: {target_file}')
        # Paired files
        self.lang_a = read_txt(base_file)
        self.lang_b = read_txt(target_file)

        logger.debug(f'Base and target size: {(len(self.lang_a), len(self.lang_b))}')
        self.length = len(self.lang_a)
        assert len(self.lang_a) == len(self.lang_b)

        logger.info((
            f'Loaded CrossLangDataset {self.data_name}/{self.data_split} with '
            f'captions: {self.length}'
        ))

    def __getitem__(self, index):
        caption_a = self.lang_a[index]
        caption_b = self.lang_b[index]
        
        target_a, char_a = self.tokenizer(caption_a)
        target_b, char_b = self.tokenizer(caption_b)

        return target_a, char_a, target_b, char_b, index

    def __len__(self):
        return self.length

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'
    