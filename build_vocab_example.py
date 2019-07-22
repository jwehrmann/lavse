import torch
from lavse.utils.logger import create_logger
from lavse import data
from lavse.data.tokenizer import Tokenizer
from lavse.data.adapters import Flickr, Coco
from pathlib import Path
import numpy as np

from lavse.utils import helper

flatten = lambda l: [item for sublist in l for item in sublist]

logger = create_logger(level='debug')

# tokenizer = Tokenizer(
#     download_tokenizer=True,
#     char_level=False,
# )

# birds_train = data.datasets.Birds(
#     '/opt/jonatas/datasets/lavse/', data_name='birds', data_split='train',
# )

# birds_test = data.datasets.Birds(
#     '/opt/jonatas/datasets/lavse/', data_name='birds', data_split='test',
# )

fkt = Coco('/opt/jonatas/datasets/lavse/coco/', data_split='train')
fkv = Coco('/opt/jonatas/datasets/lavse/coco/', data_split='dev')

texts = flatten(fkv.image_captions.values())
texts += flatten(fkt.image_captions.values())

tokenizer = Tokenizer(
    download_tokenizer=True,
    char_level=False,
)

tokenizer.fit(texts)
tokenizer.save('vocab/coco.json')

