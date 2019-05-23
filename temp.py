import torch
from lavse.utils.logger import create_logger
from lavse import data
from lavse.data.tokenizer import Tokenizer
from lavse.data.adapters import Flickr
from pathlib import Path
import numpy as np

from lavse.utils import helper 

flatten = lambda l: [item for sublist in l for item in sublist]

logger = create_logger(level='debug')

tokenizer = Tokenizer(
    download_tokenizer=True, 
    char_level=False,
)

fkt = Flickr('/opt/jonatas/datasets/lavse/f30k/', data_split='train')
fkv = Flickr('/opt/jonatas/datasets/lavse/f30k/', data_split='dev')

loader = data.loaders.get_loader('image', '/home/jonatas/data/', 'f30k', 'test', 128, 'vocab/f30k.json', 'word',)
for i in loader:
    print(i[0].shape)

# texts = flatten(fkv.image_captions.values())
# texts += flatten(fkt.image_captions.values())

# tokenizer = Tokenizer(
#     download_tokenizer=True, 
#     char_level=False,
# )

# tokenizer.fit(texts)
# tokenizer.save('vocab/f30k.json')