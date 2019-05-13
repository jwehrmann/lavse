from lavse.utils.logger import create_logger
from lavse import data
from lavse.data.tokenizer import Tokenizer
from lavse.data.adapters import Flickr
from pathlib import Path
import numpy as np

logger = create_logger(level='debug')

tokenizer = Tokenizer(vocab_path='vocab/complete.json')

fk = Flickr('/opt/jonatas/datasets/lavse/f30k/', data_split='train')
exit()

ds = data.datasets.ImageDataset(
    '/home/jonatas/data/',
    data_name='f30k',
    data_split='train',
    tokenizer=tokenizer,
)

print(ds[0])
