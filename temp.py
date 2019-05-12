from lavse.utils.logger import create_logger
from lavse import data
from lavse.data.tokenizer import Tokenizer
from pathlib import Path
import numpy as np

logger = create_logger(level='debug')

tokenizer = Tokenizer(vocab_path='vocab/complete.json')

ds = data.datasets.ImageDataset(
    '/home/jonatas/data/',
    data_name='f30k',
    data_split='test',
    tokenizer=tokenizer,
)

print(ds[0])
