from lavse.utils.logger import create_logger
from lavse.data import get_loaders
from lavse.tokenizer import Tokenizer
from pathlib import Path
import numpy as np 

logger = create_logger(level='debug')

data_path = Path('/home/jonatas/data/lavse/')
data_name = 'f30k_precomp'

tokenizer = Tokenizer()

# tokenizer.fit_on_files([
#     data_path / data_name / 'train_caps.en.txt',
#     data_path / data_name / 'train_caps.de.txt',
#     data_path / data_name / 'dev_caps.en.txt',
#     data_path / data_name / 'dev_caps.de.txt',
# ])
# tokenizer.save(f'vocab/{data_name}.json')
tokenizer.load(f'vocab/char.json')
lens = []
for w, i in tokenizer.vocab.word2idx.items():
    lens.append(len(w))

print(f'{np.mean(lens),np.max(lens),np.std(lens),np.min(lens)}')


# print(tokenizer.vocab.word2idx)
# a = tokenizer.tokenize(' hello mate, how are ya * !23 iwe0()!2ß ')
# print(a)

# a = tokenizer.decode_tokens(a.numpy())

# print(a)


# train_loader, valid_loader = get_loaders(
#     data_path='/home/jonatas/data/coco/',
#     data_name='f30k_precomp', 
#     loader_name='precomp',
#     vocab_path=f'vocab/{data_name}.json',
#     text_repr='word',
#     batch_size=10, 
#     workers=4, 
#     splits=['train', 'dev',], 
#     langs=['en', 'en',],
# )

# for instance in train_loader:
#     print(instance)
#     exit()