from lavse.utils import create_logger
from lavse.data import get_loaders
from lavse.tokenizers import WordTokenizer
from pathlib import Path

logger = create_logger(level='debug')

data_path = Path('/home/jonatas/data/coco/')
data_name = 'f30k_precomp'

tokenizer = WordTokenizer()

# tokenizer.fit_on_files([
#     data_path / data_name / 'train_caps.en.txt',
#     data_path / data_name / 'train_caps.de.txt',
#     data_path / data_name / 'dev_caps.en.txt',
#     data_path / data_name / 'dev_caps.de.txt',
# ])
# tokenizer.save(f'vocab/{data_name}.json')

tokenizer.load(f'vocab/{data_name}.json')

train_loader, valid_loader = get_loaders(
    data_path='/home/jonatas/data/coco/',
    data_name='f30k_precomp', 
    loader_name='precomp',
    vocab_path=f'vocab/{data_name}.json',
    text_repr='word',
    batch_size=10, 
    workers=4, 
    splits=['train', 'dev',], 
    langs=['en', 'en',],
)

for instance in train_loader:
    print(instance)
    exit()