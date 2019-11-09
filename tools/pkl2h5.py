# from glob import glob
# import h5py
# import torch
# from pathlib import Path
# import numpy as np
# import torch.multiprocessing as mp
# from tqdm import tqdm

# from torch.utils.data import Dataset, DataLoader

# ctx = mp.get_context("forkserver")

# root = '/opt/jonatas/datasets/lavse/f30k_resnet152/'
# pickle_pattern='images/*/*.pkl'
# outfile='f30k_resnet152.h5'
# dataset = h5py.File(f'{outfile}', 'w')
# pickle_path = f'{root}{pickle_pattern}'
# pickle_files = glob(pickle_path)
# print(f'found {len(pickle_files)} files')

# pickle_pattern='images/*/*.pkl'
# outfile='f30k_resnet152.h5'

# n = len(pickle_files)
# max_shape = [n, 2048, 14, 14]

# images = dataset.create_dataset('images', max_shape, dtype=np.float)
# keys = dataset.create_dataset('keys', (n,), dtype=bytes)

# for i, file in tqdm(enumerate(pickle_files), total=len(pickle_files)):
#     features = torch.load(file)
#     key = Path(file).relative_to(root)
#     hkey = str(key).replace('.pkl', '')
#     images[i] = features
#     keys[i] = hkey

# print(f'{dataset.outfile} created with {len(dataset.dataset)} keys')
# print('Done.')

from glob import glob
import h5py
import torch
from pathlib import Path
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

ctx = mp.get_context("forkserver")

root = '/opt/jonatas/datasets/lavse/f30k_resnet152/'
pickle_pattern='images/*/*.pkl'
outfile='f30k_resnet152.h5'
dataset = h5py.File(f'{outfile}', 'w')
pickle_path = f'{root}{pickle_pattern}'
pickle_files = glob(pickle_path)
print(f'found {len(pickle_files)} files')

pickle_pattern='images/*/*.pkl'
outfile='f30k_resnet152.h5'

n = len(pickle_files)
max_shape = [n, 2048, 14, 14]

group = dataset.create_group('data',)
dataset.close()

for i, file in tqdm(enumerate(pickle_files), total=len(pickle_files)):
    dataset = h5py.File(f'{outfile}', 'r+')
    group = dataset['data']
    features = torch.load(file)
    key = Path(file).relative_to(root)
    hkey = str(key).replace('.pkl', '')
    ds = group.create_dataset(hkey, data=features.numpy(), dtype=np.float)
    # ds.flush()
    dataset.close()

print(f'{dataset.outfile} created with {len(dataset.dataset)} keys')
print('Done.')
