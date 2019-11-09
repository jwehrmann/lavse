from glob import glob
import h5py
import torch
from pathlib import Path
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

ctx = mp.get_context("spawn")

class ExportToHDF5(Dataset):

    def __init__(self, root, pickle_pattern='images/*/*.pkl', outfile='f30k_resnet152.h5'):
        self.outfile = outfile
        self.dataset = h5py.File(outfile, 'w')
        pickle_path = f'{root}{pickle_pattern}'
        self.pickle_files = glob(pickle_path)
        print(f'found {len(self.pickle_files)} files')
        self.dataset.close()

    def __getitem__(self, index):
        self.dataset = h5py.File(self.outfile, 'r+')
        file = self.pickle_files[index]
        # features = torch.load(file)
        features = np.load(file)
        features = np.array(features)
        key = Path(file).relative_to(root)
        hkey = str(key).replace('.pkl', '')
        self.dataset.create_dataset(hkey, data=features, dtype=features.dtype)
        self.dataset.flush()
        self.dataset.close()
        return hkey, features

    def __len__(self, ):
        return len(self.pickle_files)


root = '/opt/jonatas/datasets/lavse/f30k_resnet152/'

dataset = ExportToHDF5(
    root=root,
    pickle_pattern='images/*/*.npy',
    outfile='f30k_resnet152_temp.h5'
)

exporter = DataLoader(dataset, num_workers=6, batch_size=50)

for d in tqdm(dataset, total=len(dataset)):
    pass

print(f'{dataset.outfile} created with {len(dataset.dataset)} keys')
print('Done.')
