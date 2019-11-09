from glob import glob
import h5py
import torch
from pathlib import Path
import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class ExportToHDF5(Dataset):

    def __init__(self, root, pickle_pattern='images/*/*.pkl', outfile='f30k_resnet152.h5'):
        self.outfile = outfile
        self.dataset = h5py.File(f'{outfile}', 'w')
        pickle_path = f'{root}{pickle_pattern}'
        self.pickle_files = glob(pickle_path)
        print(f'found {len(self.pickle_files)} files')

    def __getitem__(self, index):
        file = self.pickle_files[index]
        features = torch.load(file)
        key = Path(file).relative_to(root)
        hkey = str(key).replace('.pkl', '')
        self.dataset.create_dataset(hkey, features.shape, dtype=np.float)
        return hkey, features

    def __len__(self, ):
        return len(self.pickle_files)


root = '/opt/jonatas/datasets/lavse/f30k_resnet152/'

dataset = ExportToHDF5(
    root=root,
    pickle_pattern='images/*/*.pkl',
    outfile='f30k_resnet152.h5'
)

exporter = DataLoader(dataset, num_workers=1, batch_size=1)

for d in tqdm(exporter, total=len(exporter)):
    pass

print(f'{dataset.outfile} created with {len(dataset.dataset)} keys')
print('Done.')
