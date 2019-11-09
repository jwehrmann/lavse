import math
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import pretrainedmodels
import pretrainedmodels.utils as utils
from torch import nn

from munch import munchify
from tqdm import tqdm

import numpy as np


def ensure_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def stack_op(op,):
    return lambda crops: torch.stack([op(crop) for crop in crops])


class TenCropTransformImage(object):

    def __init__(self, opts, scale=0.875, preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        tfs.append(transforms.TenCrop(max(self.input_size)))
        tfs.append(stack_op(transforms.ToTensor()))
        tfs.append(stack_op(utils.ToSpaceBGR(self.input_space=='BGR')))
        tfs.append(stack_op(utils.ToRange255(max(self.input_range)==255)))
        tfs.append(stack_op(transforms.Normalize(mean=self.mean, std=self.std)))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class ImageFolderPath(datasets.ImageFolder):
    def __init__(
            self, root, transform=None, target_transform=None,
            loader=datasets.folder.default_loader, is_valid_file=None
        ):
        super(ImageFolderPath, self).__init__(
            root=root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        self.root = Path(root)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        path = str(Path(path).relative_to(self.root))
        return sample, path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = pretrainedmodels.__dict__['resnet152'](num_classes=1000, pretrained='imagenet')
# model.input_size = [3, 448, 448]

import types

def forward(self, x):
    return self.features(x)

model.forward = types.MethodType(forward, model)
transform_image = TenCropTransformImage(model)

# model = nn.DataParallel(model).cuda()
# model = model.cuda()

outpath = Path('/opt/jonatas/datasets/lavse/f30k_resnet152/')

ensure_dir(outpath)

image_path = Path('/opt/jonatas/datasets/lavse/f30k/')
image_dataset = ImageFolderPath(image_path, transform=transform_image,)
loader = DataLoader(
    image_dataset, shuffle=False,
    batch_size=20, num_workers=6
)

model.eval()
model.to(device)

import h5py
import os
outfile = 'resnet_7x7.h5'
h5 = h5py.File(outfile, 'w')
h5.close()

@torch.no_grad()
def extract_features():
    for images, paths in tqdm(loader, total=len(loader)):
        bs, ncrops, c, h, w = images.shape
        images = images.view(bs*ncrops, c, h, w)
        images = images.cuda()
        pred = model(images)
        result = pred.view(bs, ncrops, *pred.shape[1:]).mean(1)
        result = result.data.cpu()
        # print(result.shape)
        h5 = h5py.File(outfile, 'r+')
        for i, p in zip(result, paths):
            full_path = outpath / p
            ensure_dir(full_path.parent)
            # dataset(f'{outpath / p}.npy', i.numpy())
            h5.create_dataset(p, data=i.numpy(), dtype=np.float)
        h5.flush()
        h5.close()

        # for i, p in zip(result, paths):
        #     full_path = outpath / p
        #     ensure_dir(full_path.parent)
        #     np.save(f'{outpath / p}.npy', i.numpy())
            # print(i.shape)

extract_features()
