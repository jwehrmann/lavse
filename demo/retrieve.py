#!flask/bin/python
from flask import Flask
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin
import os
import sys
sys.path.append('../')

import torch
from addict import Dict
from lavse.utils import options, helper
from lavse.data import loaders
from lavse.data import collate_fns
from lavse.data import adapters
from lavse.model import similarity
from lavse.model import LAVSE, lavse
from lavse.train.evaluation import predict_loader
from lavse.utils.logger import create_logger
from pathlib import Path
import numpy as np
from PIL import Image

logger = create_logger(
    level='info')

device = torch.device('cuda')

model_path = Path('../logs/m30k_precomp.en-de/liwe-adamax/')

opt = options.Options.load_yaml_opts(model_path / 'options.yaml')
# opt = options.Options()
opt = Dict(opt)
print(opt)
opt.dataset.vocab_paths = [Path('../') / Path(opt.dataset.vocab_paths[0])]
opt.dataset.val.batch_size = 256

data_path = helper.get_data_path(opt)
data_name = 'm30k_precomp'
lang = 'en'
f30k_path = data_path / 'f30k'

checkpoint_path = model_path / 'checkpoint_-1.pkl'

loader = loaders.get_loader(
    data_split='dev',
    data_path=data_path,
    data_name=data_name,
    loader_name='precomp',
    local_rank=0,
    lang=lang,
    text_repr=opt.dataset.text_repr,
    vocab_paths=opt.dataset.vocab_paths,
    ngpu=1,
    cnn=None,
    **opt.dataset.val
)

tokenizers = loader.dataset.tokenizers
if type(tokenizers) != list:
    tokenizers = [tokenizers]

dataset = loader.dataset
ids = np.loadtxt(data_path / data_name / 'dev_ids.txt', dtype=np.int)
dataset.ids = ids

print(len(dataset.ids))
print(len(dataset.images))
print(len(dataset.captions))

f30k = adapters.Flickr(f30k_path, 'dev')
a = f30k.get_filename_by_image_id(dataset.ids[0])


def predict_caption(caption, model, dataset,):
    print(caption)
    a = dataset.tokenizers[0](caption)
    a = collate_fns.liwe_padding([a])
    _, l = a
    a = {'caption': a}
    p = model.embed_captions(a).to(device).float() # (1, 1024)
    return p, l

from torchvision import transforms

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])

def retrieve(model, img_emb, cap_emb, lens):
    img_emb = torch.tensor(img_emb).clone().detach().to(device)
    cap_emb = torch.tensor(cap_emb).clone().detach().to(device)

    sim = model.compute_pairwise_similarity(
        model.similarity, img_emb, cap_emb, lens,
        shared_size=256
    )[:,0]
    # print(sim.min(), sim.mean(), sim.max())
    # exit()

    sim = sim.cpu().data.numpy()
    idxs = sim.argsort()[::-1][:3]
    idxs = idxs * 5
    image_ids = dataset.ids[idxs]

    # print(sim[idxs//5])

    images = [
        Image.open(f30k_path / f30k.get_filename_by_image_id(img_id))
        for img_id
        in image_ids
    ]
    return images

# load model and options
with torch.no_grad():

    _model = lavse(checkpoint_path, tokenizers)
    _model.eval()
    _model = _model.to(device)


    # img_embs, cap_embs, cap_lens = predict_loader(_model, loader, 'cuda')
    # torch.save(img_embs, 'm30k_precomp_embed.pkl')

    img_embs = torch.load('m30k_precomp_embed.pkl')
    img_embs = torch.tensor(img_embs).to(device).float()
    img_embs = img_embs.to(device).float()



if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
