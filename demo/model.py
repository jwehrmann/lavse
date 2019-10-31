# import the necessary packages
import numpy as np
import settings
import redis
import time
import json


def tob64(pil_img):
    from PIL import Image
    from io import BytesIO
    import base64
    # img = Image.fromarray(pil_img, 'RGB')
    buffer = BytesIO()
    pil_img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()
    st = base64.b64encode(myimage).decode("utf-8")
    b64 = f"data:image/jpeg;base64,{st}"
    return b64


print('Connecting to ', settings.REDIS_HOST)
# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
    port=settings.REDIS_PORT, db=settings.REDIS_DB)
db.flushdb()

print('Xonnected to redis')

def classify_process():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)


    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.QUEUE, 0,
            settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = None
        print('QUEUE len', db.llen(settings.QUEUE))

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            query = q['query']['query']
            imageID = q["id"]
            print(query)
            c, l = predict_caption(
                query,
                _model,
                dataset
            )

            images = retrieve(
                _model, img_embs, c, l,
            )

            imgs = []
            for i, image in enumerate(images):
                # image.save(f'{i}.png')
                image = transform(image)
                str_img = tob64(image)
                imgs.append(str_img)

            db.set(imageID, json.dumps(imgs))

            # remove the set of images from our queue
            db.ltrim(settings.QUEUE, 1, -1)
            # image = helpers.base64_decode_image(q["image"],
            #     settings.IMAGE_DTYPE,
            #     (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
            #         settings.IMAGE_CHANS))

            # check to see if the batch list is None
            # if batch is None:
            #     batch = image

            # # otherwise, stack the data
            # else:
            #     batch = np.vstack([batch, image])

            # # update the list of image IDs
            # imageIDs.append(q["id"])

        # # check to see if we need to process the batch
        # if len(imageIDs) > 0:
        #     # classify the batch
        #     print("* Batch size: {}".format(batch.shape))
        #     preds = model.predict(batch)
        #     results = imagenet_utils.decode_predictions(preds)

        #     # loop over the image IDs and their corresponding set of
        #     # results from our model
        #     for (imageID, resultSet) in zip(imageIDs, results):
        #         # initialize the list of output predictions
        #         output = []

        #         # loop over the results and add them to the list of
        #         # output predictions
        #         for (imagenetID, label, prob) in resultSet:
        #             r = {"label": label, "probability": float(prob)}
        #             output.append(r)

        #         # store the output predictions in the database, using
        #         # the image ID as the key so we can fetch the results
        #         db.set(imageID, json.dumps(output))

        #     # remove the set of images from our queue
        #     db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process


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
import settings

logger = create_logger(
    level='info')

device = torch.device('cuda')

model_path = Path(settings.MODEL_PATH)

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

checkpoint_path = model_path / 'best_model.pkl'

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


    img_embs, cap_embs, cap_lens = predict_loader(_model, loader, 'cuda')
    torch.save(img_embs, 'm30k_precomp_embed.pkl')

    img_embs = torch.load('m30k_precomp_embed.pkl')
    img_embs = torch.tensor(img_embs).to(device).float()
    img_embs = img_embs.to(device).float()


#if __name__ == "__main__":
classify_process()
