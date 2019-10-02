# Language-Agnostic Visual Semantic Embeddings

This is LAVSE (pronounced læːvɪs), the official source code for our ICCV'19 paper. This repository is inspired by VSEPP, SCAN, and Bootstrap. 


## This code features: 

* Training and validation of SOTA models (COCO, Flickr30k, Multi30k, and YJCaptions).
* Single language and multi-language support (English, German, Japanese)
* Text encoders (GRU, Liwe, BERT, etc)
* Image encoders (Precomp, full ConvNet encoders)
* Similarity computation (easy to extend, it supports attention layers)
* Warm-up hinge-loss function
* Tensorboard logging

## Supported methods
* LIWE
* CLMR 
* SCAN (i2t, t2i)
* VSEPP
* Order Embeddings

## Pretrained models: 
* COCO
* Flickr30k
* Multi30k 
* YJCaptions

## Data downloading

### Precomp features

If you want to download everything, just run:
* All data compressed

Alternatively, you can download each dataset as follows: 

* COCO Features
* COCO Annotations
* Flickr30/Multi30k Features
* Flickr30/Multi30k Annotations
* YJCaptions Features
* YJCaptions Annotations

### Original images

To use full image encoders (only pure ConvNets is supported, i.e., no FasterRCNN-based encoders for now), you need to download the original images from:

* COCO train/val/test
* COCO annotations
* Flickr30k train/val/test
* Flickr30k annotations
* YJCaptions
* Multi30k


## Setup environment

Dependencies:
* Python >= 3.7
* Pytorch >= 1.0
* Addict
* PyYaml


The easiest way to setup your env is by running: 

``conda env create -f lavse370.yaml`` or ``pip install -f requirements.txt``

In addition, you can easily change the DATA_PATH (path for the downloaded data), and OUT_PATH (path used for storing models and logs), by running:

```
export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/lavse/
```

## Training Models

Model configuration is done via yaml files. So training models is super easy:

```
python run.py options/<yaml_path>.yaml
```

## Evaluating Models 

```
python test.py options/<path>.yaml --data_split <train/dev/test>
```

### Print your results by running
```

```

## Additional tools

### Build vocabulary 

### Align Vocabulary

## Demo
