# Language-Agnostic Visual-Semantic Embeddings

This is LAVSE (pronounced læːvɪs), the official source code for the ICCV'19 paper Language-Agnostic Visual-Semantic Embeddings. This repository is inspired by [VSEPP](https://github.com/fartashf/vsepp), [SCAN](https://github.com/kuanghuei/SCAN), and [BootstrapPytorch](https://github.com/Cadene/bootstrap.pytorch). 

Project page with live demo, another details and figures: https://jwehrmann.github.io/projects.lavse/.

## This code features: 

* Training and validation of SOTA models in multiple datasets (COCO, Flickr30k, Multi30k, and YJCaptions).
* Single language and multi-language support (English, German, Japanese).
* Text encoders (GRU, Liwe, Glove embeddings) easy to add new options, for instance BERT.
* Image encoders (Precomp, full ConvNet encoders).
* Similarity computation (easy to extend, it supports attention layers).
* Warm-up hinge-loss function.
* Tensorboard logging.
* We introduce novel retrieval splits for YJ captions for retrieval evaluation. 

## Supported methods
* LIWE
* CLMR
* SCAN (i2t, t2i)
* VSEPP

# Results 
## Single-language results

### COCO

| Approach        | Image Annotation R@1 | Image Retrieval R@1 | Test Time | 
|:-------------   |:------------------   |:------              |:---  | 
| LIWE+Glove (ours)     | 73.2                 | 57.9                | 1s   |
| LIWE (ours)     | 71.8                 | 55.5                | 1s   |
| CMLR (ours)     | 71.8                 | 56.2                | 1s   |      
| SCAN-t2i (ours) | 70.9                 | 56.4                | 50s  |
| SCAN-t2i        | 70.9                 | 56.4                | 250s |
| SCAN-i2t        | 69.2                 | 54.4                | 250s |

Note that our implementation of SCAN is 5x faster than the original code.

### Flickr30k

| Approach        | Image Annotation R@1 | Image Retrieval R@1 | Test Time |
|:-------------   |:------------------   |:------              |:---       |
| LIWE+Glove (ours)     | 69.6           | 51.2                | 1s        |
| LIWE (ours)     | 66.4                 | 47.5                | 1s        |
| CMLR (ours)     | 64.0                 | 46.8                | 1s        |
| SCAN-i2t        | 67.9                 | 43.9                | 250s      |
| SCAN-t2i        | 61.8                 | 45.8                | 250s      |

Note that our implementation of SCAN is 5x faster than the original code.

## Language-Invariant Results

### Multi30k 

| Approach        | Annotation (en) | Retrieval (en) | Annotation (de) | Retrieval (de) | #Params | 
|:-----   |:----------   |:----              |:---                    | :----  | :--- |
| LIWE (ours)     | 64.4                 | 47.5                | 53.0                 | 36.7         | 3M |
| CMLR (ours)     | 59.9                 | 43.9                | 50.4                 | 34.6         | 12M |
| BERT-ML         | 62.0                 | 42.7                | 50.9               | 33.2       | 110M |
 
### YJ Captions

| Approach        | Annotation (en) | Retrieval (en) | Annotation (jt) | Retrieval (jt) | Test Retrieval | 
|:-------------   |:------------------   |:------              |:---                    | :-- | :--- |
| LIWE (ours)     | 59.2                 | 46.1                | 48.6                 | 37.0       | 1s |
| CMLR (ours)     | 56.9                 | 43.2                | 51.4                 | 38.6       | 1s | 
| SCAN-t2i        | 58.2                 | 47.4                | 48.2                 | 39.6       |  250s |

# Datasets

## Precomp features

You can download each dataset as follows.

* COCO+F30k Data: `wget https://scanproject.blob.core.windows.net/scan-data/data.zip`
* Only annotations for COCO and F30k: `wget https://scanproject.blob.core.windows.net/scan-data/data_no_feature.zip`
* YJCaptions: `wget https://wehrmann.s3-us-west-2.amazonaws.com/jap_precomp.tar`
* Multi30k: `wget https://wehrmann.s3-us-west-2.amazonaws.com/m30k_precomp.tar`


IMPORTANT: set your data path using: 

```
export DATA_PATH=/path/to/data
```

Save the data in the $DATA_PATH, as follows: 

```
$DATA_PATH
├── coco_precomp
│   └── train_caps.en.txt
│   └── train_ims.npy
│   └── train_ids.npy
│   └── dev_caps.txt
│   ...
├── f30k_precomp
│   └── train_caps.en.txt
│   └── train_ims.npy
│   └── train_ids.npy
│   └── dev_caps.txt
│   ...
├── m30k_precomp
│   └── train_caps.en.txt
│   └── train_caps.de.txt
│   └── train_ims.npy
│   └── train_ids.npy
│   └── dev_caps.en.txt
│   └── dev_caps.de.txt
│   ...
├── jap_precomp
│   └── train_caps.jp.txt
│   └── train_caps.jt.txt
│   └── train_caps.en.txt
│   └── train_ims.npy
│   └── train_ids.npy
│   └── dev_caps.jp.txt
│   └── dev_caps.jt.txt
│   └── dev_caps.en.txt
│   ...

```

## Original images

To use full image encoders (only pure ConvNets is supported, i.e., no FasterRCNN-based encoders for now), you need to download the original images from their original sources.

# Environment Setup

Dependencies:
* Python >= 3.6
* Pytorch >= 1.1
* Addict
* PyYaml


The easiest way to setup your env is by running: 

``conda env create -f lavse370.yaml``

In addition, you can easily change the DATA_PATH (path for the downloaded data), by running.

```
export DATA_PATH=/opt/jonatas/datasets/lavse/
```

# Training Models

Model configuration is done via yaml files. So training models is super easy:

```
python run.py -o options/<yaml_path>.yaml
```

Inside `options/` you can find all the configuration files to reproduce our results. Scripts used to train models in our work are in `options/liwe/train.sh`. 

# Evaluating Models 

Evaluating models is also quite straightforward. It all depends on the yaml config file. 

```
python test.py options/<path>.yaml --data_split <train/dev/test>
```

## Print/compare results by running

```
$ cd tools
$ find ../logs/ -name *json -print0 | xargs -0 python print_result.py
```

# Citation

If you find this code/paper useful, please consider citing our work: 

```
@inproceedings{wehrmann2019iccv,
  title={Language-Agnostic Visual-Semantic Embeddings},
  author={Wehrmann, Jonatas and Souza, Douglas M. and Lopes, Mauricio A. and Barros, Rodrigo C.},
  booktitle={International Conference on Computer Vision},
  year={2019}
}

@inproceedings{wehrmann2018cvpr,
  title={Bidirectional retrieval made simple},
  author={Wehrmann, Jonatas and Barros, Rodrigo C},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7718--7726},
  year={2018}
}
```

