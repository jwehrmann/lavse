from . import txtenc
from . import embedding
from . import pooling
import torch.nn as nn

import torch
import math


_text_encoders = {
    'gru': txtenc.RNNEncoder,
    'gru_glove': txtenc.GloveRNNEncoder,
    # 'scan_t2i': txtenc.EncoderText,
    'liwe_gru': txtenc.LiweGRU,
    'liwe_gru_glove': txtenc.LiweGRUGlove,
    'bert': txtenc.Bert,
    'bag': txtenc.BagOfEmbeddings,
}


def get_available_txtenc():
    return _text_encoders.keys()


def get_text_encoder(name, tokenizers, **kwargs):
    model_class = _text_encoders[name]
    model = model_class(tokenizers=tokenizers, **kwargs)
    return model


def get_txt_pooling(pool_name):

    _pooling = {
        'mean': pooling.mean_pooling,
        'max': pooling.max_pooling,
        'lens': pooling.last_hidden_state_pool,
        'none': pooling.none,
    }

    return _pooling[pool_name]
