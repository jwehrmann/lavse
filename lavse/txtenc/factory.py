from . import txtenc
from . import pooling
import torch.nn as nn


__text_encoders__ = {
    'gru': {
        'class': txtenc.RNNEncoder,
        'args': {
            'use_bi_gru': True,
            'rnn_type': nn.GRU,

        },
    },
    'scan': {
        'class': txtenc.EncoderText,
        'args': {
            'use_bi_gru': True,
            'num_layers': 1,
        },
    },
    'sa': {
        'class': txtenc.SelfAttn,
        'args': {},
    },
    'attngru': {
        'class': txtenc.SelfAttnGRU,
        'args': {
            'use_bi_gru': True,
            'no_txtnorm': True,

        },
    },
    'attngru_cat': {
        'class': txtenc.SelfAttnGRUWordCat,
        'args': {
            'use_bi_gru': True,
            'no_txtnorm': True,

        },
    },
    'attngru_cat_ek2': {
        'class': txtenc.SelfAttnGRUWordCat,
        'args': {
            'use_bi_gru': True,
            'no_txtnorm': True,
            'embed_k': 2,
        },
    },
    'attngru_cat_ekk': {
        'class': txtenc.SelfAttnGRUWordCat,
        'args': {
            'use_bi_gru': True,
            'no_txtnorm': True,
            'embed_k': k,
        },
    },
    'convgru_sa': {
        'class': txtenc.ConvGRU,
        'args': {
            'use_bi_gru': True,
        },
    },
    'liwe_gru': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
        },
    },
    'emb_proj': {
        'class': txtenc.WordEmbeddingProj,
        'args': {
            'word_sa': True,
            'projection': False,
            'non_linear_proj': False,
            'projection_sa': False,
        },
    },
}


def get_available_txtenc():
    return __text_encoders__.keys()


def get_text_encoder(model_name, **kwargs):
    model_settings = __text_encoders__[model_name]
    model_class = model_settings['class']
    model_args = model_settings['args']
    arg_dict = dict(kwargs)
    arg_dict.update(model_args)
    model = model_class(**arg_dict)
    return model


def get_txt_pooling(pool_name):

    _pooling = {
        'mean': pooling.mean_pooling,
        'max': pooling.max_pooling,
        'lens': pooling.last_hidden_state_pool,
        'none': pooling.none,
    }

    return _pooling[pool_name]
