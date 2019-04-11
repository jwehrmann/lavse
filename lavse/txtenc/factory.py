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
