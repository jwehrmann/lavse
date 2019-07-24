from . import txtenc
from . import embedding
from . import pooling
import torch.nn as nn

import torch
import math


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


__text_encoders__ = {
    'gru': {
        'class': txtenc.RNNEncoder,
        'args': {
            'use_bi_gru': True,
            'rnn_type': nn.GRU,
        },
    },
    'gru_glove': {
        'class': txtenc.GloveRNNEncoder,
        'args': {
        },
    },
    'gru_glove_global': {
        'class': txtenc.GloveRNNEncoderGlobal,
        'args': {
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
    'attngru_cat_ek4': {
        'class': txtenc.SelfAttnGRUWordCat,
        'args': {
            'use_bi_gru': True,
            'no_txtnorm': True,
            'embed_k': 4,
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
    'liwe_gru_scale_384_relu': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'partial_class': embedding.PartialConcatScale,
            'liwe_activation': nn.ReLU(),
            # 'liwe_activation': GELU(),
            'liwe_batch_norm': True,
            'liwe_neurons': [384, 384],
            # 'liwe_dropout': 0.1,
        },
    },
    'liwe_gru_scale_384_gelu_nobn': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'partial_class': embedding.PartialConcatScale,
            # 'liwe_activation': nn.ReLU(),
            'liwe_activation': GELU(),
            'liwe_batch_norm': False,
            'liwe_neurons': [384, 384],
            # 'liwe_dropout': 0.1,
        },
    },
    'liwe_gru_gru': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'partial_class': embedding.PartialGRUs,
        },
    },
    'liwe_gru_gru_proj': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'partial_class': embedding.PartialGRUProj,
        },
    },
    'liwe_gru_256': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [256, 256],
        },
    },
    'liwe_gru_128_384': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [128, 384],
        },
    },
    'liwe_convgru_256_256': {
        'class': txtenc.LiweConvGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [256, 256],
        },
    },
    'liwe_gru_256_256_256': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [256, 256, 256],
        },
    },
    'liwe_gru_256_512': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [256, 512,],
        },
    },
    'liwe_convgru_384_384': {
        'class': txtenc.LiweConvGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [384, 384],
        },
    },
    'liwe_gru_384_384_gelu': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [384, 384],
            'partial_class': embedding.PartialConcat,
            'liwe_activation': GELU(),
            'liwe_batch_norm': False,
        },
    },
    'liwe_gru_384_384_gelu_withbn': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [384, 384],
            'partial_class': embedding.PartialConcat,
            'liwe_activation': GELU(),
            'liwe_batch_norm': True,
        },
    },
    'liwe_gru_512_512': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [512, 512],
        },
    },
    'liwe_gru_1024_512': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [1024, 512],
        },
    },
    'liwe_gru_128_512': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [128, 512],
        },
    },
    'liwe_gru_384_384': {
        'class': txtenc.LiweGRU,
        'args': {
            'use_bi_gru': True,
            'liwe_neurons': [384, 384],
        },
    },
    'emb_proj': {
        'class': txtenc.WordEmbeddingProj,
        'args': {
            'word_sa': False,
            'projection': False,
            'non_linear_proj': False,
            'projection_sa': False,
        },
    },
}

__text_encoders__['liwe_gru_384'] = __text_encoders__['liwe_gru_384_384']


def get_available_txtenc():
    return __text_encoders__.keys()

def get_text_encoder(name, **kwargs):
    model_class = __text_encoders__[name]['class']
    model = model_class(**kwargs)
    return model

# def get_text_encoder(name, **kwargs):
#     model_settings = __text_encoders__[name]
#     model_class = model_settings['class']
#     model_args = model_settings['args']
#     arg_dict = dict(kwargs)
#     arg_dict.update(model_args)
#     model = model_class(**arg_dict)
#     return model


def get_txt_pooling(pool_name):

    _pooling = {
        'mean': pooling.mean_pooling,
        'max': pooling.max_pooling,
        'lens': pooling.last_hidden_state_pool,
        'none': pooling.none,
    }

    return _pooling[pool_name]
