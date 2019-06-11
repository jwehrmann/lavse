# TODO: improve this
from . import similarity as sim
from addict import Dict


_similarities = {
    'cosine': {
        'class': sim.Cosine,
        'args': {},
    },
    'order': None,
    'scan_i2t': {
        'class': sim.StackedAttention,
        'args': Dict(
            i2t=True, agg_function='Mean',
            feature_norm='clipped_l2norm',
            lambda_lse=None, smooth=4,
        ),
    },
    'adaptive': {
        'class': sim.AdaptiveEmbedding,
        'args': Dict(
            task='t2i',
        ),
    },
    'adaptive_norm': {
        'class': sim.AdaptiveEmbedding,
        'args': Dict(
            task='t2i',
            norm=True,
        ),
    },
    'adaptive_k4': {
        'class': sim.AdaptiveEmbedding,
        'args': Dict(
            task='t2i',
            k=4,
        ),
    },
    'adaptive_i2t': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': {},
    },
    'adapt_i2t': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': {},
    },
    'adapt_i2t_g4': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': {
            'groups': 4,
        },
    },
    'adaptive_i2t_feat_norm_bn': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            norm=True,
            nonlinear_proj=False,
        ),
    },
    'adaptive_i2t_condvec': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            cond_vec=True,
        ),
    },
    'adaptive_i2t_condvec_linear': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            cond_vec=True,
            nonlinear_proj=False,
        ),
    },
    'adaptive_i2t_bn_linear': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            k=8,
            normalization='batchnorm',
            nonlinear_proj=False
        ),
    },
    'adaptive_i2t_in': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            k=8,
            normalization='instancenorm',
            nonlinear_proj=True
        ),
    },
    'adaptive_i2t_no_norm': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            k=8,
            normalization=None,
            nonlinear_proj=True,
        ),
    },
    'adaptive_i2t_no_norm_linear': {
        'class': sim.AdaptiveEmbeddingI2T,
        'args': Dict(
            k=8,
            normalization=None,
            nonlinear_proj=False,
        ),
    },
    'proj_conv_reduced': {
        'class': sim.ProjConvReducedI2T,
        'args': Dict(
            k=8,
        ),
    },
    'proj_conv_reduced': {
        'class': sim.ProjConvReducedI2T,
        'args': Dict(
            k=8,
        ),
    },
    'dynconv': {
        'class': sim.DynConvI2T,
        'args': Dict(
        ),
    },
    'projconv': {
        'class': sim.ProjConvI2T,
        'args': Dict(
            kernel_size=3,
            reduce_proj=8,
            groups=512,
        ),
    },
    'projrnn': {
        'class': sim.ProjRNNReducedI2T,
        'args': Dict(
            k=8
        ),
    },
}


def get_similarity_object(similarity_name, **kwargs):
    settings = _similarities[similarity_name]
    args_dict = settings['args']
    args_dict.update(**kwargs)
    return settings['class'](**args_dict)


def get_sim_names():
    return _similarities.keys()
