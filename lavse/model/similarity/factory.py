# TODO: improve this
from . import similarity as sim
from addict import Dict


# _similarities = {
#     'cosine': {
#         'class': sim.Cosine,
#         'args': {},
#     },
#     'attentive': {
#         'class': sim.Attentive,
#         'args': {},
#     },
#     'order': None,
#     'scan_i2t': {
#         'class': sim.StackedAttention,
#         'args': Dict(
#             i2t=True, agg_function='Mean',
#             feature_norm='clipped_l2norm',
#             lambda_lse=None, smooth=4,
#         ),
#     },
#     'scan_t2i': {
#         'class': sim.StackedAttention,
#         'args': Dict(
#             i2t=False, agg_function='Mean',
#             feature_norm='clipped_l2norm',
#             lambda_lse=None, smooth=9,
#         ),
#     },
#     'adapt_t2i': {
#         'class': sim.AdaptiveEmbeddingT2I,
#         'args': Dict(
#             task='t2i',
#         ),
#     },
#     'attn_adapt_t2i': {
#         'class': sim.AttentionAdaptiveEmbeddingT2I,
#         'args': {},
#     },
#     'adapt_i2t': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': {},
#     },
#     'adapt_i2t_g4': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': {
#             'groups': 4,
#         },
#     },
#     'adaptive_i2t_feat_norm_bn': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             norm=True,
#             nonlinear_proj=False,
#         ),
#     },
#     'adaptive_i2t_condvec': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             cond_vec=True,
#         ),
#     },
#     'adaptive_i2t_condvec_linear': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             cond_vec=True,
#             nonlinear_proj=False,
#         ),
#     },
#     'adaptive_i2t_bn_linear': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             k=8,
#             normalization='batchnorm',
#             nonlinear_proj=False
#         ),
#     },
#     'adaptive_i2t_in': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             k=8,
#             normalization='instancenorm',
#             nonlinear_proj=True
#         ),
#     },
#     'adaptive_i2t_no_norm': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             k=8,
#             normalization=None,
#             nonlinear_proj=True,
#         ),
#     },
#     'adaptive_i2t_no_norm_linear': {
#         'class': sim.AdaptiveEmbeddingI2T,
#         'args': Dict(
#             k=8,
#             normalization=None,
#             nonlinear_proj=False,
#         ),
#     },
#     'projconv': {
#         'class': sim.ProjConvI2T,
#         'args': Dict(),
#     },
#     'projconv_t2i': {
#         'class': sim.ProjConvT2I,
#         'args': Dict(),
#     },
#     'projconv_t2i_cond': {
#         'class': sim.ProjConvT2IAgg,
#         'args': Dict(),
#     },
#     'scan_adapt': {
#         'class': sim.AdaptiveStackedAttention,
#         'args': {},
#     }
# }

_similarities = {
    'cosine': sim.Cosine,
}

# _similarities['projconv_i2t'] = _similarities['projconv']


def get_similarity_object(name, **kwargs):
    # args_dict = settings['args']
    # args_dict.update(**kwargs)
    return _similarities[name](**kwargs)


def get_sim_names():
    return _similarities.keys()
