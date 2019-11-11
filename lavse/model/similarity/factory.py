from . import similarity as sim


_similarities = {
    'cosine': sim.Cosine,
    'scan_t2i': sim.StackedAttention,
    'kp_t2i': sim.KernelProjectionT2I,
    'kp_i2t': sim.KernelProjectionI2T,
    # 'kp_i2t': sim.KPTextToImageOneToMany,
    'dynconv_fb_t2i': sim.DynConvT2i,
    'sta': sim.STASimilarity,
    'dyn_mm_t2i': sim.DynConvMultimodalT2i,
}


def get_similarity_object(name, **kwargs):
    return _similarities[name](**kwargs)


def get_available_similarities():
    return _similarities.keys()
