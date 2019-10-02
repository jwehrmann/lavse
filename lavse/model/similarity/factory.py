from . import similarity as sim


_similarities = {
    'cosine': sim.Cosine,
    'scan_t2i': sim.StackedAttention,
}


def get_similarity_object(name, **kwargs):
    return _similarities[name](**kwargs)


def get_available_similarities():
    return _similarities.keys()
