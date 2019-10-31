from . import precomp
from . import fullencoder
from . import pooling
import torchvision


_image_encoders = {
    'hierarchical': precomp.HierarchicalEncoder,
    'vsepp': precomp.VSEImageEncoder,
    'scan': precomp.SCANImagePrecomp,
    'simple': precomp.SimplePrecomp,
    'image': fullencoder.FullImageEncoder,
}


def get_available_imgenc():
    return _image_encoders.keys()


def get_image_encoder(name, **kwargs):
    return _image_encoders[name](**kwargs)


def get_img_pooling(pool_name):

    _pooling = {
        'mean': pooling.mean_pooling,
        'max': pooling.max_pooling,
        'none': lambda x: x,
    }

    return _pooling[pool_name]
