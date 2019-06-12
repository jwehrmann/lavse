from . import precomp
from . import fullencoder
from . import pooling
import torchvision


_image_encoders = {
    'hierarchical': {
        'class': precomp.HierarchicalEncoder,
        'args': {
            'img_dim': 2048,
        },
    },
    'sa': {
        'class': precomp.SAImgEncoder,
        'args': {
            'img_dim': 2048,
        },
    },
    'sagru': {
        'class': precomp.SAGRUImgEncoder,
        'args': {
            'img_dim': 2048,
        },
    },
    'scan': {
        'class': precomp.SCANImagePrecomp,
        'args': {
            'img_dim': 2048,
        },
    },
    'vsepp_precomp': {
        'class': precomp.VSEImageEncoder,
        'args': {
            'img_dim': 2048,
        },
    },
    'full_image': {
        'class': fullencoder.ImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet152,
            'img_dim': 248,
        },
    },
    'resnet50': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet50,
            'img_dim': 2048,
        },
    },
    'resnet50_ft': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet50,
            'img_dim': 2048,
            'finetune': True,
        },
    },
    'resnet101': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet101,
            'img_dim': 2048,
            'proj_regions': False,
        },
    },
    'resnet101_ft': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet101,
            'img_dim': 2048,
            'finetune': True,
        },
    },
    'resnet152': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet152,
            'img_dim': 2048,
            'proj_regions': False,
        },
    },
    'resnet152_ft': {
        'class': fullencoder.FullImageEncoder,
        'args': {
            'cnn': torchvision.models.resnet152,
            'img_dim': 2048,
            'finetune': True,
        },
    },
    'vsepp_pt': {
        'class': fullencoder.VSEPPEncoder,
        'args': {
            'cnn_type': 'resnet152',
        },
    },
    'img_proj': {
        'class': precomp.ImageProj,
        'args': {
            'img_sa': False,
            'projection': False,
            'non_linear_proj': False,
            'projection_sa': False,
        },
    },

}


def get_available_imgenc():
    return _image_encoders.keys()


def get_image_encoder(model_name, **kwargs):
    model_settings = _image_encoders[model_name]
    model_class = model_settings['class']
    model_args = model_settings['args']
    arg_dict = dict(kwargs)
    arg_dict.update(model_args)
    model = model_class(**arg_dict)
    return model


def get_img_pooling(pool_name):

    _pooling = {
        'mean': pooling.mean_pooling,
        'max': pooling.max_pooling,
        'none': lambda x: x,
    }

    return _pooling[pool_name]
