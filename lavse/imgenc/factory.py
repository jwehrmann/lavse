from . import precomp


__image_encoders__ = {
    'hierarchical': {
        'class': precomp.HierarchicalEncoder, 
        'args': {},
    },
    'scan': {
        'class': precomp.SCANImagePrecomp, 
        'args': {},
    },
    'vsepp_precomp': {
        'class': precomp.VSEImageEncoder, 
        'args': {},
    },
}


def get_available_imgenc():
    return __image_encoders__.keys()


def get_image_encoder(model_name, **kwargs):
    model_settings = __image_encoders__[model_name]
    model_class = model_settings['class']
    model_args = model_settings['args']
    arg_dict = dict(kwargs)
    arg_dict.update(model_args)
    model = model_class(**arg_dict)
    return model 
