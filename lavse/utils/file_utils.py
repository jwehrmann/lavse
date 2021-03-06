import logging
import yaml
from yaml import Dumper
import copy
from addict import Dict
import os


def read_txt(path):
    return open(path).read().strip().split('\n')


def save_json(path, obj):
    import json
    with open(path, 'w') as fp:
        json.dump(obj, fp)


def load_json(path):
    import json
    with open(path, 'rb') as fp:
        return json.load(fp)


def save_yaml_opts(path_yaml, opts):

        # Warning: copy is not nested
    options = copy.copy(opts)

    # https://gist.github.com/oglops/c70fb69eef42d40bed06
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())
    Dumper.add_representer(Dict, dict_representer)

    with open(path_yaml, 'w') as yaml_file:
        yaml.dump(options, yaml_file, Dumper=Dumper, default_flow_style=False)


def merge_dictionaries(dict1, dict2):
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dictionaries(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]


def load_yaml_opts(path_yaml):
    """ Load options dictionary from a yaml file
    """
    result = {}
    with open(path_yaml, 'r') as yaml_file:
        options_yaml = yaml.safe_load(yaml_file)
        includes = options_yaml.get('__include__', False)
        if includes:
            if type(includes) != list:
                includes = [includes]
            for include in includes:
                filename = '{}/{}'.format(os.path.dirname(path_yaml), include)
                if os.path.isfile(filename):
                    parent = load_yaml_opts(filename)
                else:
                    parent = load_yaml_opts(include)
                merge_dictionaries(result, parent)
        merge_dictionaries(result, options_yaml) # to be sure the main options overwrite the parent options
    result.pop('__include__', None)
    result = Dict(result)
    return result


def parse_loader_name(data_name):
    if '.' in data_name:
        name, lang = data_name.split('.')
        return name, lang
    else:
        return data_name, None


def get_logdir(logdir, comment=''):
    import shutil
    import os
    if logdir is None:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(
            'runs', current_time + '_' + comment
        )

    if os.path.exists(logdir):
        a = input(f'{logdir} already exists! Do you want to rewrite it? [y/n] ')
        if a.lower() == 'y':
            shutil.rmtree(logdir)
        else:
            exit()


    return logdir

