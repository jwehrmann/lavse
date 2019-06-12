import logging


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
