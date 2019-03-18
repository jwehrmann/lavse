import logging


def read_txt(path):
    return open(path).read().strip().split('\n')

