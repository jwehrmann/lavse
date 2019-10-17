

from . import tokenizer
import torch.nn as nn

import torch
import math


_tokenizers = {
    'default': tokenizer.Tokenizer,
    'bert': tokenizer.BertTokenizer,
}


def get_available_txtenc():
    return _tokenizers.keys()


def get_tokenizer(name, vocab_path):
    model_class = _tokenizers[name]
    if name == 'bert':
        return model_class()
    return model_class(vocab_path=vocab_path)
