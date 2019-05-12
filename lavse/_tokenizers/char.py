# coding=utf-8

import json
from collections import Counter
import numpy as np
from tqdm import tqdm
from .word import Vocabulary
import torch

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt


logger = get_logger()


class CharTokenizer(object):
    """
    This class converts texts into character or word-level tokens
    """

    def __init__(self, maxlen=None,):
        # Create a vocab wrapper and add some special tokens.
        vocab = Vocabulary()
        vocab.add_word(' ')
        vocab.add_word('<pad>')
        # vocab.add_word('<start>')
        # vocab.add_word('<end>')
        vocab.add_word('<unk>')
        self.vocab = vocab

        logger.debug(f'Created CharTokenizer with {len(self.vocab)} init tokens.')

    def fit_on_files(self, txt_files):
        logger.debug('Fit on files.')
        for file in txt_files:
            logger.info(f'Updating vocab with {file}')
            sentences = read_txt(file)
            self.fit(sentences)

    def fit(self, sentences, threshold=4):
        logger.debug(
            f'Fit char  on {len(sentences)} and t={threshold}'
        )
        counter = Counter()

        for sentence in tqdm(sentences, total=len(sentences)):
            for c in sentence:
                counter.update(c)
        # Discard if the occurrence of the word is less than threshold
        chars = [
            char for char, cnt in counter.items()
            if cnt >= threshold
        ]

        # Add words to the vocabulary.
        for char in chars:
            self.vocab.add_word(char)
        
        logger.info(f'Vocab built. Chars found {len(self.vocab)}')
        return self.vocab

    def save(self, outpath):
        logger.debug(f'Saving vocab to {outpath}')

        with open(outpath, "w") as f:
            json.dump(self.vocab.word2idx, f)

        logger.info(
            f'Vocab stored into {outpath} with {len(self.vocab)} chars.'
        )
    
    def load(self, path):
        logger.debug(f'Loading vocab from {path}')
        with open(path) as f:
            word2idx = json.load(f)
        vocab = Vocabulary()
        vocab.word2idx = word2idx
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        vocab.idx = max(vocab.idx2word)
        self.vocab = vocab
        logger.info(f'Loaded vocab containing {len(self.vocab)} chars')
        return self

    def chars_to_tokens(self, chars):
        return [self.vocab(char) for char in chars]

    def tokenize(self, sentence):
        tokens = self.chars_to_tokens(sentence) 
        return torch.LongTensor(tokens)

    def tokenize_sentences(self, texts):
        pass

    def decode_tokens(self, tokens):
        logger.debug(f'Decode tokens {tokens}')
        text = ''.join([
            self.vocab.get_word(token) for token in tokens
        ])
        return text

    def __len__(self):
        return len(self.vocab)
