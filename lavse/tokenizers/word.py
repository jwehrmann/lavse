import json
import logging
from collections import Counter

import torch
from tqdm import tqdm

import nltk

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt


logger = get_logger()

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        if idx in self.idx2word:
            return self.idx2word[idx]
        else:
            return '<unk>'

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class WordTokenizer(object):
    """
    This class converts texts into character or word-level tokens
    """

    def __init__(self, maxlen=None, download_tokenizer=False):
        # Create a vocab wrapper and add some special tokens.
        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')
        self.vocab = vocab

        if download_tokenizer:
            nltk.download('punkt')

        logger.debug(f'Created WordTokenizer with init {len(self.vocab)} tokens.')

    def fit_on_files(self, txt_files):
        logger.debug('Fit on files.')
        for file in txt_files:
            logger.info(f'Updating vocab with {file}')
            sentences = read_txt(file)
            self.fit(sentences)

    def fit(self, sentences, threshold=4):
        logger.debug(
            f'Fit word vocab on {len(sentences)} and t={threshold}'
        )
        counter = Counter()

        for sentence in tqdm(sentences, total=len(sentences)):
            words = self.split_sentence(sentence)
            counter.update(words)

        # Discard if the occurrence of the word is less than threshold
        words = [
            word for word, cnt in counter.items()
            if cnt >= threshold
        ]

        # Add words to the vocabulary.
        for word in words:
            self.vocab.add_word(word)
        
        logger.info(f'Vocab built. Words found {len(self.vocab)}')
        return self.vocab

    def save(self, outpath):
        logger.debug(f'Saving vocab to {outpath}')

        with open(outpath, "w") as f:
            json.dump(self.vocab.word2idx, f)

        logger.info(
            f'Vocab stored into {outpath} with {len(self.vocab)} words.'
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
        logger.info(f'Loaded vocab containing {len(self.vocab)} words')
        return self
    
    def split_sentence(self, sentence):
        tokens = nltk.tokenize.word_tokenize(
            sentence.lower()
        )
        return tokens

    def words_to_tokens(self, words):
        return [self.vocab(word) for word in words]

    def tokenize(self, sentence):
        words = self.split_sentence(sentence)
        tokens = self.words_to_tokens(words) 
        return torch.LongTensor(tokens)

    def tokenize_sentences(self, texts):
        pass

    def decode_tokens(self, tokens):
        logger.debug(f'Decode tokens {tokens}')
        text = ' '.join([
            self.vocab.get_word(token) for token in tokens
        ])
        return text

    def __len__(self):
        return len(self.vocab)
