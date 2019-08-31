import argparse
import sys
sys.path.append('../')
from lavse.utils.logger import create_logger
from lavse.data.tokenizer import Tokenizer
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


def loadGloveModel(gloveFile,):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    v = []
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        try:
            embedding = np.array([float(val) for val in splitLine[1:]])
        except:
            print(len(v), line)
        model[word] = embedding
        v.append(embedding)
    mean = np.array(v).mean(0)
    print(mean.shape)
    model['<unk>'] = torch.tensor(mean)
    model['<pad>'] = torch.zeros(embedding.shape)
    model['<start>'] = torch.zeros(embedding.shape)
    model['<end>'] = torch.zeros(embedding.shape)
    print("Done.",len(model)," words loaded!")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vocab_path',
    )
    parser.add_argument(
        '--glove_path',
    )
    parser.add_argument(
        '--outpath',
    )
    args = parser.parse_args()

    logger = create_logger(level='debug')

    files = []
    tokenizer = Tokenizer()
    tokenizer.load(args.vocab_path)
    nmax = max(tokenizer.vocab.idx2word.keys()) + 1

    glove = loadGloveModel(args.glove_path)
    dummy = glove['hi']
    print(dummy.shape)
    word_matrix = torch.zeros(nmax, dummy.shape[-1])
    print(word_matrix.shape)
    total_unk = 0
    for k, v in tqdm(tokenizer.vocab.idx2word.items(), total=len(tokenizer)):
        try:
            word_matrix[k] = torch.tensor(glove[v])
        except KeyError:
            word_matrix[k] = glove['<unk>']
            total_unk += 1

    print(f'Finished. Total UNK: {total_unk}')
    torch.save(word_matrix, args.outpath)
    print(f'Saved into: {args.outpath}')
