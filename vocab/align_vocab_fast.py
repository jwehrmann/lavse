import argparse
import sys
sys.path.append('../')
from lavse.utils.logger import create_logger
from lavse.data.tokenizer import Tokenizer
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embed = np.array(list(map(float, tokens[1:])))
        data[tokens[0]] = embed

    data['<unk>'] = torch.zeros(embed.shape)
    data['<pad>'] = torch.zeros(embed.shape)
    data['<start>'] = torch.zeros(embed.shape)
    data['<end>'] = torch.zeros(embed.shape)
    return data


# def loadGloveModel(file,):
#     print("Loading FastText Model")
#     f = load_vectors(file)
#     model = {}
#     v = []
#     for word, embedding in f.items():
#         print(word, embedding.shape)
#     # mean = np.array(v).mean(0)
#     # print(mean.shape)
#     model['<unk>'] = torch.zeros(embedding.shape)
#     model['<pad>'] = torch.zeros(embedding.shape)
#     model['<start>'] = torch.zeros(embedding.shape)
#     model['<end>'] = torch.zeros(embedding.shape)
#     print("Done.",len(model)," words loaded!")
#     return model


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

    glove = load_vectors(args.glove_path)
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
