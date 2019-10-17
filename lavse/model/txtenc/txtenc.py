import torch
import torch.nn as nn

from ..similarity.measure import l2norm
from ...utils.layers import default_initializer

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ...model.layers import attention, convblocks
from .embedding import PartialConcat, GloveEmb, PartialConcatScale
from . import pooling

import numpy as np
import pytorch_transformers


# Default text encoder
class RNNEncoder(nn.Module):

    def __init__(
        self, tokenizers, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_type=nn.GRU):

        super(RNNEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        assert len(tokenizers) == 1
        num_embeddings = len(tokenizers[0])

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_type(
            embed_dim, latent_size, num_layers,
            batch_first=True,
            bidirectional=use_bi_gru
        )

        self.apply(default_initializer)

    def forward(self, batch):
        """Handles variable size captions
        """
        captions, lengths = batch['caption']
        captions = captions.to(self.device)

        # Embed word ids to vectors
        x = self.embed(captions)
        # Forward propagate RNN
        # self.rnn.flatten_parameters()
        cap_emb, _ = self.rnn(x)

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


# GRU Text encoder with Glove support
class GloveRNNEncoder(nn.Module):

    def __init__(
        self, tokenizers, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_type=nn.GRU, glove_path=None, add_rand_embed=False):

        super().__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        assert len(tokenizers) == 1

        num_embeddings = len(tokenizers[0])

        self.embed = GloveEmb(
            num_embeddings,
            glove_dim=embed_dim,
            glove_path=glove_path,
            add_rand_embed=add_rand_embed,
            rand_dim=embed_dim,
        )


        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_type(
            self.embed.final_word_emb,
            latent_size, num_layers,
            batch_first=True,
            bidirectional=use_bi_gru
        )

        if hasattr(self.embed, 'embed'):
            self.embed.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, batch):
        """Handles variable size captions
        """
        captions, lengths = batch['caption']
        captions = captions.to(self.device)
        # Embed word ids to vectors
        emb = self.embed(captions)

        # Forward propagate RNN
        # self.rnn.flatten_parameters()
        cap_emb, _ = self.rnn(emb)

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


class LiweGRU(nn.Module):

    def __init__(
        self,
        tokenizers, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, partial_class=PartialConcat,
        liwe_neurons=[128, 256], liwe_dropout=0.0,
        liwe_wnorm=True, liwe_char_dim=24, liwe_activation=nn.ReLU(),
        liwe_batch_norm=True,
    ):

        super(LiweGRU, self).__init__()

        assert len(tokenizers) == 1

        num_embeddings = len(tokenizers[0])

        __max_char_in_words = 30
        self.latent_size = latent_size
        self.embed_dim = embed_dim
        self.no_txtnorm = no_txtnorm

        if type(partial_class) == str:
            partial_class = eval(partial_class)

        self.embed = partial_class(
            num_embeddings=num_embeddings, embed_dim=embed_dim,
            liwe_neurons=liwe_neurons, liwe_dropout=liwe_dropout,
            liwe_wnorm=liwe_wnorm, liwe_char_dim=liwe_char_dim,
            liwe_activation=liwe_activation, liwe_batch_norm=liwe_batch_norm,
        )

        # caption embedding
        self.use_bi_gru = True

        self.rnn = nn.GRU(
            embed_dim, latent_size, num_layers,
            batch_first=True, bidirectional=True
        )

    def forward(self, batch):
        x, lens = batch['caption']
        x = x.to(self.device)
        B, W, Ct = x.size()

        word_embed = self.embed(x).contiguous()
        x = word_embed.permute(0, 2, 1).contiguous()

        # Forward propagate RNN
        cap_emb, _ = self.rnn(x)

        # Reshape *final* output to (batch_size, hidden_size)

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lens


class LiweGRUGlove(nn.Module):

    def __init__(
        self,
        tokenizers, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, partial_class=PartialConcat,
        liwe_neurons=[128, 256], liwe_dropout=0.0,
        liwe_wnorm=True, liwe_char_dim=24, liwe_activation=nn.ReLU(),
        liwe_batch_norm=True, glove_path=None
    ):

        super().__init__()

        __max_char_in_words = 30
        self.latent_size = latent_size
        self.embed_dim = embed_dim
        self.no_txtnorm = no_txtnorm


        self.glove = GloveEmb(
            len(tokenizers[0]), glove_dim=300, glove_path=glove_path,
        )

        self.embed = partial_class(
            num_embeddings=len(tokenizers[1]), embed_dim=embed_dim,
            liwe_neurons=liwe_neurons, liwe_dropout=liwe_dropout,
            liwe_wnorm=liwe_wnorm, liwe_char_dim=liwe_char_dim,
            liwe_activation=liwe_activation, liwe_batch_norm=liwe_batch_norm,
        )

        self.use_bi_gru = True

        self.rnn = nn.GRU(
            embed_dim + 300, latent_size, 1,
            batch_first=True, bidirectional=True
        )

    def forward(self, batch):

        captions = batch['caption']
        (words, wlen), (chars, clen) = captions
        B, W, Ct = chars.size()

        words = words.to(self.device)[:,1:-1]
        chars = chars.to(self.device)

        word_embed = self.embed(chars).contiguous()
        x = word_embed.permute(0, 2, 1).contiguous()

        glove = self.glove(words)
        x = torch.cat([x, glove], dim=2)

        # Forward propagate RNN
        cap_emb, _ = self.rnn(x)

        # Reshape *final* output to (batch_size, hidden_size)
        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, clen


class Bert(nn.Module):

    def __init__(
        self, latent_size, tokenizers=None):

        super(Bert, self).__init__()
        self.latent_size = latent_size
        self.bert = pytorch_transformers.BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, self.latent_size)

    def forward(self, batch):
        """Handles variable size captions
        """
        captions, lengths = batch['caption']
        captions = captions.to(self.device)
        _, sent_emb = self.bert(captions)
        out = self.fc(sent_emb)

        return out.unsqueeze(1), lengths
