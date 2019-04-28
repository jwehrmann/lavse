import torch
import torch.nn as nn

from ..utils.layers import default_initializer, l2norm

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ..layers import attention, convblocks
from .embedding import PartialConcat

import numpy as np


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, num_embeddings, embed_dim, latent_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(embed_dim, latent_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors

        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (
                cap_emb[:,:,:cap_emb.size(2)//2] + cap_emb[:,:,cap_emb.size(2)//2:]
            )/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len

# RNN Based Language Model
class RNNEncoder(nn.Module):

    def __init__(
        self, num_embeddings, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_type=nn.GRU):

        super(RNNEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

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

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        # Forward propagate RNN
        cap_emb, _ = self.rnn(x)

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


class WordEmbeddingProj(nn.Module):

    def __init__(
        self, num_embeddings, embed_dim,
        latent_size, word_sa=True, projection=False,
        non_linear_proj=False, projection_sa=False,
    ):

        super(WordEmbeddingProj, self).__init__()
        self.latent_size = latent_size

        # word embedding
        layers = []
        self.emb = nn.Embedding(num_embeddings, embed_dim)
        if word_sa:
            layers.append(
                attention.SelfAttention(
                    in_dim=embed_dim,
                    activation=nn.LeakyReLU(0.1)
                )
            )
        if projection:
            layers.append(
                nn.Conv1d(embed_dim, latent_size, 1)
            )
        if non_linear_proj:
            layers.append(
                nn.LeakyReLU(0.1, inplace=True)
            )
        if projection_sa:
            layers.append(
                attention.SelfAttention(
                    in_dim=latent_size,
                    activation=nn.LeakyReLU(0.1),
                )
            )

        self.layers = nn.Sequential(*layers)

        self.apply(default_initializer)

    def forward(self, captions, lengths):
        '''
        Extract text features

        Arguments:
            images {torch.FloatTensor} -- shape: (batch, timesteps, dims)

        Returns:
            [torch.FloatTensor] -- shape: (batch, [timesteps, or 1], dims)
        '''
        x = self.emb(captions)
        # Embed word ids to vectors
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = x.permute(0, 2, 1)

        return x, lengths


class SelfAttnGRU(nn.Module):

    def __init__(
        self, num_embeddings, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, activation=nn.LeakyReLU(0.1),
    ):

        super(SelfAttnGRU, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_cell(
            embed_dim, latent_size, num_layers,
            batch_first=True, bidirectional=use_bi_gru
        )

        self.sa1 = attention.SelfAttention(latent_size, activation)

        self.fc = nn.Sequential(*[
            nn.Conv1d(1024, latent_size, 1,),
            # nn.LeakyReLU(0.1, ),
        ])

        # self.init_weights()
        self.apply(default_initializer)

    # def init_weights(self):
    #     self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        b, t, e = x.shape
        # Forward propagate RNN
        # packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        cap_emb, _ = self.rnn(x)

        # Reshape *final* output to (batch_size, hidden_size)
        # padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb, cap_len = padded

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        cap_emb = cap_emb.permute(0, 2, 1)
        cap_emb = self.sa1(cap_emb)

        cap_emb = self.fc(cap_emb)
        cap_emb = cap_emb.permute(0, 2, 1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


class SelfAttnGRUWordCat(nn.Module):

    def __init__(
        self, num_embeddings, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, activation=nn.LeakyReLU(0.1),
    ):

        super(SelfAttnGRUWordCat, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed_sa = attention.SelfAttention(embed_dim, activation)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_cell(
            embed_dim, latent_size, num_layers,
            batch_first=True, bidirectional=use_bi_gru
        )

        self.sa1 = attention.SelfAttention(latent_size, activation)

        self.fc = nn.Sequential(*[
            nn.Conv1d(latent_size+embed_dim, latent_size, 1,),
            # nn.LeakyReLU(0.1, ),
        ])

        # self.init_weights()
        self.apply(default_initializer)

    # def init_weights(self):
    #     self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        b, t, e = x.shape
        # Forward propagate RNN
        # packed = pack_padded_sequence(x, lengths, batch_first=True)
        xt = x.permute(0, 2, 1)
        wsa = self.embed_sa(xt)

        # Forward propagate RNN
        cap_emb, _ = self.rnn(x)

        # Reshape *final* output to (batch_size, hidden_size)
        # padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb, cap_len = padded

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        cap_emb = cap_emb.permute(0, 2, 1)
        cap_emb = self.sa1(cap_emb)

        cap_emb = torch.cat([cap_emb, wsa])

        cap_emb = self.fc(cap_emb)
        cap_emb = cap_emb.permute(0, 2, 1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


class SelfAttn(nn.Module):

    def __init__(
        self, num_embeddings, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, activation=nn.LeakyReLU(0.1),
    ):

        super(SelfAttn, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)

        self.conv0 = nn.ConvBlock(
            in_channels=embed_dim,
            out_channel=latent_size,
            kernel_size=1,
            padding=0,
        )

        self.sa0 = attention.SelfAttention(latent_size, activation)

        self.conv1 = convblocks.ParallelBlock(
            in_channels=embed_dim,
            out_channels=[512, 512],
            kernel_sizes=[1, 2,],
            paddings=[0, 1,],
        )
        self.sa1 = attention.SelfAttention(latent_size, activation)

        self.conv2 = convblocks.ParallelBlock(
            in_channels=latent_size,
            out_channels=[512, 512],
            kernel_sizes=[1, 2,],
            paddings=[0, 1,],
        )
        self.sa2 = attention.SelfAttention(latent_size, activation)

        self.conv3 = convblocks.ParallelBlock(
            in_channels=latent_size,
            out_channels=[512, 512],
            kernel_sizes=[1, 2],
            paddings=[0, 1,],
        )
        self.projection = nn.Conv1d(latent_size*3, latent_size, 1)
        # self.sa3 = attention.SelfAttention(latent_size, activation)

        # self.init_weights()
        self.apply(default_initializer)

    def forward(self, x, lengths):
        # Embed word ids to vectors
        x = self.embed(x)
        b, t, e = x.shape
        x = x.permute(0, 2, 1)
        # Forward propagate RNN
        # packed = pack_padded_sequence(x, lengths, batch_first=True)
        # print(x.shape)
        a = self.sa1(self.conv1(x))
        # print(x.shape)

        b = self.sa2(self.conv2(a))
        # print(x.shape)
        # b = b + a

        c = self.conv3(b)
        # x = c + b
        x = torch.cat([a, b, c], dim=1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            x = l2norm(x, dim=-1)

        return x, lengths


class ConvGRU(nn.Module):

    def __init__(
        self, num_embeddings, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, activation=nn.LeakyReLU(0.1),
    ):

        super(ConvGRU, self).__init__()
        self.latent_size = latent_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(num_embeddings, embed_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = rnn_cell(
            embed_dim, latent_size, num_layers,
            batch_first=True, bidirectional=use_bi_gru)

        self.conv1 = convblocks.ConvBlock(
            in_channels=embed_dim,
            out_channels=latent_size,
            kernel_size=1,
            padding=0,
        )
        self.conv2 = convblocks.ConvBlock(
            in_channels=embed_dim,
            out_channels=latent_size,
            kernel_size=2,
            padding=1,
        )
        self.conv3 = convblocks.ConvBlock(
            in_channels=embed_dim,
            out_channels=latent_size,
            kernel_size=3,
            padding=2,
        )
        self.sa1 = attention.SelfAttention(latent_size, activation)
        self.sa2 = attention.SelfAttention(latent_size, activation)
        self.sa3 = attention.SelfAttention(latent_size, activation)
        self.sa4 = attention.SelfAttention(latent_size, activation)

        # self.fc = nn.Linear(
        #     1024*4,
        #     latent_size
        # )

        self.fc = nn.Sequential(*[
            nn.Conv1d(1024*4, latent_size, 1, ),
            nn.LeakyReLU(0.1, ),
        ])

        # self.init_weights()
        self.apply(default_initializer)

    # def init_weights(self):
    #     self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        b, t, e = x.shape
        # Forward propagate RNN
        # packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        cap_emb, _ = self.rnn(x)

        # Reshape *final* output to (batch_size, hidden_size)
        # padded = pad_packed_sequence(out, batch_first=True)
        # cap_emb, cap_len = padded

        if self.use_bi_gru:
            b, t, d = cap_emb.shape
            cap_emb = cap_emb.view(b, t, 2, d//2).mean(-2)

        cap_emb = cap_emb.permute(0, 2, 1)
        d = self.sa1(cap_emb)

        xt = x.permute(0, 2, 1)
        a = self.conv1(xt)[:,:,:t]
        b = self.conv2(xt)[:,:,:t]
        c = self.conv3(xt)[:,:,:t]

        a = self.sa2(a)
        b = self.sa3(b)
        c = self.sa4(c)

        cap_emb = torch.cat([a, b, c, d], 1)
        cap_emb = self.fc(cap_emb)
        cap_emb = cap_emb.permute(0, 2, 1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, lengths


class LiweGRU(nn.Module):

    def __init__(
        self,
        num_embeddings, embed_dim, latent_size,
        num_layers=1, use_bi_gru=True, no_txtnorm=False,
        rnn_cell=nn.GRU, partial_class=PartialConcat,
        liwe_neurons=[128, 256], liwe_dropout=0.0,
        liwe_wnorm=True, liwe_char_dim=24,
    ):

        super(LiweGRU, self).__init__()

        __max_char_in_words = 30
        self.latent_size = latent_size
        self.embed_dim = embed_dim
        self.no_txtnorm = no_txtnorm

        self.embed = partial_class(
            num_embeddings=num_embeddings, embed_dim=embed_dim,
            liwe_neurons=liwe_neurons, liwe_dropout=liwe_dropout,
            liwe_wnorm=liwe_wnorm, liwe_char_dim=liwe_char_dim,
        )

        # caption embedding
        self.use_bi_gru = True
        self.rnn = nn.GRU(
            embed_dim, latent_size, 1,
            batch_first=True, bidirectional=True
        )

    def forward(self, x, lens=None):

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
