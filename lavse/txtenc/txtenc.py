import torch 
import torch.nn as nn

from ..utils.layers import default_initializer, l2norm

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ..layers import attention, convblocks

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
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

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