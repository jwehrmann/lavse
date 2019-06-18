import torch
import torch.nn as nn


class PartialConcat(nn.Module):

    def __init__(
            self,
            num_embeddings,
            embed_dim=300,
            liwe_char_dim=24,
            liwe_neurons=[128, 256],
            liwe_dropout=0.0,
            liwe_wnorm=True,
            max_chars = 26
        ):
        super(PartialConcat, self).__init__()

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)

        self.total_embed_size = liwe_char_dim * max_chars

        layers = []
        liwe_neurons = liwe_neurons + [embed_dim]
        in_sizes = [liwe_char_dim * max_chars] + liwe_neurons

        if not liwe_wnorm:
            weight_norm = lambda x: x
        else:
            from torch.nn.utils import weight_norm

        for n, i in zip(liwe_neurons, in_sizes):
            layer = nn.Sequential(*[
                weight_norm(
                    nn.Conv1d(i, n, 1)
                ),
                nn.Dropout(liwe_dropout),
                nn.BatchNorm1d(n),
                nn.ReLU(inplace=True),
            ])
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward_embed(self, x):

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)
        # char_embed = l2norm(char_embed, 2)
        char_embed = char_embed.view(self.B, self.W, -1)
        # a, b, c = char_embed.shape
        # left = self.total_embed_size - c
        # char_embed = nn.ReplicationPad1d(left//2)(char_embed)
        char_embed = char_embed.permute(0, 2, 1)
        word_embed = self.layers(char_embed)
        return word_embed

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        self.B, self.W, self.Ct = x.size()
        return self.forward_embed(x)



class PartialGRUs(nn.Module):

    def __init__(
            self,
            num_embeddings,
            embed_dim=300,
            liwe_char_dim=24,
            **kwargs
        ):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings, liwe_char_dim)

        self.rnn = nn.GRU(liwe_char_dim, embed_dim, 1,
            batch_first=True, bidirectional=False)

    def forward_embed(self, x):

        partial_words = x.view(self.B, -1) # (B, W*Ct)
        char_embed = self.embed(partial_words) # (B, W*Ct, Cw)
        char_embed = char_embed.view(self.B*self.W, self.Ct, -1)
        x, _ = self.rnn(char_embed)

        b, t, d = x.shape
        # x = x.view(b, t, 2, d//2).mean(-2)
        x = x.max(1)[0]

        return x

    def forward(self, x):
        '''
            x: (batch, nb_words, nb_characters [tokens])
        '''
        x = x[:,:,:30]
        self.B, self.W, self.Ct = x.size()
        embed_word = self.forward_embed(x)
        embed_word = embed_word.view(self.B, self.W, -1)

        return embed_word.permute(0, 2, 1)
