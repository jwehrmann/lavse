from collections import OrderedDict

import torch
import torch.nn as nn

from ...utils.layers import default_initializer
from ..similarity.measure import l1norm, l2norm
from ..layers import attention, convblocks

import numpy as np


def load_state_dict_with_replace(state_dict, own_state):
    new_state = OrderedDict()
    for name, param in state_dict.items():
        if name in own_state:
            new_state[name] = param
    return new_state


class SCANImagePrecomp(nn.Module):

    def __init__(self, img_dim, latent_size, no_imgnorm=False):
        super(SCANImagePrecomp, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, latent_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(SCANImagePrecomp, self).load_state_dict(new_state)


class VSEImageEncoder(nn.Module):

    def __init__(self, img_dim, latent_size, no_imgnorm=False):
        super(VSEImageEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, latent_size)

        self.apply(default_initializer)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        images = images.mean(1) # Global pooling
        features = self.fc(images)
        features = features.unsqueeze(1)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(VSEImageEncoder, self).load_state_dict(new_state)


class HierarchicalEncoder(nn.Module):

    def __init__(
            self, img_dim, latent_size,
            no_imgnorm=False, activation=nn.LeakyReLU(0.1),
            proj_leaky=True, embed_sa=False,
            use_sa=True,
        ):
        super(HierarchicalEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm
        self.use_sa = use_sa
        self.embed_sa = embed_sa

        projection_layers = [
            nn.Conv1d(img_dim, latent_size, 1, ),
        ]
        if proj_leaky:
            projection_layers.append(activation)

        if embed_sa:
            sa_embed = attention.SelfAttention(latent_size, activation)
            projection_layers.append(sa_embed)

        # self.fc = nn.Sequential(*projection_layers)
        self.projection = nn.Sequential(*projection_layers)

        if use_sa:
            self.sa1 = attention.SelfAttention(img_dim, activation)

        self.apply(default_initializer)

    def forward(self, images):
        """Extract image feature vectors."""
        images = images.permute(0, 2, 1)
        if self.use_sa:
            images = self.sa1(images)

        features = self.projection(images) # n, 36, 1024
        features = features.permute(0, 2, 1)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(HierarchicalEncoder, self).load_state_dict(new_state)


class ImageProj(nn.Module):

    def __init__(
            self, img_dim, latent_size,
            img_sa=True, projection=False,
            non_linear_proj=False, projection_sa=False,
        ):
        super(ImageProj, self).__init__()
        self.latent_size = latent_size

        layers = []
        if img_sa:
            layers.append(
                attention.SelfAttention(
                    in_dim=img_dim,
                    activation=nn.LeakyReLU(0.1)
                )
            )
        if projection:
            layers.append(
                nn.Conv1d(img_dim, latent_size, 1)
            )
        if non_linear_proj:
            layers.append(
                nn.LeakyReLU(0.1, inplace=True)
            )
        if projection_sa:
            layers.append(
                attention.SelfAttention(
                    in_dim=latent_size,
                    activation=nn.LeakyReLU(0.1)
                )
            )

        self.layers = nn.Sequential(*layers)

        self.apply(default_initializer)

    def forward(self, images):
        '''
        Extract image features

        Arguments:
            images {torch.FloatTensor} -- shape: (batch, regions, dims)

        Returns:
            [torch.FloatTensor] -- shape: (batch, anything, dims)
        '''

        # Permute to allow using self-attention
        # and conv projections layers
        images = images.permute(0, 2, 1)

        features = self.layers(images)
        features = features.permute(0, 2, 1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(ImageProj, self).load_state_dict(new_state)



class MultiheadAttentionEncoder(nn.Module):

    def __init__(
            self, img_dim, latent_size, dropout=0.
        ):
        super(MultiheadAttentionEncoder, self).__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Sequential(*[
            nn.Conv1d(img_dim, img_dim//4, 1,),
            nn.BatchNorm1d(img_dim//4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        ])

        self.multihead = attention.MultiHeadAttention(
            img_dim//4, h=4, k=4, r=4,
            inner_activation=nn.LeakyReLU(0.1),
            dropout=dropout
        )

        self.fc2 = nn.Sequential(*[
            nn.Conv1d(img_dim//4, latent_size, 1,),
            nn.LeakyReLU(0.1),
        ])

        self.apply(default_initializer)

    def forward(self, images):
        """Extract image feature vectors."""
        x = images.permute(0, 2, 1)
        a = self.fc1(x)
        a = self.multihead(a)
        a = self.fc2(a)
        a = a.permute(0, 2, 1)

        return a

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(MultiheadAttentionEncoder, self).load_state_dict(new_state)



class GRUImgEncoder(nn.Module):

    def __init__(
            self, img_dim, latent_size,
        ):
        super(GRUImgEncoder, self).__init__()
        self.latent_size = latent_size

        self.gru = nn.GRU(img_dim, latent_size, 1,
            batch_first=True, bidirectional=True)

        self.apply(default_initializer)

    def forward(self, images):
        """Extract image feature vectors."""

        x, _ = self.gru(images)
        b, t, d = x.shape
        x = x.view(b, t, 2, d//2).mean(-2)

        # if not self.no_txtnorm:
        #     x = l2norm(x, dim=-1)

        return x

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        new_state = load_state_dict_with_replace(
            state_dict=state_dict, own_state=self.state_dict()
        )

        super(SAGRUImgEncoder, self).load_state_dict(new_state)

