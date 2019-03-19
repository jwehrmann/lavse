from collections import OrderedDict

import torch
import torch.nn as nn

from ..utils.layers import default_initializer, l1norm, l2norm
from ..layers import attention, convblocks


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

        self.apply(default_initializer)

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
