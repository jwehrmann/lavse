'''Neural Network Assembler and Extender'''
import types
from typing import Dict, List

import torch
from torch import nn

from torchvision import models

from ..layers import attention, convblocks
from ..utils.layers import default_initializer, l1norm, l2norm
from .common import load_state_dict_with_replace


class BaseFeatures(nn.Module):

    def __init__(self, model):
        super(BaseFeatures, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.num_features = model.fc.in_features

    def forward(self, _input):
        x = self.conv1(_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Aggregate(nn.Module):

    def __init__(self, ):
        super(Aggregate, self).__init__()

    def forward(self, _input):
        x = nn.AdaptiveAvgPool2d(1)(_input)
        x = x.squeeze(2).squeeze(2)
        return x


class FullImageEncoder(nn.Module):

    def __init__(self, cnn, img_dim, latent_size, no_imgnorm=False):
        super(FullImageEncoder, self).__init__()
        self.latent_size = latent_size
        self.no_imgnorm = no_imgnorm        
        self.fc = nn.Linear(img_dim, latent_size)

        self.apply(default_initializer)
        
        self.cnn = BaseFeatures(cnn(pretrained=True))
        # self.aggregate = Aggregate()

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.cnn(images)
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1)
        features = self.fc(features)
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

        super(FullImageEncoder, self).load_state_dict(new_state)
