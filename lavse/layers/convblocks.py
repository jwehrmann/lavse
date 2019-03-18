import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(*[
            nn.Conv1d(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                kernel_size=kwargs['kernel_size'],
                padding=kwargs['padding'],
            ),
            nn.BatchNorm1d(kwargs['out_channels']),
            nn.ReLU(),
        ])

    def forward(self,x):
        return self.conv(x)
        