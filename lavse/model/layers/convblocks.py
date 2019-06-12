import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self, 
        batchnorm=True, 
        activation=nn.LeakyReLU(0.1, inplace=True),
        **kwargs
    ):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(
            nn.Conv1d(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                kernel_size=kwargs['kernel_size'],
                padding=kwargs['padding'],
            )
        )
        
        if batchnorm:            
            layers.append(nn.BatchNorm1d(kwargs['out_channels']))            
        if activation is not None:
            layers.append(activation)
            
        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        return self.conv(x)


class ParallelBlock(nn.Module):

    def __init__(
        self, in_channels, kernel_sizes, 
        out_channels, paddings, 
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            ConvBlock(
                in_channels=in_channels, 
                out_channels=o, 
                kernel_size=k, 
                padding=p,
            ) 
            for k, p, o 
            in zip(kernel_sizes, paddings, out_channels)
        ])
    
    def forward(self, x):
        
        outs = [
            conv(x) for conv in self.convs
        ]
        t = min([out.shape[-1] for out in outs])
        outs = torch.cat([out[:,:,:t] for out in outs], dim=1)
        return outs
      
