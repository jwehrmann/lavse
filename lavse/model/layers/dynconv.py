import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import _VF


class MixedConv1d(nn.Module):

    def __init__(
        self, in_channels,
        query_size, out_channels,
        kernel_size, groups, padding=0,
        **kwargs
    ):
        super().__init__()

        self.conv1p = KernelProjection1d(
            in_channels=in_channels,
            query_size=query_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            **kwargs
        )
        self.conv1a = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
        )

        self.conv1 = nn.Conv1d(
            in_channels=in_channels*2,
            out_channels=out_channels,
            kernel_size=1,
        )

        # self.sa1 = attention.SelfAttention(
        #     out_channels,
        #     nn.LeakyReLU(0.1, inplace=True),
        # )

    def forward(self, x, base_feat):
        proj_x = self.conv1p(x, base_feat)
        for_x = self.conv1a(x)
        x = torch.cat([proj_x, for_x], 1)
        # x = self.batch_norm(x)
        # x = nn.ReLU(inplace=True)(x)
        x = self.conv1(x)
        # x_sa = self.sa1(x)
        return x


class KernelProjection1d(nn.Module):

    def __init__(
        self, query_size, in_channels, out_channels,
        kernel_size, padding=0, groups=1, proj_bias=True,
        weightnorm=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.proj_bias = proj_bias
        self.query_size = query_size
        self.groups = groups

        if weightnorm == None:
            weightnorm = False

        self.weightnorm = weightnorm

        self.kernel = nn.Linear(
            query_size,
            (in_channels * out_channels * kernel_size) // groups
        )
        if weightnorm == 'batchnorm':
            self.weight_norm = nn.BatchNorm1d(out_channels)
        elif weightnorm == 'softmax':
            self.weight_norm = nn.Softmax(dim=-1)
        elif weightnorm == 'l2':
            self.weight_norm = lambda x: l2norm(x, dim=1)
        if proj_bias:
            self.bias = nn.Linear(query_size, out_channels)

    def forward(self, input, base_proj):
        kernel = self.kernel(base_proj)
        kernel = kernel.view(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size
        )
        if self.weightnorm:
            kernel = self.weight_norm(
                kernel.permute(1, 0, 2)
            ).permute(1, 0, 2)
        if self.proj_bias:
            bias = self.bias(base_proj)
            bias = bias.view(self.out_channels)

        return F.conv1d(
            input, kernel, bias=bias, stride=1,
            padding=self.padding, dilation=1,
            groups=self.groups,
        )


class KernelProjection2d(nn.Module):

    def __init__(
        self, base_proj_channels, in_channels, out_channels,
        kernel_size, padding=0, groups=1, proj_bias=True,
        weightnorm='batchnorm',
    ):
        super().__init__()


        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (padding,)
        self.dilation = (1,)
        self.stride = (1,)
        self.proj_bias = proj_bias
        self.base_proj_channels = base_proj_channels
        self.groups = groups

        if weightnorm == None:
            weightnorm = False

        self.weightnorm = weightnorm

        self.kernel = nn.Linear(
            base_proj_channels,
            (in_channels * out_channels *
            kernel_size[0] * kernel_size[1]) // groups
        )

        self.kernel_shape = (
            in_channels, out_channels, *kernel_size
        )

        if weightnorm == 'batchnorm':
            self.weight_norm = nn.BatchNorm2d(out_channels)
        elif weightnorm == 'softmax':
            self.weight_norm = nn.Softmax(dim=-1)
        elif weightnorm == 'l2':
            self.weight_norm = lambda x: l2norm(x, dim=1)
        if proj_bias:
            self.bias = nn.Linear(base_proj_channels, out_channels)

    def forward(self, input, base_proj):
        kernel = self.kernel(base_proj)
        kernel = kernel.view(
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size
        )
        if self.weightnorm:
            kernel = self.weight_norm(
                kernel.permute(1, 0, 2)
            ).permute(1, 0, 2)
        if self.proj_bias:
            bias = self.bias(base_proj)
            bias = bias.view(self.out_channels)

        return F.conv2d(
            input, kernel, bias=bias, stride=1,
            padding=self.padding, dilation=1,
            groups=self.groups,
        )

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_shape={kernel_shape}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
