import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import _VF


class MixedConv1d(nn.Module):

    def __init__(
        self, base_proj_channels,
        in_channels, out_channels,
        kernel_size, groups,
    ):
        super().__init__()

        self.conv1p = ProjConv1d(
            base_proj_channels=base_proj_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
        )
        self.conv1a = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
        )

        self.batch_norm = nn.BatchNorm1d(out_channels*2)

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
        x = self.batch_norm(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv1(x)
        # x_sa = self.sa1(x)
        return x


class MixedConv2d(nn.Module):

    def __init__(
        self, base_proj_channels,
        in_channels, out_channels,
        kernel_size, groups,
    ):
        super().__init__()

        self.conv1p = ProjConv2d(
            base_proj_channels=base_proj_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
        )
        self.conv1a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
        )

        self.batch_norm = nn.BatchNorm2d(out_channels*2)

        self.conv1 = nn.Conv2d(
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
        x = self.batch_norm(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv1(x)
        # x_sa = self.sa1(x)
        return x


class ProjConv1d(nn.Module):

    def __init__(
        self, base_proj_channels, in_channels, out_channels,
        kernel_size, padding=0, groups=1, proj_bias=True,
        weightnorm='batchnorm',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.proj_bias = proj_bias
        self.base_proj_channels = base_proj_channels
        self.groups = groups

        if weightnorm == None:
            weightnorm = False

        self.weightnorm = weightnorm

        self.kernel = nn.Linear(
            base_proj_channels,
            (in_channels * out_channels * kernel_size) // groups
        )
        if weightnorm == 'batchnorm':
            self.weight_norm = nn.BatchNorm1d(out_channels)
        elif weightnorm == 'softmax':
            self.weight_norm = nn.Softmax(dim=-1)
        elif weightnorm == 'l2':
            self.weight_norm = lambda x: l2norm(x, dim=1)
        if proj_bias:
            self.bias = nn.Linear(base_proj_channels, out_channels)

        print(self.kernel)

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


class ProjConv2d(nn.Module):

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


class ProjRNN(nn.Module):

    def __init__(
        self, base_proj_channels, rnn_input, rnn_units, device
    ):
        super().__init__()

        self.base_proj_channels = base_proj_channels
        self.rnn_units = rnn_units
        self.weights_dim = rnn_units * 3
        self.device = device
        self.rnn_input = rnn_input

        self.weight_ih = nn.Linear(base_proj_channels, self.weights_dim * rnn_input)
        self.weight_hh = nn.Linear(base_proj_channels, self.weights_dim * rnn_units)
        self.bias_ih = nn.Linear(base_proj_channels, self.weights_dim)
        self.bias_hh = nn.Linear(base_proj_channels, self.weights_dim)

        print(self.weight_ih)
        print(self.weight_hh)
        print(self.bias_ih)
        print(self.bias_hh)

    def forward(self, input, base_proj):
        b, t, d = input.shape

        weight_ih = self.weight_ih(base_proj)
        weight_ih = weight_ih.view(
            self.weights_dim, self.rnn_input
        )
        weight_hh = self.weight_hh(base_proj)
        weight_hh = weight_hh.view(
            self.weights_dim, self.rnn_units
        )
        bias_ih = self.bias_ih(base_proj).view(-1)
        bias_hh = self.bias_hh(base_proj).view(-1)

        hx = torch.zeros(b,  self.rnn_units).to(self.device)

        outputs = []
        for i in range(t):
            input_vec = input[:,i]
            out = _VF.gru_cell(
                input_vec, hx,
                weight_ih, weight_hh,
                bias_ih, bias_hh,
            )
            outputs.append(out)
        outputs = torch.stack(outputs, 0)
        outputs = outputs.permute(1, 0, 2)
        return outputs

