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



def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class DynamicConv1dTBC(nn.Module):
    '''Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False,
                 renorm_padding=False, bias=False, conv_bias=False,
                 query_size=None, in_proj=False):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding

        if in_proj:
            self.weight_linear = Linear(self.input_size, self.input_size + num_heads * kernel_size * 1)
        else:
            self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x, incremental_state=None, query=None, unfold=None):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''
        unfold = x.size(0) > 512 if unfold is None else unfold  # use unfold mode as default for long sequence to save memory
        unfold = unfold or (incremental_state is not None)
        assert query is None or not self.in_proj

        if query is None:
            query = x

        if unfold:
            output = self._forward_unfolded(x, incremental_state, query)
        else:
            output = self._forward_expanded(x, incremental_state, query)

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        return output

    def _forward_unfolded(self, x, incremental_state, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H*K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)

        # renorm_padding is only implemented in _forward_expanded
        assert not self.renorm_padding or incremental_state is not None

        if incremental_state is not None:
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = x.new()
            x_unfold = torch.cat([input_buffer, x.unsqueeze(3)], dim=3)
            if self.kernel_size > 1:
                self._set_input_buffer(incremental_state, x_unfold[:, :, :, -self.kernel_size+1:])
            x_unfold = x_unfold.view(T*B*H, R, -1)
        else:
            padding_l = self.padding_l
            if K > T and padding_l == K-1:
                weight = weight.narrow(1, K-T, T)
                K, padding_l = T, T-1
            # unfold the input: T x B x C --> T' x B x C x K
            x_unfold = unfold1d(x, K, padding_l, 0)
            x_unfold = x_unfold.view(T*B*H, R, K)

        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)

        if incremental_state is not None:
            weight = weight[:, -x_unfold.size(2):]
            K = weight.size(1)

        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)

        output = torch.bmm(x_unfold, weight.unsqueeze(2))  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, incremental_stat, query):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H*K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)

        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        x = x.view(T, B*H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B*H, T, T+K-1).fill_(float('-inf'))
            weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            # For efficieny, we cut the kernel size and reduce the padding when the kernel is larger than the length
            if K > T and P == K-1:
                weight = weight.narrow(2, K-T, T)
                K, P = T, T-1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(B*H, T, T+K-1, requires_grad=False)
            weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)  # B*H x T x T

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

    def extra_repr(self):
        s = '{}, kernel_size={}, padding_l={}, num_heads={}, weight_softmax={}, conv_bias={}, renorm_padding={}, in_proj={}'.format(
            self.input_size, self.kernel_size, self.padding_l,
            self.num_heads, self.weight_softmax, self.conv_bias is not None, self.renorm_padding,
            self.in_proj,
        )

        if self.query_size != self.input_size:
            s += ', query_size={}'.format(self.query_size)
        if self.weight_dropout > 0.:
            s += ', weight_dropout={}'.format(self.weight_dropout)
        return s

