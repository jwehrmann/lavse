import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    """ Self attention Layer """
    def __init__(self, in_dim, activation, k=8):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim//k,
            kernel_size=1,
        )
        self.key_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim//k,
            kernel_size=1,
        )
        self.value_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, return_attn=False):
        """
            inputs :
                x : input feature maps(B X C X T)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*T)
        """
        B, C, T = x.size()

        # B X C X (N)
        proj_query  = self.query_conv(x).view(B, -1, T).permute(0,2,1)
        # B X C x (W*H)
        proj_key =  self.key_conv(x).view(B, -1, T)
        energy =  torch.bmm(proj_query, proj_key)
        # B X (N) X (N)
        attention = self.softmax(energy)
        # B X C X N
        proj_value = self.value_conv(x).view(B, -1, T)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        out = self.gamma * out + x

        if not return_attn:
            return out

        return out, attention



class MultiHeadAttention(nn.Module):

    def __init__(
            self, input_dim, h=8,
            k=8, r=8, inner_activation=nn.Identity(),
            dropout=0.1
        ):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = h
        if inner_activation is None:
            inner_activation = nn.Identity()

        self_attentions = []
        for i in range(h):
            sa = SelfAttention(input_dim, inner_activation, k=k)
            self_attentions.append(sa)

        self.fcs = nn.Sequential(
            nn.Linear(input_dim, input_dim//r, 1),
            # nn.BatchNorm1d(input_dim//r//h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//r, input_dim // h, 1),
            # nn.BatchNorm1d(input_dim//h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.inner_size = input_dim // h

        self.self_attentions = nn.ModuleList(self_attentions)

    def forward(self, x):
        """Extract image feature vectors."""
        residual = x

        outs = []
        for sa in self.self_attentions:
            _x = sa(x)
            outs.append(_x)

        out = torch.stack(outs, 2)
        b, d, heads, regions = out.shape

        out = out.view(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
        out = self.fcs(out)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(b, self.inner_size * self.num_heads, regions)

        out = out + residual

        return out


class ModuleSelfAttention(nn.Module):

    """ Self attention Layer """
    def __init__(
        self, module, in_dim,
        activation, groups=1, k=8,
        **kwargs
    ):
        super(ModuleSelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = module(
            in_channels=in_dim,
            out_channels=in_dim//k,
            kernel_size=1,
            groups=groups,
            **kwargs,
        )
        self.key_conv = module(
            in_channels=in_dim,
            out_channels=in_dim//k,
            kernel_size=1,
            groups=groups,
            **kwargs,
        )
        self.value_conv = module(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1,
            groups=groups,
            **kwargs,
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, base, return_attn=False):
        """
            inputs :
                x : input feature maps(B X C X T)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*T)
        """
        B, C, T = x.size()

        # B X C X (N)
        proj_query  = self.query_conv(x, base).view(B, -1, T).permute(0,2,1)
        # B X C x (W*H)
        proj_key =  self.key_conv(x, base).view(B, -1, T)
        energy =  torch.bmm(proj_query, proj_key)
        # B X (N) X (N)
        attention = self.softmax(energy)
        # B X C X N
        proj_value = self.value_conv(x, base).view(B, -1, T)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        out = self.gamma * out + x

        if not return_attn:
            return out

        return out, attention
