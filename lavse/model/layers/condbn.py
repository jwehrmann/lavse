import torch
import torch.nn as nn


class CondBatchNorm1d(nn.Module):

    def __init__(
        self, in_features, k, cond_vector_size=None,
        normalization='batchnorm', nonlinear_proj=False,
        groups=1,
    ):
        super().__init__()

        if normalization is None:
            self.bn = lambda x: x
        elif normalization == 'batchnorm':
            self.bn = nn.BatchNorm1d(in_features, affine=False)
        elif normalization == 'instancenorm':
            self.bn = nn.InstanceNorm1d(in_features, affine=False)

        self.nb_channels = in_features
        self.cond_vector_size = cond_vector_size
        self.groups = groups

        if cond_vector_size is None:
            cond_vector_size = in_features

        if nonlinear_proj:
            self.fc_gamma = nn.Sequential(
                nn.Linear(cond_vector_size, cond_vector_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(cond_vector_size//k, in_features),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(cond_vector_size, cond_vector_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(cond_vector_size//k, in_features),
            )
        else:
            self.fc_gamma = nn.Sequential(
                nn.Linear(cond_vector_size, in_features//groups),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(cond_vector_size, in_features//groups),
            )

            # self.fc_gamma = nn.Linear(cond_vector_size, in_features)
            # self.fc_beta = nn.Linear(cond_vector_size, in_features)

    def forward(self, feat_matrix, cond_vector):
        '''
        Forward conditional bachnorm using
        predicted gamma and beta returning
        the normalized input matrix

        Arguments:
            feat_matrix {torch.FloatTensor}
                -- shape: batch, features, timesteps
            cond_vector {torch.FloatTensor}
                -- shape: batch, features

        Returns:
            torch.FloatTensor
                -- shape: batch, features, timesteps
        '''

        B, D, _ = feat_matrix.shape
        Bv, Dv = cond_vector.shape

        norm_feat = self.bn(feat_matrix)
        norm_feat = norm_feat.view(
            B, D//self.groups, self.groups, -1
        )

        gammas = self.fc_gamma(cond_vector).view(
            Bv, Dv//self.groups, 1, 1
        )
        betas  = self.fc_beta(cond_vector).view(
            Bv, Dv//self.groups, 1, 1
        )

        normalized = norm_feat * (gammas + 1) + betas
        normalized = normalized.view(B, D, -1)
        return normalized


# class CondBatchNorm1d(nn.Module):

#     def __init__(
#         self, in_features, k, cond_vector_size=None,
#         normalization='batchnorm', nonlinear_proj=False,
#     ):
#         super().__init__()

#         if normalization is None:
#             self.bn = lambda x: x
#         elif normalization == 'batchnorm':
#             self.bn = nn.BatchNorm1d(in_features, affine=False)
#         elif normalization == 'instancenorm':
#             self.bn = nn.InstanceNorm1d(in_features, affine=False)

#         self.nb_channels = in_features
#         self.cond_vector_size = cond_vector_size

#         if cond_vector_size is None:
#             cond_vector_size = in_features

#         if nonlinear_proj:
#             self.fc_gamma = nn.Sequential(
#                 nn.Linear(cond_vector_size, cond_vector_size//k),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(cond_vector_size//k, in_features),
#             )

#             self.fc_beta = nn.Sequential(
#                 nn.Linear(cond_vector_size, cond_vector_size//k),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(cond_vector_size//k, in_features),
#             )
#         else:
#             self.fc_gamma = nn.Sequential(
#                 nn.Linear(cond_vector_size, in_features),
#             )

#             self.fc_beta = nn.Sequential(
#                 nn.Linear(cond_vector_size, in_features),
#             )

#     def forward(self, feat_matrix, cond_tensor):
#         '''
#         Forward conditional bachnorm using
#         predicted gamma and beta returning
#         the normalized input matrix

#         Arguments:
#             feat_matrix {torch.FloatTensor}
#                 -- shape: batch, features, timesteps
#             cond_vector {torch.FloatTensor}
#                 -- shape: batch, features

#         Returns:
#             torch.FloatTensor
#                 -- shape: batch, features, timesteps
#         '''

#         B, D, _ = feat_matrix.shape
#         Bv, R, Dv = cond_tensor.shape

#         gammas = self.fc_gamma(cond_tensor).view(Bv, R, Dv)
#         betas  = self.fc_beta(cond_tensor).view(Bv, R, Dv)

#         norm_feat = self.bn(feat_matrix)

#         norm_feat = norm_feat.unsqueeze(2)
#         gammas = gammas.unsqueeze(2)
#         betas = betas.unsqueeze(2)

#         norm_feat = norm_feat.permute(0, 2, 3, 1)
#         normalized = norm_feat * (gammas + 1) + betas

#         return normalized

