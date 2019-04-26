from timeit import default_timer as dt

import torch
from addict import Dict
from torch import nn
from torch.nn import functional as F

from .layers import attention
from .loss import cosine_sim
from .txtenc.pooling import mean_pooling
from .utils.layers import l2norm
from .utils.logger import get_logger
from .utils import helper

from tqdm import tqdm

logger = get_logger()


class XAttn(nn.Module):

    def __init__(self, dim=1024, k=8, smooth=4):
        super().__init__()
        self.dim = dim
        self.k = k
        self.gamma = nn.Parameter(torch.zeros(1))
        self.smooth = smooth

    def forward(self, query_a, value_a, key_b, value_b):
        '''
            When it is a caption:
                query_a: torch.Size([128, 4, 128])
                value_a: torch.Size([128, 1024, 4])
                key_b  : torch.Size([128, 128, 36])
                x attn : torch.Size([128, 4, 36])
                output : torch.Size([128, 1024, 36])

            When it is an image:
                query_a: torch.Size([128, 36, 128])
                value_a: torch.Size([128, 1024, 36])
                key_b  : torch.Size([128, 128, 4])
                x attn : torch.Size([128, 36, 4])
                output : torch.Size([128, 1024, 4])

        '''
        # print('\n\n\n')
        # print('query_a:', query_a.shape)
        # print('value_a:', value_a.shape)
        # print('key_b  :', key_b.shape)

        energy_cross = torch.bmm(query_a, key_b)
        cross_attention = nn.Softmax(dim=-1)(energy_cross * self.smooth)

        output = torch.bmm(value_a, cross_attention)
        output = self.gamma + output + value_b

        return output


class CrossAttn(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=8
        ):
        super().__init__()

        self.device = device

        self.query_conv_img = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_size,
                out_channels=latent_size//k,
                kernel_size=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.key_conv_img = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_size,
                out_channels=latent_size//k,
                kernel_size=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.value_conv_img = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_size,
                out_channels=latent_size,
                kernel_size=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


        self.query_conv_txt = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_size,
                out_channels=latent_size//k,
                kernel_size=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.key_conv_txt = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_size,
                out_channels=latent_size//k,
                kernel_size=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.value_conv_txt = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_size,
                out_channels=latent_size,
                kernel_size=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.gamma_img = nn.Parameter(torch.zeros(1))
        self.gamma_txt = nn.Parameter(torch.zeros(1))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        # B, 1024, 36
        img_embed = img_embed.permute(0, 2, 1).to(self.device)
        cap_embed = cap_embed.permute(0, 2, 1).to(self.device)

        # B, 36, 128
        query_img = self.query_conv_img(img_embed).permute(0, 2, 1)
        # B, 128, 36
        key_img = self.key_conv_img(img_embed)
        # B, 36, 36
        # energy_img =  torch.bmm(query_img.permute(0, 2, 1), key_img)

        # B, T, 128
        query_txt = self.query_conv_img(cap_embed).permute(0, 2, 1)
        # B, 128, T
        key_txt = self.key_conv_img(cap_embed)

        # B, 1024, 36
        value_img = self.value_conv_img(img_embed)
        # B, 1024, T
        value_txt = self.value_conv_txt(cap_embed)

        sims = torch.zeros(img_embed.shape[0], cap_embed.shape[0]).to(self.device)
        for i, cap in enumerate(cap_embed):
            n_words = lens[i]
            # cap: 1024, T
            cap = cap[:,:n_words]
            query_t = query_txt[i][:n_words]
            query_t = torch.stack([query_t] * len(key_img), 0)
            # energy   : B, T, 36
            energy_cross = torch.bmm(query_t, key_img) * self.alpha + self.beta
            cross_attention = self.softmax(energy_cross)
            value_t = torch.stack([value_txt[i][:,:n_words]] * len(key_img), 0)

            img_output = torch.bmm(value_t, cross_attention)
            img_output = self.gamma_img * img_output + value_img

            img_vector = img_output.mean(-1)
            img_vector = l2norm(img_vector, dim=-1)
            cap_vector = cap[:,:n_words].mean(1)
            cap_vector = l2norm(cap_vector, dim=-1)
            sim = cosine_sim(img_vector, cap_vector.unsqueeze(0)).squeeze(-1)
            sims[:,i] = sim

        return sims


class CondBatchNorm1d(nn.Module):

    def __init__(self, in_features, k):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_features, affine=False)

        self.fc_gamma = nn.Sequential(
            nn.Linear(in_features, in_features//k),
            nn.ReLU(inplace=True),
            nn.Linear(in_features//k, in_features),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(in_features, in_features//k),
            nn.ReLU(inplace=True),
            nn.Linear(in_features//k, in_features),
        )

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

        B, D = cond_vector.shape

        gammas = self.fc_gamma(cond_vector).view(B, D, 1)
        betas  = self.fc_beta(cond_vector).view(B, D, 1)

        norm_feat = self.bn(feat_matrix)
        normalized = norm_feat * (gammas + 1) + betas
        return normalized


class AdaptiveEmbedding(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=8, norm=False, task='t2i'
        ):
        super().__init__()

        self.device = device

        # self.fc = nn.Conv1d(latent_size, latent_size*2, 1).to(device)

        self.cbn_img = CondBatchNorm1d(latent_size, k)
        self.cbn_txt = CondBatchNorm1d(latent_size, k)

        # self.alpha = nn.Parameter(torch.ones(1))
        # self.beta = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.norm = norm
        if norm:
            self.feature_norm = ClippedL2Norm()
        self.task = task

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        cap_embed = cap_embed.permute(0, 2, 1).to(self.device)
        img_embed = img_embed.permute(0, 2, 1).to(self.device)

        # (B, 1024)
        if self.norm:
            cap_embed = self.feature_norm(cap_embed)
            img_embed = self.feature_norm(img_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        for i, cap_tensor in enumerate(cap_embed):
            # cap: 1024, T
            # img: 1024, 36

            if self.task == 't2i':
                n_words = lens[i]
                cap_repr = cap_tensor[:,:n_words].mean(-1).unsqueeze(0)

                img_output = self.cbn_img(img_embed, cap_repr)
                img_vector = img_output.mean(-1)

                img_vector = l2norm(img_vector, dim=-1)
                cap_vector = cap_repr
                cap_vector = l2norm(cap_vector, dim=-1)

                sim = cosine_sim(img_vector, cap_vector).squeeze(-1)

            if self.task == 'i2t':
                img_vectors = img_embed.mean(-1)
                cap_i_expand = cap_tensor.repeat(img_vectors.shape[0], 1, 1)
                txt_output = self.cbn_txt(cap_i_expand, img_vectors).mean(-1, keepdim=True)

                sim = cosine_similarity(
                    img_vectors.unsqueeze(2), txt_output, 1,
                )

            sims[:,i] = sim


        return sims


class AdaptiveEmbeddingI2T(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=8, norm=False,
        ):
        super().__init__()

        self.device = device

        # self.fc = nn.Conv1d(latent_size, latent_size*2, 1).to(device)

        self.cbn_img = CondBatchNorm1d(latent_size, k)
        self.cbn_txt = CondBatchNorm1d(latent_size, k)

        # self.alpha = nn.Parameter(torch.ones(1))
        # self.beta = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.norm = norm
        if norm:
            self.feature_norm = ClippedL2Norm()

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        cap_embed = cap_embed.permute(0, 2, 1).to(self.device)
        img_embed = img_embed.permute(0, 2, 1).to(self.device)
        # print('cap_embed', cap_embed.shape)
        # print('img_embed', img_embed.shape)

        # (B, 1024)
        if self.norm:
            cap_embed = self.feature_norm(cap_embed)
            img_embed = self.feature_norm(img_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        for i, img_tensor in enumerate(img_embed):
            # cap: 1024, T
            # img: 1024, 36
            img_repr = img_tensor.mean(-1).unsqueeze(0)

            txt_output = self.cbn_txt(cap_embed, img_repr)
            txt_vector = mean_pooling(txt_output.permute(0, 2, 1), lens)
            # print('txt vector', txt_vector.shape)
            txt_vector = l2norm(txt_vector, dim=-1)
            img_vector = img_repr
            img_vector = l2norm(img_vector, dim=-1)
            # print('txt vector -- ', txt_vector.shape)
            # print('img_vector: ', img_vector.shape)
            sim = cosine_sim(img_vector, txt_vector).squeeze(-1)
            # print('sim', sim.shape)
            sims[:,i] = sim

        return sims


class RNNProj(nn.Module):

    def __init__(
            self, device, latent_size=1024, rnn_units=256, norm=False,
            num_layers=1, bidirectional=True,
        ):
        super().__init__()

        self.device = device
        self.rnn = nn.GRU(
            latent_size, rnn_units, num_layers,
            batch_first=True, bidirectional=bidirectional,
        )
        import numpy as np
        total  = 0
        for k, v in self.rnn.named_parameters():
            c = np.prod(v.shape).astype(np.int)
            total += c
            print(f'{k:45s}: {c:6d} {v.shape}')
        print('total ', total)
        self.sa = attention.SelfAttention(rnn_units, activation=nn.LeakyReLU(0.1))
        print()
        total  = 0
        for k, v in self.sa.named_parameters():
            c = np.prod(v.shape).astype(np.int)
            total += c
            print(f'{k:45s}: {c:6d} {v.shape}')
        print('total', total)
        exit()

    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        cap_embed = cap_embed.permute(0, 2, 1).to(self.device)
        img_embed = img_embed.permute(0, 2, 1).to(self.device)

        # (B, 1024)
        if self.norm:
            cap_embed = self.feature_norm(cap_embed)
            img_embed = self.feature_norm(img_embed)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        for i, cap_tensor in enumerate(cap_embed):
            # cap: 1024, T
            # img: 1024, 36

            if self.task == 't2i':
                n_words = lens[i]
                cap_repr = cap_tensor[:,:n_words].mean(-1).unsqueeze(0)

                img_output = self.cbn_img(img_embed, cap_repr)
                img_vector = img_output.mean(-1)

                img_vector = l2norm(img_vector, dim=-1)
                cap_vector = cap_repr
                cap_vector = l2norm(cap_vector, dim=-1)

                sim = cosine_sim(img_vector, cap_vector).squeeze(-1)

            if self.task == 'i2t':
                img_vectors = img_embed.mean(-1)
                cap_i_expand = cap_tensor.repeat(img_vectors.shape[0], 1, 1)
                txt_output = self.cbn_txt(cap_i_expand, img_vectors).mean(-1, keepdim=True)

                sim = cosine_similarity(
                    img_vectors.unsqueeze(2), txt_output, 1,
                )

            sims[:,i] = sim


        return sims


class ExcEmbedding(nn.Module):

    def __init__(
            self, device, latent_size=1024, k=8,
        ):
        super().__init__()

        self.device = device

        self.squeeze = nn.Sequential(*[
            nn.Linear(latent_size, latent_size//k),
            # nn.BatchNorm1d(latent_size//k),
            nn.ReLU(inplace=True),
        ])

        self.excite = nn.Sequential(*[
            nn.Linear(latent_size//k, latent_size),
            nn.Sigmoid(),
        ])

        self.beta = nn.Parameter(torch.ones(1))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.norm = ClippedL2Norm()


    def forward(self, img_embed, cap_embed, lens, **kwargs):
        '''
            img_embed: (B, 36, latent_size)
            cap_embed: (B, T, latent_size)
        '''
        # (B, 1024, T)
        cap_embed = cap_embed.permute(0, 2, 1).to(self.device)
        img_embed = img_embed.permute(0, 2, 1).to(self.device)

        cap_embed = self.norm(cap_embed)
        img_embed = self.norm(img_embed)

        img_vectors = img_embed.mean(-1)

        sims = torch.zeros(
            img_embed.shape[0], cap_embed.shape[0]
        ).to(self.device)

        for i, cap in enumerate(cap_embed):
            n_words = lens[i]
            # cap: 1024, T
            # img: 1024, 36
            cap = cap[:,:n_words].mean(-1).unsqueeze(0)
            gate = self.excite(self.squeeze(cap))
            img_vector = img_vectors * gate + self.beta

            img_vector = l2norm(img_vector, dim=-1)
            cap_vector = cap
            cap_vector = l2norm(cap_vector, dim=-1)

            sim = cosine_sim(img_vector, cap_vector).squeeze(-1)
            sims[:,i] = sim

        return sims


class Cosine(nn.Module):

    def __init__(self, device, latent_size=1024):
        super().__init__()
        self.device = device

    def forward(self, img_embed, cap_embed, *args, **kwargs):
        img_embed = l2norm(img_embed, dim=1)
        cap_embed = l2norm(cap_embed, dim=1)
        return cosine_sim(img_embed, cap_embed)


class Similarity(nn.Module):

    def __init__(self, device, similarity_name='cosine', **kwargs):
        super().__init__()
        self.device = device
        self.similarity = get_similarity_object(similarity_name, device=device, **kwargs)
        logger.info(f'Created similarity: {similarity_name} with fn: {self.similarity}')
        self.pbar = tqdm(total=0, desc='Test  ')

    def forward(self, img_embed, cap_embed, lens, shared=False):
        logger.debug(f'Similarity - img_shape: {img_embed.shape} cap_shape: {cap_embed.shape}')
        return self.similarity(img_embed, cap_embed, lens)

    def forward_shared(self, img_embed, cap_embed, lens, shared_size=128):
        """
        Computer pairwise i2t image-caption distance with locality sharding
        """

        img_embed = img_embed.to(self.device)
        cap_embed = cap_embed.to(self.device)

        n_im_shard = (len(img_embed)-1)//shared_size + 1
        n_cap_shard = (len(cap_embed)-1)//shared_size + 1

        logger.debug('Calculating shared similarities')

        self.pbar = helper.reset_pbar(self.pbar)
        self.pbar.total = (n_im_shard * n_cap_shard) + 1

        d = torch.zeros(len(img_embed), len(cap_embed)).to(self.device)
        for i in range(n_im_shard):
            im_start, im_end = shared_size*i, min(shared_size*(i+1), len(img_embed))
            for j in range(n_cap_shard):
                cap_start, cap_end = shared_size*j, min(shared_size*(j+1), len(cap_embed))
                im = img_embed[im_start:im_end]
                s = cap_embed[cap_start:cap_end]
                l = lens[cap_start:cap_end]
                sim = self.forward(im, s, l)

                d[im_start:im_end, cap_start:cap_end] = sim
                self.pbar.update(1)

        logger.debug('Done computing shared similarities.')
        return d


class LogSumExp(nn.Module):
    def __init__(self, lambda_lse):
        self.lambda_lse = lambda_lse

    def forward(self, x):
        x.mul_(self.lambda_lse).exp_()
        x = x.sum(dim=1, keepdim=True)
        x = torch.log(x)/self.lambda_lse
        return x


class ClippedL2Norm(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return l2norm(self.leaky(x), 2)


class StackedAttention(nn.Module):

    def __init__(
        self, i2t=True, agg_function='Mean',
        feature_norm='softmax', lambda_lse=None,
        smooth=4, **kwargs,
    ):
        super().__init__()
        self.i2t = i2t
        self.lambda_lse = lambda_lse
        self.agg_function = agg_function
        self.feature_norm = feature_norm
        self.lambda_lse = lambda_lse
        self.smooth = smooth
        self.kwargs = kwargs

        self.attention = Attention(
            smooth=smooth, feature_norm=feature_norm,
        )

        if agg_function == 'LogSumExp':
            self.aggregate_function = LogSumExp(lambda_lse)
        elif agg_function == 'Max':
            self.aggregate_function = lambda x: x.max(dim=1, keepdim=True)[0]
        elif agg_function == 'Sum':
            self.aggregate_function = lambda x: x.sum(dim=1, keepdim=True)
        elif agg_function == 'Mean':
            self.aggregate_function = lambda x: x.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_function))

        self.task = 'i2t' if i2t else 't2i'

    def forward(self, images, captions, cap_lens, ):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d) or (n_image, n_region, d)
                attn: (n_image, n_region, n_word)
            """
            emb_a = cap_i_expand
            emb_b = images
            if self.i2t:
                emb_a = images
                emb_b = cap_i_expand

            weiContext, attn = self.attention(emb_a, emb_b)
            emb_a = emb_a.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            row_sim = cosine_similarity(emb_a, weiContext, dim=2)
            row_sim = self.aggregate_function(row_sim)
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities

    def __repr__(self, ):
        return (
            f'StackedAttention(task: {self.task},'
            f'i2t: {self.i2t}, '
            f'attention: {self.attention}, '
            f'lambda_lse: {self.lambda_lse}, '
            f'agg_function: {self.agg_function}, '
            f'feature_norm: {self.feature_norm}, '
            f'lambda_lse: {self.lambda_lse}, '
            f'smooth: {self.smooth}, '
            f'kwargs: {self.kwargs})'
        )


class RegionCorr(nn.Module):

    def __init__(
            self, **kwargs,
        ):
        super().__init__()
        self.normalize_attn = ClippedL2Norm()
        pass

    def forward(self, images, captions, cap_lens, ):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        images = self.normalize_attn(images)

        image_vec = images.mean(1)
        caption_vec = torch.stack([
            x[:l].mean(0) for x, l in zip(captions, cap_lens)
        ], 0)

        img_embed = l2norm(image_vec, dim=1)
        cap_embed = l2norm(caption_vec, dim=1)
        sim_matrix = cosine_sim(img_embed, cap_embed)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = self.normalize_attn(cap_i)
            cap_i_expand = cap_i_expand.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d) or (n_image, n_region, d)
                attn: (n_image, n_region, n_word)
            """
            # (n_image, 36, d) x (n_image, d, words) -> n_image, 36, words
            reg_sims = images.bmm(cap_i_expand.permute(0, 2, 1))
            row_sim = reg_sims.mean(-1).mean(-1).unsqueeze(1)
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        similarities = (similarities + sim_matrix) / 2.

        return similarities


def attn_softmax(attn):
    batch_size, sourceL, queryL = attn.shape
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)
    # --> (batch, sourceL, queryL)
    attn = attn.view(batch_size, sourceL, queryL)
    return attn


class Attention(nn.Module):

    def __init__(self, smooth, feature_norm='softmax'):
        super().__init__()
        self.smooth = smooth
        self.feature_norm = feature_norm

        if feature_norm == "softmax":
            self.normalize_attn = attn_softmax
        # elif feature_norm == "l2norm":
        #     attn = lambda x: l2norm(x, 2)
        elif feature_norm == "clipped_l2norm":
            self.normalize_attn = ClippedL2Norm()
        # elif feature_norm == "l1norm":
        #     attn = l1norm_d(attn, 2)
        # elif feature_norm == "clipped_l1norm":
        #     attn = nn.LeakyReLU(0.1)(attn)
        #     attn = l1norm_d(attn, 2)
        elif feature_norm == "clipped":
            self.normalize_attn = lambda x: nn.LeakyReLU(0.1)(x)
        elif feature_norm == "no_norm":
            self.normalize_attn = lambda x: x
        else:
            raise ValueError("unknown first norm type:", feature_norm)

    def forward(self, query, context, ):
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

         # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)
        attn = self.normalize_attn(attn)
        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size*queryL, sourceL)
        attn = nn.Softmax(dim=-1)(attn*self.smooth)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)

        return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


_similarities = {
    'cosine': {
        'class': Cosine,
        'args': {},
    },
    'region': {
        'class': RegionCorr,
        'args': {},
    },
    'order': None,
    'cross': {
        'class': CrossAttn,
        'args': Dict(k=8,),
    },
    'squeeze': {
        'class': ExcEmbedding,
        'args': Dict(k=8,),
    },
    'scan_i2t': {
        'class': StackedAttention,
        'args': Dict(
            i2t=True, agg_function='Mean',
            feature_norm='clipped_l2norm',
            lambda_lse=None, smooth=4,
        ),
    },
    'adaptive': {
        'class': AdaptiveEmbedding,
        'args': Dict(
            task='t2i',
        ),
    },
    'adaptive_norm': {
        'class': AdaptiveEmbedding,
        'args': Dict(
            task='t2i',
            norm=True,
        ),
    },
    'adaptive_k4': {
        'class': AdaptiveEmbedding,
        'args': Dict(
            task='t2i',
            k=4,
        ),
    },
    'adaptive_i2t': {
        'class': AdaptiveEmbeddingI2T,
        'args': Dict(),
    },
    'rnn_proj': {
        'class': RNNProj,
        'args': Dict(
            norm=False, num_layers=1,
            bidirectional=True,
            rnn_units=256,
        ),
    },
}


def get_similarity_object(similarity_name, **kwargs):
    settings = _similarities[similarity_name]
    args_dict = settings['args']
    args_dict.update(**kwargs)
    return settings['class'](**args_dict)


def get_sim_names():
    return _similarities.keys()
