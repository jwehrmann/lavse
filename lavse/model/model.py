import torch
import torch.nn as nn

from .imgenc import get_image_encoder, get_img_pooling
from .txtenc import get_text_encoder, get_txt_pooling
from .similarity.similarity import Similarity
from .similarity.measure import l2norm
from .similarity.factory import get_similarity_object
from ..utils.logger import get_logger

logger = get_logger()


class LAVSE(nn.Module):

    def __init__(
        self, txt_enc={}, img_enc={}, similarity={},
        tokenizer=None, latent_size=1024, **kwargs
    ):
        super(LAVSE, self).__init__()

        self.latent_size = latent_size
        self.img_enc = get_image_encoder(
            name=img_enc.name,
            latent_size=latent_size,
            **img_enc.params
        )

        logger.info((
            'Image encoder created: '
            f'{img_enc.name,}'
        ))

        self.txt_enc = get_text_encoder(
            name = txt_enc.name,
            latent_size=latent_size,
            num_embeddings=len(tokenizer),
            **txt_enc.params,
        )

        self.txt_pool = get_txt_pooling(txt_enc.pooling)
        self.img_pool = get_img_pooling(img_enc.pooling)

        logger.info((
            'Text encoder created: '
            f'{txt_enc.name}'
        ))

        sim_obj = get_similarity_object(
            similarity.name,
            **similarity.params
        )

        self.similarity = Similarity(
            similarity_object=sim_obj,
            device=similarity.device,
            latent_size=latent_size,
            **kwargs
        )

        logger.info(f'Using similarity: {similarity.name,}')

    def set_devices_(
        self, txt_devices=['cuda'],
        img_devices=['cuda'], loss_device='cuda',
    ):
        from . import data_parallel

        if len(txt_devices) > 1:
            self.txt_enc = data_parallel.DataParallel(self.txt_enc)
            self.txt_device = torch.device('cuda')
        elif len(txt_devices) == 1:
            self.txt_device = torch.device(txt_devices[0])
            self.txt_enc.to(txt_devices[0])

        if len(img_devices) > 1:
            self.img_enc = data_parallel.DataParallel(self.img_device)
            self.img_device = torch.device('cuda')
        elif len(img_devices) == 1:
            self.img_device = torch.device(img_devices[0])
            self.img_enc.to(img_devices[0])

        self.loss_device = torch.device(
            loss_device
        )

        logger.info((
            f'Setting devices: '
            f'img: {self.img_device},'
            f'txt: {self.txt_device}, '
            f'loss: {self.loss_device}'
        ))

    def extract_caption_features(
        self, captions, lengths,
    ):
        captions = captions.to(self.txt_device)
        return self.txt_enc(captions, lengths)

    def extract_image_features(
        self, images,
    ):
        images = images.to(self.img_device)
        return self.img_enc(images)

    def embed_caption_features(self, cap_features, lengths):
        return self.txt_pool(cap_features, lengths)

    def embed_image_features(self, img_features):
        return self.img_pool(img_features)

    def embed_images(self, images):
        img_tensor = self.extract_image_features(images)
        img_embed  = self.embed_image_features(img_tensor)
        # img_embed = l2norm(img_embed, dim=1)
        return img_embed

    def embed_captions(self, captions, lengths):
        txt_tensor, lengths = self.extract_caption_features(captions, lengths)
        txt_embed = self.embed_caption_features(txt_tensor, lengths)
        # txt_embed = l2norm(txt_embed, dim=1)
        return txt_embed

    def forward(
        self, images, captions, lengths,
    ):
        img_embed = self.embed_images(images)
        txt_embed = self.embed_captions(captions, lengths)

        return img_embed, txt_embed

    def get_sim_matrix(self, embed_a, embed_b, lens=None):
        return self.similarity(embed_a, embed_b, lens)

    def get_sim_matrix_shared(
        self, embed_a, embed_b, lens=None, shared_size=128
    ):
        return self.similarity.forward_shared(
            embed_a, embed_b, lens,
            shared_size=shared_size
        )
