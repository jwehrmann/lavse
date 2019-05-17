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
        self, imgenc_name, txtenc_name,
        num_embeddings, embed_dim=300,
        latent_size=1024, txt_pooling='lens',
        img_pooling='mean',
        similarity_name='cosine',
        device=None, **kwargs
    ):
        super(LAVSE, self).__init__()

        self.latent_size = latent_size

        self.img_enc = get_image_encoder(
            model_name=imgenc_name,
            latent_size=latent_size,
        )

        logger.info((
            'Image encoder created: '
            f'{imgenc_name}'
        ))

        self.txt_enc = get_text_encoder(
            model_name=txtenc_name,
            latent_size=latent_size,
            embed_dim=embed_dim,
            num_embeddings=num_embeddings,
        )
        self.txt_pool = get_txt_pooling(txt_pooling)
        self.img_pool = get_img_pooling(img_pooling)

        logger.info((
            'Text encoder created: '
            f'{txtenc_name}'
        ))

        sim_obj = get_similarity_object(
            similarity_name,
            device=device,
            **kwargs
        )
        self.similarity = Similarity(
            similarity_object=sim_obj,
            device=device,
            latent_size=latent_size,
            **kwargs
        )
        logger.info(f'Using similarity: {similarity_name}')
    
    def set_master_(self, is_master=True):
        self.master = is_master
        self.similarity.set_master_(is_master)

    def extract_caption_features(
        self, captions, lengths,
    ):
        return self.txt_enc(captions, lengths)

    def extract_image_features(
        self, images,
    ):
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
