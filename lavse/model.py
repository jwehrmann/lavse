import torch 
import torch.nn as nn

from .imgenc import get_image_encoder
from .txtenc import get_text_encoder, get_txt_pooling
from .utils.layers import l2norm
from .utils.logger import get_logger

logger = get_logger()


class LAVSE(nn.Module):

    def __init__(
        self, imgenc_name, txtenc_name, 
        num_embeddings, embed_dim=300,
        latent_size=1024, img_dim=2048,
        txt_pooling='lens',
    ):
        super(LAVSE, self).__init__()

        self.latent_size = latent_size

        self.img_enc = get_image_encoder(
            model_name=imgenc_name,
            latent_size=latent_size, 
            img_dim=img_dim,
        )
        logger.info((
            'Image encoder created\n'
            f'{self.img_enc}'
        ))
        
        self.txt_enc = get_text_encoder(
            model_name=txtenc_name,
            latent_size=latent_size,
            embed_dim=embed_dim,
            num_embeddings=num_embeddings,
        )
        self.txt_pool = get_txt_pooling(txt_pooling)

        logger.info((
            'Text encoder created\n'
            f'{self.txt_enc}'
        ))
    
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
        return img_features.mean(1)
    
    def embed_images(self, images):
        img_tensor = self.extract_image_features(images)
        img_embed  = self.embed_image_features(img_tensor)
        img_embed = l2norm(img_embed, dim=1)
        return img_embed

    def embed_captions(self, captions, lengths):
        txt_tensor, lengths = self.extract_caption_features(captions, lengths)
        txt_embed  = self.embed_caption_features(txt_tensor, lengths)
        txt_embed = l2norm(txt_embed, dim=1)
        return txt_embed

    def forward(
        self, images, captions, lengths,
    ):
        img_embed = self.embed_images(images)
        txt_embed = self.embed_captions(captions, lengths)

        return img_embed, txt_embed