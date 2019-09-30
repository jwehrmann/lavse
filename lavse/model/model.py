import torch
import torch.nn as nn

from ..utils.logger import get_logger
from .imgenc import get_image_encoder, get_img_pooling
from .similarity.factory import get_similarity_object
from .similarity.measure import l2norm
from .similarity.similarity import Similarity
from .txtenc import get_text_encoder, get_txt_pooling

logger = get_logger()


class LAVSE(nn.Module):

    def __init__(
        self, txt_enc={}, img_enc={}, similarity={},
        ml_similarity={}, tokenizers=None, latent_size=1024,
        **kwargs
    ):
        super(LAVSE, self).__init__()

        # Flag for distributed dataparallel
        self.master = True
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
            tokenizers=tokenizers,
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

        self.ml_similarity = nn.Identity()
        if ml_similarity is not None:
            self.ml_similarity = self.similarity

            # FIXME: this is bad
            if ml_similarity != {}:
                ml_sim_obj = get_similarity_object(
                    ml_similarity.name,
                    **ml_similarity.params
                )

                self.ml_similarity = Similarity(
                    similarity_object=ml_sim_obj,
                    device=similarity.device,
                    latent_size=latent_size,
                    **kwargs
                )

        logger.info(f'Using similarity: {similarity.name,}')

        self.set_devices_(
            txt_devices=txt_enc.devices,
            img_devices=img_enc.devices,
            loss_device=similarity.device,
        )

    def set_device_(
        self, module, devices
    ):
        from . import data_parallel
        if type(devices) == list:
            raise Exception('Devices must be a list of strings.')
        if len(devices) > 1:
            module = data_parallel.DataParallel(module).cuda()
            module.device = torch.device('cuda')
        elif len(devices) == 1:
            module.to(devices[0])
            module.device = torch.device(devices[0])
        else:
            raise Exception(
                f'Wrong number of provided devices: {devices}'
            )

    def set_devices_(
        self, txt_devices=['cuda'],
        img_devices=['cuda'], loss_device='cuda',
    ):
        from . import data_parallel

        self.set_device_(self.txt_enc, txt_devices)
        self.set_device_(self.img_enc, img_devices)

        self.loss_device = torch.device(loss_device)
        self.similarity = self.similarity.to(self.loss_device)
        self.ml_similarity = self.ml_similarity.to(self.loss_device)

        logger.info((
            f'Setting devices: '
            f'img: {self.img_enc.device},'
            f'txt: {self.txt_enc.device}, '
            f'loss: {self.loss_device}'
        ))

    def embed_caption_features(self, cap_features, lengths):
        return self.txt_pool(cap_features, lengths)

    def embed_image_features(self, img_features):
        return self.img_pool(img_features)

    def embed_images(self, batch):
        img_tensor = self.img_enc(batch)
        img_embed  = self.embed_image_features(img_tensor)
        return img_embed

    def embed_captions(self, batch):
        txt_tensor, lengths = self.txt_enc(batch)
        txt_embed = self.embed_caption_features(txt_tensor, lengths)
        return txt_embed

    def forward_batch(
        self, batch
    ):
        img_embed = self.embed_images(batch)
        txt_embed = self.embed_captions(batch)

        return img_embed, txt_embed

    def forward(
        self, images, captions, lengths,
    ):
        img_embed = self.embed_images(images)
        txt_embed = self.embed_captions(captions, lengths)

        return img_embed, txt_embed

    def get_sim_matrix(self, embed_a, embed_b, lens=None):
        return self.similarity(embed_a, embed_b, lens)

    def get_ml_sim_matrix(self, embed_a, embed_b, lens=None):
        return self.ml_similarity(embed_a, embed_b, lens)

    def get_sim_matrix_shared(
        self, embed_a, embed_b, lens=None, shared_size=128
    ):
        return self.similarity.forward_shared(
            embed_a, embed_b, lens,
            shared_size=shared_size
        )


# def lavse_from_checkpoint(model_path):
#     from .utils import helper
#     checkpoint = helper.restore_checkpoint(model_path)

