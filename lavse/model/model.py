import torch
import torch.nn as nn

from . import loss
from ..utils.logger import get_logger
from .imgenc import get_image_encoder, get_img_pooling
from .similarity.factory import get_similarity_object
from .similarity.measure import l2norm
from .txtenc import get_text_encoder, get_txt_pooling

from tqdm import tqdm

logger = get_logger()


class LAVSE(nn.Module):

    def __init__(
        self, txt_enc={}, img_enc={}, similarity={},
        ml_similarity={}, criterion={}, ml_criterion={},
        tokenizers=None, latent_size=1024, **kwargs
    ):
        super(LAVSE, self).__init__()
        '''
            txt_enc: parameters passed to the text encoder
            img_enc: parameters passed to the image encoder
            similarity: similarity object parameters
            criterion: required only for training
            ml_criterion: required only for training
        '''

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

        self.similarity = get_similarity_object(
            similarity.name,
            **similarity.params
        )

        # self.similarity = Similarity(
        #     similarity_object=sim_obj,
        #     device=similarity.device,
        #     latent_size=latent_size,
        #     **kwargs
        # )

        self.ml_similarity = nn.Identity()
        if ml_similarity:
            self.ml_similarity = get_similarity_object(
                ml_similarity.name,
                **ml_similarity.params
            )

        if criterion:
            self.multimodal_criterion = loss.get_loss(**criterion)
        if ml_criterion:
            self.multilanguage_criterion = loss.get_loss(**ml_criterion)

        logger.info(f'Using similarity: {similarity.name,}')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.img_enc.device = self.device
            self.txt_enc.device = self.device
            self.similarity.device = self.device

    #     self.set_devices_(
    #         txt_devices=txt_enc.devices,
    #         img_devices=img_enc.devices,
    #         loss_device=similarity.device,
    #     )

    # def set_device_(
    #     self, module, devices
    # ):
    #     from . import data_parallel
    #     if type(devices) == list:
    #         raise Exception('Devices must be a list of strings.')
    #     if len(devices) > 1:
    #         module = data_parallel.DataParallel(module).cuda()
    #         module.device = torch.device('cuda')
    #     elif len(devices) == 1:
    #         module.to(devices[0])
    #         module.device = torch.device(devices[0])
    #     else:
    #         raise Exception(
    #             f'Wrong number of provided devices: {devices}'
    #         )

    # def set_devices_(
    #     self, txt_devices=['cuda'],
    #     img_devices=['cuda'], loss_device='cuda',
    # ):
    #     from . import data_parallel

    #     self.set_device_(self.txt_enc, txt_devices)
    #     self.set_device_(self.img_enc, img_devices)

    #     self.loss_device = torch.device(loss_device)
    #     self.similarity = self.similarity.to(self.loss_device)
    #     self.ml_similarity = self.ml_similarity.to(self.loss_device)

    #     logger.info((
    #         f'Setting devices: '
    #         f'img: {self.img_enc.device},'
    #         f'txt: {self.txt_enc.device}, '
    #         f'loss: {self.loss_device}'
    #     ))

    # def embed_caption_features(self, cap_features, lengths):
    #     return self.txt_pool(cap_features, lengths)

    # def embed_image_features(self, img_features):
    #     return self.img_pool(img_features)

    def embed_images(self, batch):
        img_tensor = self.img_enc(batch)
        img_embed  = self.img_pool(img_tensor)
        return img_embed

    def embed_captions(self, batch):
        txt_tensor, lengths = self.txt_enc(batch)
        txt_embed = self.txt_pool(txt_tensor, lengths)
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

    # def get_sim_matrix(self, embed_a, embed_b, lens=None):
    #     return self.similarity(embed_a, embed_b, lens)

    # def get_ml_sim_matrix(self, embed_a, embed_b, lens=None):
    #     return self.ml_similarity(embed_a, embed_b, lens)

    def get_ml_sim_matrix(
        self, embed_a, embed_b, lens=None, shared_size=128
    ):
        return self.compute_pairwise_similarity(
            self.ml_similarity, embed_a, embed_b, lens,
            shared_size=shared_size
        )

    def get_sim_matrix(
        self, embed_a, embed_b, lens=None, shared_size=128
    ):
        return self.compute_pairwise_similarity(
            self.ml_similarity, embed_a, embed_b, lens,
            shared_size=shared_size
        )

    def compute_pairwise_similarity(
        self, similarity, img_embed, cap_embed, lens, shared_size=128
    ):
    # def forward_shared(self, img_embed, cap_embed, lens, shared_size=128):
        """
        Compute pairwise i2t image-caption distance
        """

        #img_embed = img_embed.to(self.device)
        #cap_embed = cap_embed.to(self.device)

        n_im_shard = (len(img_embed)-1)//shared_size + 1
        n_cap_shard = (len(cap_embed)-1)//shared_size + 1

        logger.debug('Calculating shared similarities')

        pbar_fn = lambda x: range(x)
        if self.master:
            pbar_fn = lambda x: tqdm(
                range(x), total=x,
                desc='Test  ',
                leave=False,
            )

        sim_matrix = torch.zeros(len(img_embed), len(cap_embed)).cpu()
        for i in pbar_fn(n_im_shard):
            im_start = shared_size*i
            im_end = min(shared_size*(i+1), len(img_embed))
            for j in range(n_cap_shard):
                cap_start = shared_size*j
                cap_end = min(shared_size*(j+1), len(cap_embed))
                im = img_embed[im_start:im_end]
                s = cap_embed[cap_start:cap_end]
                l = lens[cap_start:cap_end]
                sim = similarity(im, s, l)
                sim_matrix[im_start:im_end, cap_start:cap_end] = sim

        logger.debug('Done computing shared similarities.')
        return sim_matrix

    # def loss(self, batch):

    #     loss_values = {}
    #     loss_values['loss']self.forward_multimodal_loss(batch)

    def forward_multimodal_loss(
        self, batch
    ):
        img_emb, cap_emb = self.forward_batch(batch)
        _, lens = batch['caption']

        sim_matrix = self.compute_pairwise_similarity(
            self.similarity, img_emb, cap_emb, lens)

        loss = self.multimodal_criterion(sim_matrix)

        # cap_vec = pooling.last_hidden_state_pool(cap_emb, lens)
        # sim_global = self.model.cosine.forward_shared(
        #     img_emb.mean(1), cap_vec, lens,
        # ).cuda()
        # global_loss = self.model.mm_criterion(sim_global).cuda()
        # self.model.mm_criterion.iteration -= 1

        # loss = self.mm_criterion(sim_matrix)
        return loss

    def forward_multilanguage_loss(
        self, captions_a, lens_a, captions_b, lens_b, *args
    ):

        cap_a_embed = self.embed_captions({'caption': (captions_a, lens_a)})
        cap_b_embed = self.embed_captions({'caption': (captions_b, lens_b)})

        if len(cap_a_embed.shape) == 3:
            from ..model.txtenc import pooling
            cap_a_embed = pooling.last_hidden_state_pool(cap_a_embed, lens_a)
            cap_b_embed = pooling.last_hidden_state_pool(cap_b_embed, lens_b)

        sim_matrix = self.get_ml_sim_matrix(cap_a_embed, cap_b_embed, lens_b)
        loss = self.ml_criterion(sim_matrix)

        return loss

# def lavse_from_checkpoint(model_path):
#     from .utils import helper
#     checkpoint = helper.restore_checkpoint(model_path)
