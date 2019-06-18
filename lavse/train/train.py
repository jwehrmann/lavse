import json
import logging
import os
import random
from pathlib import Path
from random import shuffle
from timeit import default_timer as dt

import numpy as np
import torch
import torch.nn as nn
from addict import Dict
from numpy.polynomial.polynomial import polyfit
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm

from . import evaluation
from ..data.loaders import DataIterator
from ..model.loss import cosine_sim, cosine_sim_numpy
from ..utils import helper, layers, logger
from .evaluation import i2t, t2i

torch.manual_seed(0)
random.seed(0, version=2)


class Trainer:

    def __init__(
        self, model=None, device=torch.device('cuda'),
        args=None, sysoutlog=tqdm.write,
        master=True,
    ):

        self.model = model
        self.device = device
        self.train_logger = logger.LogCollector()
        self.val_logger = logger.LogCollector()

        self.args = args
        self.sysoutlog = sysoutlog

        self.optimizer = None
        self.metrics = {}
        self.master = master

    def setup_optim(
        self,
        mm_criterion=None,
        ml_criterion=None,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        cnn_lr_factor=0.1,
        lr_decay_rate=0.1,
        lr_decay_interval=15,
        clip_grad=2.,
        log_histograms=True,
        log_grad_norm=True,
        early_stop=50,
        save_all=False,
        finetune_convnet=False,
        **kwargs
    ):

        # TODO: improve this! :S
        total_params = 0
        nb_trainable_params = 0
        trainable_params = []
        for k, v in self.model.named_parameters():
            total_params += np.product(tuple(v.shape))
            if 'img_enc' in k and 'cnn' in k:
                if not finetune_convnet:
                    v.requires_grad = False
                continue
            if v.requires_grad:
                nb_trainable_params += np.product(tuple(v.shape))
                trainable_params.append(v)


        self.optimizer = optimizer(
            trainable_params, lr, **kwargs
        )

        if finetune_convnet:
            _params = self.model.img_enc.module.cnn.parameters()
            self.optimizer.add_param_group({
                'params': _params,
                'lr': lr * cnn_lr_factor,
                'name': 'cnn',
            })

        count_params = lambda p: np.sum([
            np.product(tuple(x.shape)) for x in p
        ])

        for k in self.optimizer.param_groups:
            self.sysoutlog(
                f"lr: {k['lr']}, #layers: {len(k['params'])}, #params: {count_params(k['params']):,}"
            )
        self.sysoutlog(
            #f'Trainable layers: {len(trainable_params)}, '
            f'Total Params: {total_params:,}, '
            #f'Train Params: {nb_trainable_params:,}'
        )

        #self.optimizer = nn.DataParallel(self.optimizer).cuda()
        #self.optimizer.param_groups = self.optimizer.module.param_groups
        #self.optimizer.step = self.optimizer.module.step

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45*1100)
        # self.mm_criterion = mm_criterion
        self.ml_criterion = ml_criterion
        self.initial_lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_interval = lr_decay_interval
        self.clip_grad = clip_grad
        self.log_histograms = log_histograms
        self.log_grad_norm = log_grad_norm

        self.best_val = 0
        self.count = early_stop
        self.early_stop = early_stop
        self.save_all = save_all

    def fit(
        self, train_loader, valid_loaders, lang_loaders=[],
        init_iteration=0, nb_epochs=2000, path='runs/',
        log_interval=50, valid_interval=500, world_size=1,
    ):
        self.path = path
        self.world_size = world_size
        if self.optimizer is None:
            print('You forgot to setup_optim.')
            exit()

        # Set up tensorboard logger
        tb_writer = helper.get_tb_writer(path)
        path = tb_writer.file_writer.get_logdir()
        self.tb_writer = tb_writer
        # Path to store the best models
        self.best_model_path = Path(path) / Path('best_model.pkl')

        self.train_iter = None
        self.lang_iters = {}

        pbar = lambda x: range(x)
        if self.master:
            pbar = lambda x: tqdm(range(x), desc='Epochs')

        for epoch in pbar(nb_epochs):
        # for epoch in range(nb_epochs):

            # Update learning rate
            self.learning_rate = helper.adjust_learning_rate(
                optimizer=self.optimizer,
                initial_lr=self.initial_lr,
                interval=self.lr_decay_interval,
                decay=self.lr_decay_rate,
                epoch=epoch,
            )

            # Train a single epoch
            continue_training = self.train_epoch(
                train_loader=train_loader,
                lang_loaders=lang_loaders,
                epoch=epoch,
                log_interval=log_interval,
                valid_loaders=valid_loaders,
                valid_interval=valid_interval,
                path=path,
            )
            if not continue_training:
                break

    def _forward_multimodal_loss(
        self, images, captions, lens
    ):
        img_emb, cap_emb = self.model(images, captions, lens)
        sim_matrix = self.model.get_sim_matrix(img_emb, cap_emb, lens)
        loss = self.model.mm_criterion(sim_matrix)
        # loss = self.mm_criterion(sim_matrix)
        return loss

    def _forward_multilanguage_loss(
        self, captions_a, lens_a, captions_b, lens_b, *args
    ):

        cap_a_embed = self.model.embed_captions(captions_a, lens_a)
        cap_b_embed = self.model.embed_captions(captions_b, lens_b)

        sim_matrix = self.model.get_sim_matrix(cap_a_embed, cap_b_embed)
        loss = self.ml_criterion(sim_matrix)

        return loss

    def train_epoch(
        self, train_loader, lang_loaders,
        epoch, valid_loaders=[], log_interval=50,
        valid_interval=500, path=''
    ):

        # lang_iters = [
        #     DataIterator(
        #         loader=loader,
        #         device=self.device,
        #         non_stop=True
        #     )
        #     for loader in lang_loaders
        # ]

        pbar = lambda x: x
        if self.master:
            pbar = lambda x: tqdm(
                x, total=len(x),
                desc='Steps ',
                leave=False,
            )

        for instance in pbar(train_loader):
            self.model.train()

            # Update progress bar
            self.optimizer.zero_grad()

            begin_forward = dt()
            images, captions, lens, ids = instance
            images = images.to(self.device)
            captions = captions.to(self.device)
            #images = nn.DataParallel(images).cuda()
            #captions = nn.DataParallel(captions).cuda()

            multimodal_loss = self._forward_multimodal_loss(
                images, captions, lens
            )

            iteration = self.model.mm_criterion.iteration
            adjusted_iter = self.world_size * iteration

            # Cross-language update
            total_lang_loss = 0.
            # for lang_iter in lang_iters:

            #     lang_data = lang_iter.next()

            #     lang_loss = self._forward_multilanguage_loss(*lang_data)
            #     total_lang_loss += lang_loss
            #     self.train_logger.update(
            #         f'train_loss_{str(lang_iter)}', lang_loss, 1
            #     )

            total_loss = multimodal_loss + total_lang_loss
            total_loss.backward()

            if self.log_grad_norm and self.master:
                logger.log_grad_norm(
                    self.model, self.tb_writer,
                    iteration=iteration,
                )

            if self.clip_grad > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            end_backward = dt()
            batch_time = end_backward-begin_forward

            train_info = Dict({
                'loss': multimodal_loss,
                'iteration': iteration,
                'total_loss': total_loss,
                'k': self.model.mm_criterion.k,
                'batch_time': batch_time,
                'learning_rate': self.learning_rate,
                'countdown': self.count,
                'epoch': epoch,
            })

            for param_group in self.optimizer.param_groups:
                if 'name' in param_group:
                    train_info.update({f"lr_{param_group['name']}": param_group['lr']})
                else:
                    train_info.update({'lr_base': param_group['lr']})

            if self.master:
                logger.tb_log_dict(
                    tb_writer=self.tb_writer, data_dict=train_info,
                    iteration=iteration, prefix='train'
                )

            if iteration % valid_interval == 0:
                # Run evaluation
                metrics, val_metric = self.evaluate_loaders(valid_loaders)

                # Update early stop variables
                # and save checkpoint
                if val_metric < self.best_val:
                    self.count -= 1
                elif not self.save_all:
                    self.count = self.early_stop
                    self.best_val = val_metric

                if self.master:
                    self.save(
                        path=self.path,
                        is_best=(val_metric >= self.best_val),
                        args=self.args,
                        rsum=val_metric,
                    )

                    # Log updates
                    for metric, values in metrics.items():
                        self.tb_writer.add_scalar(metric, values, iteration)

                # Early stop
                if self.count == 0 and self.master:
                    self.sysoutlog('\n\nEarly stop\n\n')
                    return False

            if iteration % log_interval == 0 and self.master:
                helper.print_tensor_dict(train_info, print_fn=self.sysoutlog)

                if self.log_histograms:
                    logger.log_param_histograms(
                        self.model, self.tb_writer,
                        iteration=self.model.mm_criterion.iteration,
                    )
        return True

    def evaluate_loaders(self, loaders):
        loader_metrics = {}
        final_sum = 0.

        nb_loaders = len(loaders)

        for i, loader in enumerate(loaders):
            loader_name = str(loader.dataset)
            self.sysoutlog(
                f'Evaluating {i+1:2d}/{nb_loaders:2d} - {loader_name}'
            )
            img_emb, txt_emb, lens = evaluation.predict_loader(
                model=self.model, data_loader=loader, device=self.device
            )

            result = evaluation.evaluate(
                model=self.model, img_emb=img_emb,
                txt_emb=txt_emb, lengths=lens,
                device=self.device, shared_size=128,
            )

            for k, v in result.items():
                self.sysoutlog(f'{k:<10s}: {v:>6.1f}')
            result = {
                f'{loader_name}/{metric_name}': v
                for metric_name, v in result.items()
            }

            loader_metrics.update(result)
            final_sum += result[f'{loader_name}/rsum']

        return loader_metrics, final_sum/float(nb_loaders)

    def save(
        self, path=None,
        is_best=False, args=None,
        **kwargs
    ):

        helper.save_checkpoint(
            path, self.model,
            optimizer=self.optimizer,
            is_best=is_best,
            save_all=self.save_all,
            iteration=self.model.mm_criterion.iteration,
            args=self.args,
            **kwargs
        )

    def load(self, path=None):
        if path is None:
            path = self.best_model_path

        states = helper.restore_checkpoint(
            path, model=self.model, optimizer=None
        )
        self.model = states['model'].to(self.device)

    def __repr__(self,):
        string = (
            f'{type(self).__name__} '
            f'{type(self.model).__name__} '
        )
        return string
