import json
import logging
import os
from pathlib import Path
from random import shuffle
from timeit import default_timer as dt

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.polynomial import polyfit
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm

from .data import DataIterator, prepare_ml_data, prepare_mm_data
from .evaluation import i2t, t2i
from .loss import cosine_sim, cosine_sim_numpy
from .utils import helper, layers, logger

from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
torch.manual_seed(0)

from . import evaluation

from addict import Dict

import random
random.seed(0, version=2)


class Trainer:

    def __init__(
        self, model=None, device='cuda:0',
        args=None, sysoutlog=tqdm.write
    ):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_logger = logger.LogCollector()
        self.val_logger = logger.LogCollector()

        self.args = args
        self.sysoutlog = sysoutlog

        self.optimizer = None
        self.metrics = {}

    def setup_optim(
        self,
        mm_criterion,
        ml_criterion=None,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        lr_decay_rate=0.1,
        lr_decay_interval=15,
        clip_grad=2.,
        log_histograms=True,
        log_grad_norm=True,
        early_stop=50,
        save_all=False,
        **kwargs
    ):
        self.optimizer = optimizer(self.model.parameters(), lr, **kwargs)
        self.mm_criterion = mm_criterion
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
        nb_epochs=2000, path='runs/',
        log_interval=50, valid_interval=500
    ):
        self.path = path
        if self.optimizer is None:
            print('You forgot to setup_optim.')
            exit()

        self.pbar = tqdm(total=len(train_loader), desc='Steps ')

        # Set up tensorboard logger
        tb_writer = helper.get_tb_writer(path)
        path = tb_writer.file_writer.get_logdir()
        self.tb_writer = tb_writer
        # Path to store the best models
        self.best_model_path = Path(path) / Path('best_model.pkl')

        self.train_iter = None
        self.lang_iters = {}

        for epoch in tqdm(range(nb_epochs), desc='Epochs'):
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
        loss = self.mm_criterion(sim_matrix)
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

        pbar = helper.reset_pbar(self.pbar)

        for instance in train_loader:
            self.model.train()

            # Update progress bar
            self.pbar.update(1)
            self.optimizer.zero_grad()

            begin_forward = dt()
            images, captions, lens, ids = instance
            images = images.to(self.device)
            captions = captions.to(self.device)

            multimodal_loss = self._forward_multimodal_loss(
                images, captions, lens
            )

            iteration = self.mm_criterion.iteration

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

            if self.log_grad_norm:
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
                'k': self.mm_criterion.k,
                'batch_time': batch_time,
                'learning_rate': self.learning_rate,
                'countdown': self.count,
                'epoch': epoch,
            })
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
                else:
                    self.count = self.early_stop
                    self.best_val = val_metric
                    self.save(
                        path=self.path,
                        is_best=True,
                        args=self.args
                    )
                if self.save_all:
                    self.save(
                        path=self.path,
                        is_best=True,
                        args=self.args
                    )

                # Log updates
                for metric, values in metrics.items():
                    self.tb_writer.add_scalar(metric, values, iteration)

                # Early stop
                if self.count == 0:
                    self.sysoutlog('\n\nEarly stop\n\n')
                    return False

            if self.pbar.n % log_interval == 0:
                helper.print_tensor_dict(train_info, print_fn=self.sysoutlog)

                if self.log_histograms:
                    logger.log_param_histograms(
                        self.model, self.tb_writer,
                        iteration=self.mm_criterion.iteration,
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

    def save(self, path=None, is_best=False, args=None):

        helper.save_checkpoint(
            path, self.model,
            optimizer=self.optimizer,
            epoch=self.mm_criterion.iteration,
            args=args, classes=None,
            is_best=is_best, save_all=self.save_all,
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
