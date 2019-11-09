import json
import logging
import os
import random
from pathlib import Path
from timeit import default_timer as dt

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from addict import Dict
from tqdm import tqdm

from . import evaluation
from ..data.loaders import DataIterator
from ..utils import file_utils, helper, layers, logger
from .lr_scheduler import get_scheduler
# from .train import warmup

def freeze(module):
     for x in module.parameters():
         x.requires_grad = False


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
        optimizer={},
        lr=1e-3,
        lr_scheduler=None,
        clip_grad=2.,
        log_histograms=False,
        log_grad_norm=False,
        early_stop=50,
        save_all=False,
        freeze_modules=[],
        val_metric='rsum',
        **kwargs
    ):
        from . import optimizers
        count_params = lambda p: np.sum([
            np.product(tuple(x.shape)) for x in p
        ])

        total_params = count_params(self.model.parameters())

        for fmod in freeze_modules:
            print(f'Freezing {fmod}')
            freeze(eval(f'self.{fmod}'))

        lr_groups = ''
        lr_params = {}
        if self.args.optimizer.lr_groups_mult:
            lr_groups = self.args.optimizer.lr_groups
            lr_groups_mult = self.args.optimizer.lr_groups_mult
            lr_params['params'] = list(eval(f'self.{lr_groups}.parameters()'))
            lr_params['lr'] = lr_groups_mult
            # lr_params['lr'] = lr * lr_groups_mult
            lr_params['name'] = lr_groups
            lr_groups = lr_groups[6:]

        trainable_params = []
        for k, v in self.model.named_parameters():
            if not lr_groups or not k.startswith(lr_groups):
                if v.requires_grad:
                    trainable_params.append(v)

        self.optimizer = optimizers.get_optimizer(
            optimizer.name,
            trainable_params,
            **optimizer.params,
        )
        if lr_params:
            self.optimizer.add_param_group(lr_params)

        scheduler = None
        if lr_scheduler:
            scheduler = get_scheduler(
                optimizer=self.optimizer,
                name=lr_scheduler.name,
                **lr_scheduler.params,
            )
        # if optimizer.warmup:
        #     scheduler = warmup.GradualWarmupScheduler(
        #         self.optimizer,
        #         **optimizer.warmup
        #     )

        for k in self.optimizer.param_groups:
            self.sysoutlog(
                f"lr: {k['lr']}, #layers: {len(k['params'])}, #params: {count_params(k['params']):,}"
            )

        self.sysoutlog(
            f'Trainable layers: {len(trainable_params)}, '
            f'Total Params: {total_params:,}, '
            #f'Train Params: {nb_trainable_params:,}'
        )

        self.lr_scheduler = scheduler
        self.clip_grad = clip_grad
        self.log_histograms = log_histograms
        self.log_grad_norm = log_grad_norm
        self.save_all = save_all
        self.best_val = 1e10 if 'loss' in val_metric else 0
        self.count = early_stop
        self.early_stop = early_stop
        self.val_metric = val_metric

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

        path = file_utils.get_logdir(path)
        tb_writer = helper.get_tb_writer(path)
        file_utils.save_yaml_opts(Path(path) / 'options.yaml', self.args)
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

    def train_epoch(
        self, train_loader, lang_loaders,
        epoch, valid_loaders=[], log_interval=50,
        valid_interval=500, path=''
    ):

        lang_iters = [
            DataIterator(
                loader=loader,
                device=self.device,
                non_stop=True
            )
            for loader in lang_loaders
        ]

        pbar = lambda x: x
        if self.master:
            pbar = lambda x: tqdm(
                x, total=len(x),
                desc='Steps ',
                leave=False,
            )

        for batch in pbar(train_loader):
            self.model.train()

            # Update progress bar
            self.optimizer.zero_grad()

            begin_forward = dt()

            multimodal_loss = self.model.forward_multimodal_loss(batch)
            iteration = self.model.multimodal_criterion.iteration
            adjusted_iter = self.world_size * iteration

            # Cross-language update
            total_lang_loss = 0.
            loss_info = {}
            for lang_iter in lang_iters:

                lang_data = lang_iter.next()
                lang_loss = self.model.forward_multilanguage_loss(*lang_data)
                total_lang_loss += lang_loss
                loss_info[f'train_loss_{str(lang_iter)}'] = lang_loss

            total_loss = multimodal_loss + total_lang_loss
            total_loss.backward()

            norm, img_norm, sim_norm, txt_norm = 0., 0., 0., 0.
            if self.clip_grad > 0:
                img_norm = clip_grad_norm_(
                    self.model.img_enc.parameters(),
                    self.clip_grad
                )
                sim_norm = clip_grad_norm_(
                    self.model.similarity.parameters(),
                    self.clip_grad
                )
                txt_norm = clip_grad_norm_(
                    self.model.txt_enc.parameters(),
                    self.clip_grad
                )
                norm = img_norm + sim_norm + txt_norm

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            end_backward = dt()
            batch_time = end_backward-begin_forward

            train_info = Dict({
                'loss': multimodal_loss,
                'iteration': iteration,
                'total_loss': total_loss,
                'k': self.model.multimodal_criterion.k,
                'batch_time': batch_time,
                'countdown': self.count,
                'epoch': epoch,
                'norm': norm,
                'norm_img': img_norm,
                'norm_txt': txt_norm,
                'norm_sim': sim_norm,
            })

            train_info.update(loss_info)

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
                metrics, metric_value = self.evaluate_loaders(valid_loaders)

                # Update early stop variables
                # and save checkpoint
                if metric_value < self.best_val:
                    self.count -= 1
                elif not self.save_all:
                    self.count = self.early_stop
                    self.best_val = metric_value

                if self.master:
                    self.save(
                        path=self.path,
                        is_best=(metric_value >= self.best_val),
                        args=self.args,
                        rsum=metric_value,
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
                        iteration=self.model.multimodal_criterion.iteration,
                    )
                if self.log_grad_norm:
                    logger.log_grad_norm(
                        self.model, self.tb_writer,
                        iteration=self.model.multimodal_criterion.iteration,
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

            metric_value = result[self.val_metric]
            if 'loss' in self.val_metric:
                metric_value = -metric_value

            for k, v in result.items():
                self.sysoutlog(f'{k:<10s}: {v:>6.1f}')

            result = {
                f'{loader_name}/{metric_name}': v
                for metric_name, v in result.items()
            }

            loader_metrics.update(result)
            final_sum += metric_value

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
            iteration=self.model.multimodal_criterion.iteration,
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
