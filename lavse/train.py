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
        early_stop=50,
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
                
        self.best_val = 0
        self.count = early_stop
        self.early_stop = early_stop

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
            
            self.train_logger.update('epoch', epoch, 0)
            
            # Update learning rate
            lr = helper.adjust_learning_rate(
                optimizer=self.optimizer, 
                initial_lr=self.initial_lr,
                interval=self.lr_decay_interval, 
                decay=self.lr_decay_rate,
                epoch=epoch,
            )
            self.train_logger.update('lr', lr, 0)
            
            # Train a single epoch
            self.train_epoch(
                train_loader=train_loader,
                lang_loaders=lang_loaders,
                epoch=epoch,
                log_interval=log_interval,
                valid_loaders=valid_loaders,
                valid_interval=valid_interval,
                path=path,
            )

    
    def _forward_multimodal_loss(
        self, images, captions, lens
    ):
        self.optimizer.zero_grad()        

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

        # self.model.train()
        train_iter = DataIterator(
            loader=train_loader, 
            device=self.device,
        )
        lang_iters = [
            DataIterator(
                loader=loader, 
                device=self.device, 
                non_stop=True
            ) 
            for loader in lang_loaders
        ]

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

            # Cross-language update
            total_lang_loss = 0.
            for lang_iter in lang_iters:

                lang_data = lang_iter.next()
            
                lang_loss = self._forward_multilanguage_loss(*lang_data)
                total_lang_loss += lang_loss
                self.train_logger.update(
                    f'train_loss_{str(lang_iter)}', lang_loss, 1
                )
            
            total_loss = multimodal_loss + total_lang_loss
            total_loss.backward()

            for k, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                self.tb_writer.add_scalar(
                    f'grads/{k}', 
                    p.grad.data.norm(2).item(), 
                    self.mm_criterion.iteration,
                )

            # print('Iter s{}'.format(self.mm_criterion.iteration))
            # for k, v in self.model.txt_enc.named_parameters():
            #     print('{:35s}: {:8.5f}, {:8.5f}, {:8.5f}, {:8.5f}'.format(
            #         k, v.data.cpu().min().numpy(), 
            #         v.data.cpu().mean().numpy(),
            #         v.data.cpu().max().numpy(),
            #         v.data.cpu().std().numpy(),
            #     ))
            # for k, v in self.model.img_enc.named_parameters():
            #     print('{:35s}: {:8.5f}, {:8.5f}, {:8.5f}, {:8.5f}'.format(                k, v.data.cpu().min().numpy(), 
            #         v.data.cpu().mean().numpy(),
            #         v.data.cpu().max().numpy(),
            #         v.data.cpu().std().numpy(),
            #     ))
            # for k, p in self.model.txt_enc.named_parameters():
            #     if p.grad is None:
            #         continue
            #     print('{:35s}: {:8.5f}'.format(k, p.grad.data.norm(2).item(),))
            
            # for k, p in self.model.img_enc.named_parameters():
            #     if p.grad is None:
            #         continue
            #     print('{:35s}: {:8.5f}'.format(k, p.grad.data.norm(2).item(),))

            # print('\n\n')

            if self.clip_grad > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad)                    
            
            self.optimizer.step()
            end_backward = dt()

            self.train_logger.update('train_loss', multimodal_loss, 1)
            self.train_logger.update('total_loss', total_loss, 1)
            self.train_logger.update(
                'iteration', self.mm_criterion.iteration, 0
            )
            self.train_logger.update('k', self.mm_criterion.k, 0)
            self.train_logger.update(
                'batch_time', end_backward-begin_forward, 1
            )

            if self.pbar.n % log_interval == 0:
                self.sysoutlog(f'{self.train_logger}')

                if self.log_histograms:
                    for k, p in self.model.named_parameters():
                        self.tb_writer.add_histogram(
                            f'params/{k}', 
                            p.data, 
                            self.mm_criterion.iteration,
                        )

            self.train_logger.tb_log(
                self.tb_writer, 
                step=self.mm_criterion.iteration, prefix='train/',
            )

            if self.mm_criterion.iteration % valid_interval == 0:
                # Run evaluation
                metrics, val_metric = self.evaluate_loaders(valid_loaders)

                # Update early stop variables 
                # and save checkpoint
                if val_metric < self.best_val:
                    self.count -= 1
                else:
                    self.count = self.early_stop
                    self.best_val = val_metric
                    self.save(path=self.path, is_best=True, args=self.args)
                
                # Log updates
                self.train_logger.update('countdown', self.count, 0)
                self.val_logger.update_dict(
                    metrics,
                    self.mm_criterion.iteration,
                    self.mm_criterion.iteration,
                    self.path,
                )
                # Update train and validation metrics on tensorboard
                # self.train_logger.tb_log(
                #     tb_writer, step=epoch, prefix='train/',
                # )
                self.val_logger.tb_log(
                    self.tb_writer,
                    step=self.mm_criterion.iteration,
                    prefix='',
                )

                # Early stop
                if self.count == 0:
                    self.sysoutlog('Early stop')
                    break
            
    def predict_loader(self, data_loader):
        self.model.eval()

        # np array to keep all the embeddings
        img_embs = None
        cap_embs = None
        cap_lens = None
        
        max_n_word = 0
        for i, (images, captions, lengths, ids) in enumerate(data_loader):
            max_n_word = max(max_n_word, max(lengths))

        for i, (images, captions, lengths, ids) in enumerate(data_loader):
            
            images = images.to(self.device)
            captions = captions.to(self.device)
            # compute the embeddings
            img_emb, cap_emb = self.model(images, captions, lengths)
            
            if img_embs is None:
                if len(img_emb.shape) == 3:
                    img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                else:
                    img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
                cap_lens = [0] * len(data_loader.dataset)
            # cache embeddings
            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
            for j, nid in enumerate(ids):
                cap_lens[nid] = lengths[j]
            
            del images, captions
                
        # Remove image feature redundancy
        if img_embs.shape[0] == cap_embs.shape[0]:
            img_embs = img_embs[
                np.arange(
                    start=0,
                    stop=img_embs.shape[0], 
                    step=5
                ).astype(np.int),
            ]

        return img_embs, cap_embs, cap_lens
    
    def evaluate(self, loader,):        
        _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

        begin_pred = dt()
        img_emb, txt_emb, lengths = self.predict_loader(loader)

        img_emb = torch.FloatTensor(img_emb).to(self.device)
        txt_emb = torch.FloatTensor(txt_emb).to(self.device)

        end_pred = dt()
        with torch.no_grad():
            sims = self.model.get_sim_matrix_shared(
                embed_a=img_emb, embed_b=txt_emb, 
                lens=lengths, shared_size=128
            )
            # sims = self.model.get_sim_matrix(
            #     embed_a=img_emb, embed_b=txt_emb, 
            #     lens=lengths,
            # )
            sims = layers.tensor_to_numpy(sims)
        end_sim = dt()        

        i2t_metrics = i2t(sims)
        t2i_metrics = t2i(sims)

        rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

        i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
        t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}
        
        metrics = {
            'pred_time': end_pred-begin_pred,
            'sim_time': end_sim-end_pred,
        }        
        metrics.update(i2t_metrics)
        metrics.update(t2i_metrics)
        metrics['rsum'] = rsum

        return metrics

    def evaluate_loaders(self, loaders):
        loader_metrics = {}
        final_sum = 0.

        nb_loaders = len(loaders)

        for i, loader in enumerate(loaders):
            loader_name = str(loader.dataset)
            self.sysoutlog(
                f'Evaluating {i+1:2d}/{nb_loaders:2d} - {loader_name}'
            )
            result = self.evaluate(loader)
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
            optimizer=None, epoch=-1, 
            args=args, classes=None,
            is_best=is_best
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
