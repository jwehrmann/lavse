import json
import logging
import os
from pathlib import Path
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.polynomial import polyfit
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm

from .utils import helper, logger, layers
from .loss import cosine_sim_numpy, cosine_sim
from .evaluation import i2t, t2i
from .data import DataIterator, prepare_mm_data, prepare_ml_data


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
    ):
        self.optimizer = optimizer(self.model.parameters(), lr)
        self.mm_criterion = mm_criterion
        self.ml_criterion = ml_criterion
        self.initial_lr = lr
        self.lr_decay_rate = lr_decay_rate 
        self.lr_decay_interval = lr_decay_interval

    def fit(
        self, train_loader, valid_loaders, lang_loaders=[],
        nb_epochs=2000, early_stop=50, path='runs/',
        log_interval=50,
    ):

        if self.optimizer is None:
            print('You forgot to setup_optim.')
            exit()

        self.pbar = tqdm(total=len(train_loader), desc='Steps ')

        # Set up tensorboard logger
        tb_writer = helper.get_tb_writer(path)
        path = tb_writer.file_writer.get_logdir()

        # Path to store the best models
        self.best_model_path = Path(path) / Path('best_model.pkl')
        
        best_val = 0
        count = early_stop

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

            # Update epoch for correct k calculation 
            # Record k values for monitoring 
            self.mm_criterion.update_epoch()
            k = self.mm_criterion.update_k()
            self.train_logger.update('k', k, 0)
            
            # Train a single epoch
            self.train_epoch(
                train_loader=train_loader,
                lang_loaders=lang_loaders, 
                epoch=epoch, 
                log_interval=log_interval,
            )

            # Run evaluation
            metrics, val_metric = self.evaluate_loaders(valid_loaders)

            # Update early stop variables 
            # and save checkpoint
            if val_metric < best_val:
                count -= 1
            else:
                count = early_stop
                best_val = val_metric
                self.save(path=path, is_best=True, args=self.args)
            
            # Log updates
            self.train_logger.update('countdown', count, 0)
            self.val_logger.update_dict(
                metrics, epoch, count, path
            )
            # Update train and validation metrics on tensorboard
            self.train_logger.tb_log(
                tb_writer, step=epoch, prefix='train/'
            )
            self.val_logger.tb_log(
                tb_writer, step=epoch, prefix='valid/'
            )

            # Early stop
            if count == 0:
                self.sysoutlog('Early stop')
                break
            

    def _forward_multimodal_loss(
        self, images, captions, lens
    ):
        self.optimizer.zero_grad()            

        img_emb, cap_emb = self.model(images, captions, lens)
        loss = self.mm_criterion(img_emb, cap_emb, )

        return loss
    
    def _forward_multilanguage_loss(
        self, captions_a, lens_a, captions_b, lens_b, *args
    ):

        cap_a_embed = self.model.embed_captions(captions_a, lens_a)
        cap_b_embed = self.model.embed_captions(captions_b, lens_b)

        loss = self.ml_criterion(cap_a_embed, cap_b_embed)
        return loss

    def train_epoch(
        self, train_loader, lang_loaders, 
        epoch, log_interval=50
    ):

        self.model.train()
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

        self.pbar.clear()
        
        while True:
            
            # Image-Caption Alignment update
            try:
                instance = train_iter.next()
            # End of epoch
            except StopIteration:
                return True
            
            iteration = len(train_loader) * (epoch) + self.pbar.n
            self.train_logger.update('iteration', iteration, 0)
            
            # Update progress bar
            self.pbar.update(1)
            self.optimizer.zero_grad()

            images, captions, lens, ids = instance
            multimodal_loss = self._forward_multimodal_loss(
                images, captions, lens
            )

            # Cross-language update
            total_lang_loss = 0.
            for lang_iter in lang_iters:

                lang_data = lang_iter.next()
            
                lang_loss = self._forward_multilanguage_loss(*lang_data)
                total_lang_loss += lang_loss
                self.train_logger.update(f'train_loss_{str(lang_iter)}', lang_loss, 1)

            self.train_logger.update('train_loss', multimodal_loss, 1)
            
            total_loss = multimodal_loss + total_lang_loss
            total_loss.backward()
            self.optimizer.step()

            if self.pbar.n % log_interval == 0:
                self.sysoutlog(f'{self.train_logger}')

    def predict_loader(self, loader):
        self.model.eval()

        n = loader.dataset.length
        img_embeddings = np.zeros((n, self.model.latent_size))
        text_embeddings = np.zeros((n, self.model.latent_size))

        test_iter = DataIterator(loader, device=self.device)

        with torch.no_grad():
           while True:
                try:
                    images, captions, lens, ids = test_iter.next()
                    img_emb, txt_emb = self.model(images, captions, lens)
                    img_embeddings[ids] = layers.tensor_to_numpy(img_emb)
                    text_embeddings[ids] = layers.tensor_to_numpy(txt_emb)
                except StopIteration:
                    break
        # Remove image feature redundancy
        if img_embeddings.shape[0] == text_embeddings.shape[0]:
            img_embeddings = img_embeddings[
                np.arange(
                    start=0,
                    stop=img_embeddings.shape[0], 
                    step=5).astype(np.int),
            ]
        return img_embeddings, text_embeddings
    
    def evaluate(self, loader,):      
        # from timeit import default_timer as dt 
        _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

        # b1 = dt()
        img_emb, txt_emb = self.predict_loader(loader)
        # b2 = dt()
        sims = cosine_sim_numpy(img_emb, txt_emb)

        # img_emb = torch.Tensor(img_emb)
        # txt_emb = torch.Tensor(txt_emb)
        # b3 = dt()
        # sims = cosine_sim(img_emb, txt_emb)

        # e = dt()
        # self.sysoutlog(f'Prediction {e-b1}')
        # self.sysoutlog(f'Similarity {e-b2}')
        # self.sysoutlog(f'Pred gpu   {e-b3}')

        i2t_metrics = i2t(sims)
        t2i_metrics = t2i(sims)

        rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

        i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
        t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}
        # print(i2t_metrics)
        # print(t2i_metrics)
        metrics = {}
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
                f'{loader_name}.{metric_name}': v 
                for metric_name, v in result.items()
            }
            
            loader_metrics.update(result)
            final_sum += result[f'{loader_name}.rsum']
        
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
