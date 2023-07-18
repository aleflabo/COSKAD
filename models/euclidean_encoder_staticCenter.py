
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.eval_utils import (pad_scores, filter_vectors_by_cond, mahalanobis,
                              score_process, windows_based_loss_hy,
                              windows_based_loss_mahalanobis)
from utils.model_utils import calc_reg_loss, light_processing_data
from typing import Dict, List, Tuple, Union

from models.stse.stse_hidden_hypersphere import STSE

V_01 = [1] * 75 + [0] * 46 + [1] * 269 + [0] * 47 + [1] * 427 + [0] * 47 + [1] * 20 + [0] * 70 + [1] * 438  # 1439 Frames
V_02 = [1] * 272 + [0] * 48 + [1] * 403 + [0] * 41 + [1] * 447  # 1211 Frames
V_03 = [1] * 293 + [0] * 48 + [1] * 582  # 923 Frames
V_04 = [1] * 947  # 947 Frames
V_05 = [1] * 1007  # 1007 Frames
V_06 = [1] * 561 + [0] * 64 + [1] * 189 + [0] * 193 + [1] * 276  # 1283 Frames
V_07_to_15 = [1] * 6457
V_16 = [1] * 728 + [0] * 12  # 740 Frames
V_17_to_21 = [1] * 1317
AVENUE_MASK = np.array(V_01 + V_02 + V_03 + V_04 + V_05 + V_06 + V_07_to_15 + V_16 + V_17_to_21) == 1

masked_clips = {
    1: V_01,
    2: V_02,
    3: V_03,
    6: V_06,
    16: V_16
}

def batch_cov_mat_step(X:torch.tensor, mu:torch.tensor):
    
    if len(X.size()) < 3:
        X = torch.reshape(X, (*X.size(), 1))
    if len(mu.size()) < 3:
        mu = torch.reshape(mu, (*mu.size(), 1))
    return torch.sum(torch.matmul((X - mu), torch.transpose(X - mu, 1, 2)), dim=0)



class LitEncoder(pl.LightningModule):
    """
    """
    
    def __init__(self, args):
        """
        """
        
        super().__init__()
        self.save_hyperparameters()
        
        self.args = args 
        self.learning_rate = args.opt_lr
        self.batch_size = args.dataset_batch_size
        self.eps = args.center_tolerance # tolerance value for the center of the latent hypersphere initialization
        self.distance = args.distance
        channels = args.channels
        self.device_ = args.device
        self.hidden_out_cache = []
    
        if args.dataset_headless:
            joints_to_consider = 14
        elif args.dataset_kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17        

        self.model = STSE(c_in=args.num_coords, h_dim=args.h_dim, latent_dim=args.latent_dim, 
                            n_frames=args.dataset_seg_len, dropout=args.dropout, 
                            n_joints=joints_to_consider, channels=channels, 
                            distance=self.distance, encoder_type=args.encoder_type, projector=args.projector)
        
        
        
    def forward(self, x:List[torch.tensor]) -> Tuple[torch.tensor]:

        tensor_data = x[0]
        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        hidden_out = self.model(tensor_data)
        
        return hidden_out, transformation_idx, metadata, actual_frames
    
    
    def setup(self, stage:str=None) -> None:
        
        super().setup(stage)
        
        if stage == 'fit':
            
            print("Initialize the center of the latent hypersphere")
            
            n_samples = 0
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
            c = torch.zeros(self.model.latent_dim, device=self.device_)

            self.model.eval()
            self.model.to(self.device_)
            with torch.no_grad():
                for batch in tqdm(train_loader):
                    # get the inputs of the batch
                    data = batch[0].to(self.device_)
                    hidden_out = self.model(data)
                    n_samples += hidden_out.size(0)
                    c += torch.sum(hidden_out, dim=0)
                    self.hidden_out_cache.append(hidden_out)
                    
            c /= n_samples

            c[(abs(c) < self.eps) & (c < 0)] = -self.eps 
            c[(abs(c) < self.eps) & (c > 0)] = self.eps 

            self.model.c = c # initialize the center of the hypersphere
            self.temp = c
            if self.distance == 'mahalanobis':
                self.compute_inv_cov_mat(c) # compute the inverse of the covariance matrix
                self.register_buffer('mu', torch.zeros_like(c)) # running mean
                self.batches = 0 # number of batches
                
            self.model.train()
            
            
    def compute_inv_cov_mat(self, mu:torch.tensor) -> None:
        sum_cov_mat = torch.zeros_like(self.model.inv_cov_matrix)
        n_samples = 0
        
        for hidden_out in self.hidden_out_cache:
            # get the inputs of the batch
            sum_cov_mat += batch_cov_mat_step(hidden_out.detach(), mu)
            n_samples += hidden_out.size(0)
            
        self.model.inv_cov_matrix = torch.inverse(sum_cov_mat / (n_samples - 1))
        
    
    def on_train_epoch_end(self) -> None:
        
        if self.distance == 'mahalanobis':
            self.compute_inv_cov_mat(self.model.c)        
        if not self.args.static_center:
            c = self.cumt / self.n_samples
            
            c[(abs(c) < self.eps) & (c < 0)] = -self.eps 
            c[(abs(c) < self.eps) & (c > 0)] = self.eps 

            self.temp = c

    
    def on_train_epoch_start(self) -> None:
        if self.distance == 'mahalanobis':
            self.hidden_out_cache = []
        return super().on_train_epoch_start()
            

    def training_step(self, batch:List[torch.tensor], batch_idx:int) -> torch.float32:

        data = batch[0]
        hidden_out = self.model(data) 
        if (not self.args.static_center):
            self.model.c = self.temp
            with torch.no_grad():
                if batch_idx == 0:
                    self.cumt = torch.zeros(self.model.latent_dim, device=self.device_)
                    self.cumt += torch.sum(hidden_out.clone(), dim=0)
                    self.n_samples = hidden_out.size(0)
                else:
                    self.cumt += torch.sum(hidden_out.clone(), dim=0)
                    # self.cumt += torch.sum(hidden_out.clone(), dim=0)
                    self.n_samples += hidden_out.size(0)

        loss_reg = calc_reg_loss(self.model)
        
        if self.distance == 'mahalanobis':
            self.hidden_out_cache.append(hidden_out)
            self.batches += hidden_out.size(0)
            loss_hypersphere = mahalanobis(hidden_out, self.model.c, self.model.inv_cov_matrix)
        else:
            loss_hypersphere = F.mse_loss(hidden_out, self.model.c)
            
        loss = loss_hypersphere + self.args.alpha*loss_reg
        
        self.log("loss",loss)
        self.log("hypersphere_loss",loss_hypersphere)
        self.log("regularization",loss_reg)
        return loss
    
    
    def validation_step(self, batch:List[torch.tensor], batch_idx:int) -> Tuple[torch.tensor]:
        return self.forward(batch)
    

    def validation_epoch_end(self, validation_step_outputs:Union[Tuple[torch.tensor],List[torch.tensor]]):
        hidden_out, trans, meta, frames = light_processing_data(validation_step_outputs)
        return self.post_processing(hidden_out, trans, meta, frames)
    
    
    def configure_optimizers(self) -> Dict:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        if self.args.validation:

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='max',
                                                                   factor=0.2,
                                                                   patience=100,
                                                                   min_lr=1e-6,
                                                                   verbose=True)

            return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'validation_auc'}
        
        else:
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.args.ae_epochs,
                                                                   eta_min=self.args.opt_lr)
            return {'optimizer':optimizer,'lr_scheduler':scheduler}
        

    def post_processing(self, hidden_out:np.ndarray, trans:int, meta:np.ndarray, frames:np.ndarray) -> float:
        all_gts = [file_name for file_name in os.listdir(self.args.gt_path) if file_name.endswith('.npy')]
        all_gts = sorted(all_gts)
        scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]

        num_transform = self.args.dataset_num_transform
        loss_fn = nn.MSELoss(reduction='none')
        smoothing = self.args.smoothing
        model_scores_transf = {}
        dataset_gt_transf = {}

        for transformation in tqdm(range(num_transform)):
            # iterating over each transformation T
            
            dataset_gt = []
            model_scores = []
            cond_transform = (trans == transformation)

            hidden_out_transform, meta_transform, frames_transform = filter_vectors_by_cond([hidden_out, meta, frames], cond_transform)


            for idx in range(len(all_gts)):
                # iterating over each clip C with transformation T
                
                scene_idx, clip_idx = scene_clips[idx]
                gt = np.load(os.path.join(self.args.gt_path, all_gts[idx]))
                n_frames = gt.shape[0]
                
                cond_scene_clip = (meta_transform[:, 0] == scene_idx) & (meta_transform[:, 1] == clip_idx)
                hidden_out_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond([hidden_out_transform, meta_transform, frames_transform], cond_scene_clip) 
                
                figs_ids = sorted(list(set(meta_scene_clip[:, 2])))
                error_per_person = []
                
                for fig in figs_ids:
                    # iterating over each actor A in each clip C with transformation T
                    
                    cond_fig = (meta_scene_clip[:, 2] == fig)
                    hidden_out_fig, frames_fig = filter_vectors_by_cond([hidden_out_scene_clip, frames_scene_clip], cond_fig) 
                    
                    if self.distance == 'mahalanobis':
                        loss_matrix = windows_based_loss_mahalanobis(self.model.c, hidden_out_fig, self.model.inv_cov_matrix, 
                                                                     frames_fig, n_frames)
                    else:
                        loss_matrix = windows_based_loss_hy(self.model.c, hidden_out_fig, frames_fig, n_frames, loss_fn)
                    
                    # computing the reconstruction loss for each frame of actor A
                    loss_matrix = np.where(loss_matrix == 0.0, np.nan, loss_matrix)
                    fig_hypersphere_loss = np.nanmean(loss_matrix, 0)
                    fig_hypersphere_loss = np.where(np.isnan(fig_hypersphere_loss), 0, fig_hypersphere_loss) 
                    
                    if self.args.pad_size!=-1:
                        fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, self.args.pad_size)                    
                    
                    error_per_person.append(fig_hypersphere_loss)
                    
                # aggregating the reconstruction errors for all actors in clip C, assigning the maximum error to each frame
                clip_score = np.amax(np.stack(error_per_person, axis=0), axis=0)
                
                # removing the non-HR frames for Avenue dataset
                if clip_idx in masked_clips:
                    clip_score = clip_score[np.array(masked_clips[clip_idx])==1]
                    gt = gt[np.array(masked_clips[clip_idx])==1]

                clip_score = score_process(clip_score, win_size=smoothing, dataname=self.args.dataset_choice, use_scaler=False)
                model_scores.append(clip_score)
                dataset_gt.append(gt)
                    
            model_scores = np.concatenate(model_scores, axis=0)
            dataset_gt = np.concatenate(dataset_gt, axis=0)
            
            model_scores_transf[transformation] = model_scores
            dataset_gt_transf[transformation] = dataset_gt

        # aggregating the anomaly scores for all transformations
        pds = np.mean(np.stack(list(model_scores_transf.values()),0),0)
        gt = dataset_gt_transf[0]
        
        # computing the AUC
        auc=roc_auc_score(gt,pds)
        self.log('validation_auc', auc)

        return auc


# using LightningDataModule
class LitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dataset):
        super().__init__()
        self.save_hyperparameters()
        # or
        self.batch_size = batch_size
        self.train_dataset = train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size | self.hparams.batch_size, num_workers=8, pin_memory=True)


