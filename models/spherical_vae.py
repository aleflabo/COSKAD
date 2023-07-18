import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.eval_utils import (filter_vectors_by_cond, pad_scores, score_process,
                              windows_based_loss_hy)
from utils.model_utils import calc_reg_loss, light_processing_data

from models.stsve.stsve_hidden_hypersphere import STSVE

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


class LitEncoder(pl.LightningModule):
    def __init__(self, 
                 args,
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.args = args 
        self.learning_rate = args.opt_lr
        self.batch_size = args.dataset_batch_size
        channels = args.channels # channels for the encoder
        decoder_channels = args.decoder_channels # channels for the auxiliary decoder
        self.phi = args.phi # weight of the reconstruction loss
        self.beta = args.beta # weight of the Kullback-Leibler loss
        self.gamma = args.gamma # weight of the loss defined as the distance from the expected value (enforces points to cluster together)
        self.distribution = args.distribution # prior distribution; alternative 'normal'
        self.warmup_counter = args.warmup_epochs # number of epochs before enforcing the concentration of the distribution
        self.latent_cache = []
        self.updated_state_before_val = False
    
        if args.dataset_headless:
            joints_to_consider = 14
        elif args.dataset_kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17

        self.model = STSVE(c_in=args.num_coords, h_dim=args.h_dim, latent_dim=args.latent_dim, n_frames=args.dataset_seg_len, 
                            dropout=args.dropout, n_joints=joints_to_consider, channels=channels, distribution=self.distribution, 
                            decoder_channels=decoder_channels, projector=args.projector)
        
        
    def forward(self, x:List[torch.tensor]) -> Tuple[torch.tensor]:

        tensor_data = x[0]
        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        hidden_out, recostructed_x, _ = self.model(tensor_data)
        
        return hidden_out, recostructed_x, transformation_idx, metadata, actual_frames
            

    def training_step(self, batch:List[torch.tensor], batch_idx:int) -> torch.float32:
        
        DATA_DISTRO, TARGET_DISTRO, Z_VAR = 0, 1, 2

        data = batch[0]
        hidden_out, reconstructed_x, distributions = self.model(data)
        self.latent_cache.append(hidden_out.detach())
                
        if self.distribution == 'normal':
            loss_kl = torch.distributions.kl.kl_divergence(distributions[DATA_DISTRO], distributions[TARGET_DISTRO]).sum(-1).mean()
        elif self.distribution == 'ps':
            loss_kl = torch.distributions.kl.kl_divergence(distributions[DATA_DISTRO], distributions[TARGET_DISTRO]).mean()
        else:
            raise NotImplemented
        
        loss_rec = F.mse_loss(reconstructed_x, data)
        loss_reg = calc_reg_loss(self.model)
        loss_exp_dist = (1/distributions[Z_VAR]).mean() # 2.0 if self.warmup_counter else (1 - F.cosine_similarity(hidden_out, self.model.mean_vector)).mean() # constant loss during warmup
            
        loss = self.phi * loss_rec + self.args.alpha * loss_reg + self.beta * loss_kl + self.gamma * loss_exp_dist
        
        self.log("loss",loss)
        self.log("reconstruction_loss",loss_rec)
        self.log("kl_loss", loss_kl)
        self.log("exp_dist_loss", loss_exp_dist)
        self.log("regularization",loss_reg)
        return loss
    
    
    def update_state(self) -> None:
        
        latent_vecs = torch.cat(self.latent_cache, dim=0)
        self.model.mean_vector = latent_vecs.mean(dim=0, keepdim=True) # update the mean vector with the empirical expectation of the latent distribution
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
        self.latent_cache = []
    
    
    def on_validation_start(self) -> None:
        
        self.update_state()
        self.updated_state_before_val = True
    
    
    def on_train_epoch_end(self) -> None:
        
        if not self.updated_state_before_val:
            self.update_state()
        self.updated_state_before_val = False
    
    
    def validation_step(self, batch:List[torch.tensor], batch_idx:int) -> torch.float32:
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
                                                                   patience=2,
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
        loss_fn = lambda x, y: torch.unsqueeze(1 - F.cosine_similarity(x, y), dim=-1)
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
                    
                    # computing the reconstruction loss for each frame of actor A
                    loss_matrix = windows_based_loss_hy(self.model.mean_vector, hidden_out_fig, frames_fig, n_frames, loss_fn)
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
        auc = roc_auc_score(gt,pds)
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

