# import all missing libraries from pytorch lightning
import pytorch_lightning as pl
from utils.model_utils import calc_reg_loss, light_processing_data
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
import torch.nn as nn
from utils.eval_utils import (pad_scores, score_process, filter_vectors_by_cond, 
                              windows_based_loss_hy)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from models.stse.stse_hidden_hypersphere import STSE
import torch

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
    
        if args.dataset_headless:
            joints_to_consider = 14
        elif args.dataset_kp18_format:
            joints_to_consider = 18
        else:
            joints_to_consider = 17
            
        channels = [32,16,32]
        self.eps = args.center_tolerance # tolerance value for the center of the latent hypersphere initialization

        self.model = STSE(c_in=args.num_coords, h_dim=args.h_dim, latent_dim=args.latent_dim, n_frames=args.dataset_seg_len, 
                        dropout=args.dropout, n_joints=joints_to_consider, channels=channels,
                        encoder_type=args.encoder_type) # num_centers=args.num_centers
            
        self.args = args 
        self.learning_rate = args.opt_lr
        self.batch_size = args.dataset_batch_size
        
        self.centers = list()
        
    def forward(self, x):

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
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader() # self.trainer.train_dataloader
            c = torch.zeros(self.model.latent_dim).cuda()

            self.model.eval()
            self.model.cuda()
            with torch.no_grad():
                for batch in tqdm(train_loader):
                    # get the inputs of the batch
                    data = batch[0].cuda()
                    hidden_out = self.model(data)
                    n_samples += hidden_out.size(0)
                    c += torch.sum(hidden_out, dim=0)
                    
            c /= n_samples

            self.centers.append(c) # initialize the center of the hypersphere
            self.n_samples = n_samples
            print(self.model.c)
            

    def training_step(self, batch, batch_idx):

        data = batch[0]
        hidden_out = self.model(data)
        
        with torch.no_grad():
            # sum current point to center if epoch > 1
            if len(self.centers) > 1:
                self.centers[-1] += hidden_out.sum(axis=0)
        
        loss_reg = calc_reg_loss(self.model)
        loss_hypersphere = F.mse_loss(hidden_out, self.model.c)
        loss = loss_hypersphere + self.args.alpha*loss_reg
        
        self.log("loss",loss)
        self.log("hypersphere_loss",loss_hypersphere)
        self.log("regularization",loss_reg)
        return loss


    def training_epoch_end(self, outputs):
        # at each training_step the center accumulates the output hidden value
        # which must be then normalized at the end of the epoch (if epoch > 1)
        if len(self.centers) > 1:
            c = self.centers[-1]
            c = c / self.n_samples
            c[(abs(c) < self.eps) & (c < 0)] = -self.eps
            c[(abs(c) < self.eps) & (c > 0)] = self.eps
            self.centers[-1] = c
        
        # update center inside the model
        self.model.c = self.centers[-1]
        
        # add an empty center for the next epoch
        self.centers.append(torch.zeros(self.model.latent_dim).cuda())
        
        print("Update center of hypersphere")
        print(self.model.c)
    
    
    def validation_step(self, batch, batch_idx):
        return self.forward(batch)

    def validation_epoch_end(self, validation_step_outputs):
        hidden_out, trans, meta, frames = light_processing_data(validation_step_outputs)
        return self.post_processing(hidden_out, trans, meta, frames)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='max',
                                                        factor=0.2,
                                                        patience=2,
                                                        min_lr=1e-6,
                                                        verbose=True,
                                                       )

        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'validation_auc'}

    def post_processing(self, hidden_out, trans, meta, frames):
        all_gts = [file_name for file_name in os.listdir(self.args.gt_path) if file_name.endswith('.npy')]
        all_gts = sorted(all_gts)
        scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]


        num_transform = self.args.dataset_num_transform
        loss_fn = nn.MSELoss(reduction='none')
        smoothing = 50
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
                    loss_matrix = windows_based_loss_hy(self.model.c, hidden_out_fig, frames_fig, n_frames, loss_fn, )
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


