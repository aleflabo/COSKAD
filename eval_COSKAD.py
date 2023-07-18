import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from utils.model_utils import light_processing_data, hr_ubnormal

from utils.argparser import init_sub_args
from utils.eval_utils import (pad_scores, ROC, score_process, filter_vectors_by_cond,
                              windows_based_loss_hy, windows_based_loss_mahalanobis, windows_based_loss_rec_and_hy)
from utils.dataset import get_dataset_and_loader
import geoopt.manifolds.stereographic.math as gmath

warnings.filterwarnings("ignore")

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






if __name__== '__main__':
    

    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=True,
                       default='./config/old_ckpt.yaml')
    args = parser.parse_args()
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    
    args, ae_args, dcec_args, res_args, opt_args = init_sub_args(args)
    
    # Pass arguments as dataset arguments for PoseDatasetRobust
    exp_dir = os.path.join(args.exp_dir, args.dataset_choice, args.dir_name)
    ae_args.exp_dir = exp_dir

    if args.use_decoder:
        from models.euclidean_autoencoder import LitAutoEncoder as Litmodel
        
        loss_fn = nn.MSELoss(reduction="none") 
        rec_loss_weight = 0

        if rec_loss_weight == 0:
            loss_type = 'hyp'
        elif rec_loss_weight > 100:
            loss_type = 'rec'
        else:
            loss_type = 'rec+hyp'
    elif args.hyperbolic:
        from models.hyperbolic_encoder import \
            LitEncoder as Litmodel

    elif args.use_vae:
        from models.spherical_vae import LitEncoder as Litmodel
    
        loss_fn = lambda x, y: torch.unsqueeze(1 - F.cosine_similarity(x, y), dim=-1)
    else:
        if args.static_center:
            from models.euclidean_encoder_staticCenter import LitEncoder as Litmodel
        else:
            from models.euclidean_encoder_dynamicCenter import LitEncoder as Litmodel
        loss_fn = nn.MSELoss(reduction='none')
        windows_based_loss_hy_e = windows_based_loss_mahalanobis if args.distance == 'mahalanobis' else windows_based_loss_hy

    
    ### For HR UBnormal
    if args.use_hr:
        if 'test' in args.split:
            split = 'testing'
        else:
            split = 'validating'
            
        ubnormal_path_to_boolean_masks = f'/media/odin/data_anomaly/anomaly_detection/UBnormal/hr_bool_masks/{split}/test_frame_mask/*'
        hr_ubnormal_masked_clips = hr_ubnormal(ubnormal_path_to_boolean_masks)
    else:
        hr_ubnormal_masked_clips = {}
    

    print('Done\n')
    print(args.dataset_choice)
    print('Loading data and creating loaders.....')
    dataset, loader = get_dataset_and_loader(ae_args,split=args.split)
    
    # init model
    model = Litmodel(args)
    
    path = os.path.join(args.exp_dir,args.dataset_choice,args.dir_name, args.load_ckpt)
    print('Loading model from {}'.format(path))
    
    trainer = pl.Trainer(strategy="ddp",accelerator=args.accelerator,devices= args.devices)
    out = trainer.predict(model, dataloaders=loader,ckpt_path=path,return_predictions=True)
    if args.use_decoder:
        out, hidden_out, gt_data, trans, meta, frames = light_processing_data(out)
    else:
        hidden_out, trans, meta, frames = light_processing_data(out)

    print('Checkpoint loaded')
    print('Processing data.....')

    print('Dataset: {}, Test path: {}'.format(args.dataset_choice,args.gt_path))

    all_gts = [file_name for file_name in os.listdir(args.gt_path) if file_name.endswith('.npy')]
    all_gts = sorted(all_gts)
    scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]

    model_scores_transf = {}
    dataset_gt_transf = {}

    num_transform = ae_args.num_transform
    idx_transf = 0
    smoothing = args.smoothing
    

    print('Starting evaluation.....')
    for transformation in range(num_transform):
        # iterating over each transformation T
        
        dataset_gt = []
        model_scores = []
        errors = []
        scenes_division = {}
        cond_transform = (trans == transformation)

        if args.use_decoder:
            out_transform, hidden_out_transform, gt_data_transform, meta_transform, frames_transform = filter_vectors_by_cond([out, hidden_out, gt_data, meta, frames], cond_transform)
        else:
            hidden_out_transform, meta_transform, frames_transform = filter_vectors_by_cond([hidden_out, meta, frames], cond_transform)
        
        for idx in range(len(all_gts)):
            # iterating over each clip C with transformation T
            
            scene_idx, clip_idx = scene_clips[idx]
            if not scene_idx in scenes_division.keys():
                scenes_division[scene_idx] = []
            gt = np.load(os.path.join(args.gt_path, all_gts[idx]))
            n_frames = gt.shape[0]
            
            cond_scene_clip = (meta_transform[:, 0] == scene_idx) & (meta_transform[:, 1] == clip_idx)

            if args.use_decoder:
                out_scene_clip, hidden_out_scene_clip, gt_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond([out_transform, hidden_out_transform, gt_data_transform, meta_transform, frames_transform], cond_scene_clip) 
            else:
                hidden_out_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond([hidden_out_transform, meta_transform, frames_transform], cond_scene_clip) 
            
            figs_ids = sorted(list(set(meta_scene_clip[:, 2])))
            error_per_person = []
            actor_poses_gt = []
            actor_poses_out = []
            
            for fig in figs_ids:
                # iterating over each actor A in each clip C with transformation T

                cond_fig = (meta_scene_clip[:, 2] == fig)

                if args.use_decoder:
                    out_fig, hidden_out_fig, gt_fig, frames_fig = filter_vectors_by_cond([out_scene_clip, hidden_out_scene_clip, gt_scene_clip, frames_scene_clip], cond_fig) 
                else:
                    hidden_out_fig, frames_fig = filter_vectors_by_cond([hidden_out_scene_clip, frames_scene_clip], cond_fig) 


                # computing the reconstruction loss for each frame of actor A
                if args.use_decoder:
                    loss_matrix = windows_based_loss_rec_and_hy(gt_fig, out_fig, model.model.c.to(args.device), hidden_out_fig, frames_fig, 
                                                                n_frames, loss_fn, rec_loss_weight=rec_loss_weight, loss_type=loss_type)
                elif args.use_vae:
                    loss_matrix = windows_based_loss_hy(model.model.mean_vector.to(args.device), hidden_out_fig, frames_fig, n_frames, loss_fn)

                elif args.hyperbolic:
                    curvature = torch.tensor(-1.)
                    hyperbolic_latents = gmath.project(gmath.expmap0(torch.tensor(hidden_out_fig), k=curvature), k=curvature).cuda()
                    loss_matrix = windows_based_loss_hy(model.model.c.cuda(), hyperbolic_latents, frames_fig, n_frames, None, args.hyperbolic)
                
                else:
                    loss_matrix = windows_based_loss_hy_e(model.model.c.to(args.device), hidden_out_fig, frames_fig, n_frames, loss_fn)
                    
                loss_matrix = np.where(loss_matrix == 0.0, np.nan, loss_matrix)
                fig_reconstruction_loss = np.nanmean(loss_matrix, 0)
                fig_reconstruction_loss = np.where(np.isnan(fig_reconstruction_loss), 0, fig_reconstruction_loss) 

                if args.pad_size!=-1:
                    fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, args.pad_size)                    
                
                error_per_person.append(fig_reconstruction_loss)
              
            # aggregating the reconstruction errors for all actors in clip C, assigning the maximum error to each frame  
            clip_score = np.amax(np.stack(error_per_person, axis=0), axis=0)

            if (scene_idx, clip_idx) in hr_ubnormal_masked_clips:
                clip_score = clip_score[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                gt = gt[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                
            clip_score = score_process(clip_score, win_size=smoothing, dataname=args.dataset_choice,use_scaler=False)
            scenes_division[scene_idx].append(clip_score)
            model_scores.append(clip_score)
            dataset_gt.append(gt)
            
            try:
                auc=roc_auc_score(gt, clip_score)
                errors.append(auc)
            except Exception as e: 
                print(e)
                pass
            
            if idx%1 == 0:
                print('transf: {}/{}, clip: {},{}/{}, score: {} average_score: {}'.format(transformation+1,num_transform, scene_clips[idx], idx+1,len(all_gts),auc,np.nanmean(np.array(errors).astype(float))))
                
        model_scores = np.concatenate(model_scores, axis=0)
        dataset_gt = np.concatenate(dataset_gt, axis=0)

        print('\nTest set score for transformation {}\n'.format(transformation+1))
        if args.dataset_choice == 'HR-Avenue':
            best_threshold, auc = ROC(dataset_gt, model_scores)
        else:
            best_threshold, auc = ROC(dataset_gt, model_scores, path=path+f'_t{transformation}_roc_hyp.png')
            print(best_threshold)

        print('auc = {}'.format(auc))

        model_scores_transf[transformation] = model_scores
        dataset_gt_transf[transformation] = dataset_gt

    # aggregating the anomaly scores for all transformations
    pds = np.mean(np.stack(list(model_scores_transf.values()),0),0)
    gt = dataset_gt_transf[0]
    
    # computing the AUC
    auc=roc_auc_score(gt,pds)
    print('final AUC score: {}'.format(auc))


