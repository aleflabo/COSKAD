import argparse
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from utils.argparser import init_sub_args
from utils.dataset import get_dataset_and_loader



if __name__== '__main__':
    
    
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=False, default='/media/odin/stdrr/projects/anomaly_detection/code/COSKAD/clean_code/HRAD_lightning/config/UBnormal/hypersphere_encoder_cfg.yaml')
    
    args = parser.parse_args()
    config_path = args.config
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    
    args, dataset_args, ae_args, res_args, opt_args = init_sub_args(args)
    
    exp_dir = os.path.join(args.exp_dir, args.dataset_choice, args.dir_name)
    
    # Pass arguments as dataset arguments for PoseDatasetRobust
    dataset_args.exp_dir = exp_dir

    os.system(f'cp {config_path} {os.path.join(exp_dir, "config.yaml")}')     


    if args.use_decoder:
        from models.euclidean_autoencoder import \
            LitAutoEncoder as Litmodel   
        project_name = "AE_" + args.project_name    
    elif args.use_vae:
        from models.spherical_vae import \
            LitEncoder as Litmodel
        project_name = "VAE_" + args.project_name
    elif args.hyperbolic:
        from models.hyperbolic_encoder import \
            LitEncoder as Litmodel
        project_name = "Hyper_" + args.project_name
    else:
        if args.static_center:
            from models.euclidean_encoder_staticCenter import \
                LitEncoder as Litmodel
        else:
            from models.euclidean_encoder_dynamicCenter import \
                LitEncoder as Litmodel
        project_name = "E_" + args.project_name
    
    
    
    if args.validation:
        args.gt_path = os.path.join(args.data_dir, 'validating/test_frame_mask/')
    

    model = Litmodel(args)

    if args.use_wandb:
        wandb_logger = WandbLogger(project=args.project_name, group=args.group_name, entity=args.wandb_entity, name=args.dir_name, config=args.__dict__,log_model='all')
    else:
        wandb_logger = None
        
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=2,
                                          monitor="validation_auc" if (dataset_args.choice == 'UBnormal' or args.validation) else 'loss',
                                          mode="max" if (dataset_args.choice == 'UBnormal' or args.validation) else 'min'
                                         )

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.ckpt_dir, 
                         logger = wandb_logger, log_every_n_steps=20, max_epochs=args.ae_epochs,
                         callbacks=[checkpoint_callback], check_val_every_n_epoch=1, num_sanity_val_steps=0, 
                         strategy = DDPStrategy(find_unused_parameters=False))

    if args.validation:
        train_dataset, train_loader, val_dataset, val_loader = get_dataset_and_loader(dataset_args, split=args.split, validation=args.validation)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        train_dataset, train_loader = get_dataset_and_loader(dataset_args, split=args.split)
        trainer.fit(model=model, train_dataloaders=train_loader)
    
