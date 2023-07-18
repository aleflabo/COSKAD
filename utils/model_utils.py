from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from functools import partial
import torch.optim as optim
from typing import List, Tuple
import numpy as np
import torch
from glob import glob
import os 

class DelayerScheduler(_LRScheduler):
    """ 
    Starts with a flat lr schedule until it reaches N epochs 
    then applies a scheduler.
	
    Args:
		optimizer (Optimizer): Wrapped optimizer.
		delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
		after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
	"""
    def __init__(self, optimizer, delay_epochs, after_scheduler):
        super(DelayerScheduler, self).__init__()
        
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
    
    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        return self.base_lrs
    
    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(DelayerScheduler, self).step(epoch)




def get_optim_and_scheduler(model, epochs=10, **optim_args):
    optim_name = optim_args.get('opt_optimizer', 'adam')
    optim_lr = optim_args.get('opt_lr', 1e-4)
    optim_weight_decay = optim_args.get('opt_weight_decay', 1e-5)
    sched_name = optim_args.get('opt_scheduler', 'step')

    if optim_name.lower() == 'adam':
        optimizer = Adam #, weight_decay=optim_weight_decay)
        optimizer = partial(optimizer,lr=optim_lr)
        # optimizer = optimizer(model.parameters(), lr=optim_lr)

    else:
        print('No Optim defined with this name {}'.format(optim_name))
    
    if sched_name.lower() == 'step':
        step_size = 5
        gamma = 0.99
        
        sched = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif (sched_name.lower() == 'tri') and (epochs >= 8):
        sched = partial(optim.lr_scheduler.CyclicLR,
                          base_lr=optim_lr/10, max_lr=optim_lr*10,
                          step_size_up=epochs//8,
                          mode='triangular2',
                          cycle_momentum=False)
    else:
        print('No Scheduler with this name: {}'.format(sched_name))
        
    return optimizer, sched

def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return optimizer, new_lr, scheduler

def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)
    
    
def light_processing_data(data:List[torch.tensor]) -> Tuple[np.ndarray]:
    
    out = []
    hidden_out = []
    gt_data = []
    trans = []
    meta = []
    frames = []

    for data_array in data:
        if len(data_array) == 6:
            out.append(data_array[0].cpu().numpy())
            gt_data.append(data_array[2].cpu().numpy())
            hidden_out.append(data_array[1].cpu().numpy())
            trans.append(data_array[3].cpu().numpy())
            meta.append(data_array[4].cpu().numpy())
            frames.append(data_array[5].cpu().numpy())
            
        elif len(data_array) == 4:
            hidden_out.append(data_array[0].cpu().numpy())
            trans.append(data_array[1].cpu().numpy())
            meta.append(data_array[2].cpu().numpy())
            frames.append(data_array[3].cpu().numpy())
            
        else:
            hidden_out.append(data_array[0].cpu().numpy())
            trans.append(data_array[2].cpu().numpy())
            meta.append(data_array[3].cpu().numpy())
            frames.append(data_array[4].cpu().numpy())

    hidden_out = np.concatenate(hidden_out, axis=0)
    trans = np.concatenate(trans, axis=0)
    meta = np.concatenate(meta, axis=0)
    frames = np.concatenate(frames, axis=0)
    if len(data_array) == 6:
        out = np.concatenate(out, axis=0)
        gt_data = np.concatenate(gt_data, axis=0)
        return out, hidden_out, gt_data, trans, meta, frames
    else:
        return hidden_out, trans, meta, frames

def hr_ubnormal(path_to_boolean_masks):
    """
    """
    
    hr_boolean_masks = glob(path_to_boolean_masks)
    hr_ubnormal_masked_clips = {}
    
    for boolean_mask_path in hr_boolean_masks:
        scene_clip_id = os.path.basename(boolean_mask_path).split('.')[0]
        scene_id, clip_id = list(map(int, scene_clip_id.split('_')))
        hr_ubnormal_masked_clips[(scene_id, clip_id)] = np.load(boolean_mask_path)
    
    return hr_ubnormal_masked_clips
    
