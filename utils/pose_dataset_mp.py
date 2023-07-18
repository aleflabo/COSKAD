import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset_utils import keypoints17_to_coco18, normalize_pose, gen_clip_seg_data_np, ae_trans_list

from argparser import init_parser, init_sub_args

class PoseDatasetMP(Dataset):
    def __init__(self, path_to_json_dir,
                 transform_list=None,
                 return_indices=False, return_metadata=False,
                 debug=False, dataset_clips=None,
                 **dataset_args):
        super(PoseDatasetMP).__init__()
        self.path_to_json = path_to_json_dir
        
        self.headless = dataset_args.get('headless', False)
        self.normalize_pose_seg = dataset_args.get('normalize_pose', True)
        self.kp18_format = dataset_args.get('kp18_format', True)
        self.vid_res = dataset_args.get('vid_res', [856, 480])
        self.num_coords = dataset_args.get('num_coords', 2)
        self.return_mean = dataset_args.get('return_mean', True)
        self.debug = debug
        num_clips = 5 if debug else None
        if dataset_clips:
            num_clips = dataset_clips
        self.num_clips = num_clips
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        
        if (transform_list is None) or (transform_list == []):
            self.apply_transform = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(transform_list)
        
        self.transform_list = transform_list
        self.start_ofst = dataset_args.get('start_offset', 0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        if (self.return_mean) & (self.normalize_pose_seg):
            self.segs_data_np, self.segs_meta, self.person_keys, self.segs_ids, self.segs_mean = self.gen_dataset(ret_keys=True,
                                                                                                                  **dataset_args)
        else:
            self.segs_data_np, self.segs_meta, self.person_keys, self.segs_ids = self.gen_dataset(ret_keys=True,
                                                                                                  **dataset_args)
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.segs_meta = np.array(self.segs_meta)
        self.segs_ids = np.array(self.segs_ids)
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape
    
    def __getitem__(self, index: int):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = index // self.num_samples
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
            data_transformed = data_transformed[:self.num_coords,:,:]
        else:
            sample_index = index
            data_transformed = data_numpy = np.array(self.segs_data_np[index])
            data_transformed = data_transformed[:self.num_coords,:,:]
            trans_index = 0  # No transformations
        seg_metadata = self.segs_meta[sample_index]
        self.ids = self.segs_ids[sample_index]
        if (self.return_mean) & (self.normalize_pose_seg):
            seg_mean = self.segs_mean[sample_index]
        
        ret_arr = [data_transformed, trans_index]
        if self.return_metadata:
            ret_arr += [seg_metadata]
            ret_arr += [self.ids]
        if (self.return_mean) & (self.normalize_pose_seg):
            ret_arr += [seg_mean]

        if self.return_indices:
            ret_arr += [index]
        return ret_arr
    
    def gen_dataset(self, ret_keys=False, **dataset_args):
        
        segs_data_np = []
        segs_meta = []
        segs_ids = []
        
        person_keys = dict()
        
        dir_list = os.listdir(self.path_to_json)
        json_list = sorted([fn for fn in dir_list if fn.endswith('.json')])
        
        if self.num_clips is not None:
            json_list = json_list[:self.num_clips]
        
        for person_dict_fn in json_list:
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            clip_json_path = os.path.join(self.path_to_json, person_dict_fn)
            with open(clip_json_path, 'r') as f:
                clip_dict = json.load(f)
            clip_segs_data_np, clip_segs_meta, clip_keys, clip_segs_ids = gen_clip_seg_data_np(clip_dict=clip_dict, 
                                                                                               start_ofst=self.start_ofst, 
                                                                                               seg_stride=self.seg_stride,
                                                                                               seg_len=self.seg_len, 
                                                                                               scene_id=scene_id,
                                                                                               clip_id=clip_id, 
                                                                                               ret_keys=ret_keys)
            segs_data_np.append(clip_segs_data_np)
            segs_meta += clip_segs_meta
            segs_ids += clip_segs_ids
            person_keys = {**person_keys, **clip_keys}
        
        segs_data_np = np.concatenate(segs_data_np, axis=0)
        
        if self.kp18_format and segs_data_np.shape[-2] == 17:
            segs_data_np = keypoints17_to_coco18(segs_data_np)
        
        if self.headless:
            segs_data_np = segs_data_np[:,:,:14]
        
        if self.normalize_pose_seg:
            if self.return_mean:
                segs_data_np, segs_means = normalize_pose(segs_data_np,
                                                          **dataset_args)
            else:
                segs_data_np = normalize_pose(segs_data_np,
                                              **dataset_args)
        
        segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
        
        if ret_keys:
            if (self.return_mean) & (self.normalize_pose_seg):
                return segs_data_np, segs_meta, person_keys, segs_ids, segs_means
            else:
                return segs_data_np, segs_meta, person_keys, segs_ids
        else:
            if (self.return_mean) & (self.normalize_pose_seg):
                return segs_data_np, segs_meta, segs_ids, segs_means
            else:
                return segs_data_np, segs_meta, segs_ids

    def __len__(self):
        return self.num_transform * self.num_samples


def get_dataset_and_loader(args):

    trans_list = ae_trans_list[:args.num_transform]

    dataset_args = {'transform_list': trans_list, 'debug': args.debug, 'headless': args.headless,
                    'seg_len': args.seg_len, 'normalize_pose': args.normalize_pose, 'kp18_format': args.kp18_format,
                    'vid_res': args.vid_res, 'num_coords': args.num_coords, 'sub_mean': args.sub_mean,
                    'return_indices': False, 'return_metadata': True, 'return_mean': args.sub_mean,
                    'symm_range': args.symm_range, 'hip_center': args.hip_center}

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset, loader = dict(), dict()
    for split in ['train', 'test']:
        dataset_args['seg_stride'] = args.seg_stride if split is 'train' else 1  # No strides for test set
        dataset[split] = PoseDatasetMP(args.pose_path[split], **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    return dataset, loader


if __name__ == '__main__':
    parser = init_parser(default_data_dir='../data/STC')
    args = parser.parse_args()
    args, dataset_args, ae_args, res_args, opt_args = init_sub_args(args)

    dataset, loader = get_dataset_and_loader(dataset_args)