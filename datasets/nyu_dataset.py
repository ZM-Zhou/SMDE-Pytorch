import os
import random

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data

from datasets.utils.data_reader import h5_loader
from datasets.utils.my_transforms import (BatchRandomCrop, NoneTransform,
                                          do_fal_color_aug)
from path_my import Path
from utils import platform_manager

@platform_manager.DATASETS.add_module
class NYUv2_Dataset(data.Dataset):

    def __init__(
            self,
            dataset_mode,
            split_file,
            full_size=None,
            patch_size=None,
            random_resize=True,
            is_KTmatrix=False,
            normalize_params=[0.411, 0.432, 0.45],
            flip_mode=None,
            color_aug=True,
            multi_out_scale=None):
        super().__init__()
        self.init_opts = locals()

        self.dataset_mode = dataset_mode
        self.dataset_dir = Path.get_path_of('nyuv2')
        self.split_file = split_file
        self.full_size = full_size
        self.patch_size = patch_size
        self.random_resize = random_resize
        self.is_KTmatrix = is_KTmatrix
        self.flip_mode = flip_mode
        self.color_aug = color_aug
        self.multi_out_scale = multi_out_scale
        self.file_list = self._get_file_list(split_file)

        self.full_res_shape = (640-16*2, 480-16*2) 
        self.K = self._get_intrinsics()

        # Initialize transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])
        if dataset_mode == 'train':
            # random resize and crop
            if self.patch_size is not None:
                self.crop = BatchRandomCrop(patch_size)
                self.canny = None
            else:
                self.crop = NoneTransform()
                self.canny = None
        else:
            if self.full_size is not None:
                self.color_resize = tf.Resize(full_size,
                                              interpolation=Image.ANTIALIAS)
            else:
                self.color_resize = NoneTransform()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        """Return a data dictionary."""
        file_info = self.file_list[f_idx]
        inputs = {}
        
        if self.dataset_mode == 'Train':
            # coming soon
            pass
        else:
            line = os.path.join(self.dataset_dir, file_info)
            rgb, depth, _, _ = h5_loader(line) 

            rgb = rgb[44: 471, 40: 601, :]
            depth = depth[44: 471, 40: 601]

            rgb = Image.fromarray(rgb)
            depth = Image.fromarray(depth)

            rgb = self.normalize(self.to_tensor(self.color_resize(rgb)))
            depth = self.to_tensor(depth)

            K = self.K.copy()
            K[0, :] *= self.full_size[1]
            K[1, :] *= self.full_size[0]
            inv_K = np.linalg.pinv(K)

            inputs['color_s'] = rgb
            inputs['depth'] = depth
            inputs['K'] = torch.from_numpy(K).to(torch.float)
            inputs['inv_K'] = torch.from_numpy(inv_K).to(torch.float)

       
        inputs['file_info'] = [file_info]
        return inputs

    def _get_file_list(self, split_file):
        with open(split_file, 'r') as f:
            files = f.readlines()
            filenames = []
            for f in files:
                file_name = f.replace('\n', '')
                filenames.append(file_name)
        return filenames

    def _get_intrinsics(self):
        # 640, 480
        w, h = self.full_res_shape
        
        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = 3.2558244941119034e+02 / w
        cy = 2.5373616633400465e+02 / h

        intrinsics =np.array([[fx, 0., cx, 0.], 
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics

    @property
    def dataset_info(self):
        infos = []
        infos.append('    -{} Datasets'.format(self.dataset_mode))
        infos.append('      get {} of data'.format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ['self', '__class__']:
                infos.append('        {}: {}'.format(key, val))
        return infos
