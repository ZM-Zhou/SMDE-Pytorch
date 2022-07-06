import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as tf

from datasets.utils.data_reader import get_input_img, get_camera_params
from datasets.utils.my_transforms import BatchRandomCrop, NoneTransform
from path_my import Path
from utils import platform_manager


K_of_KITTI = [[721.54, 0, 609.56, 0],
              [0, 721.54, 172.85, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]

@platform_manager.DATASETS.add_module
class CityscapesColorDataset(data.Dataset):
    DATA_NAME_DICT = {'color_l': ('leftImg8bit', 'leftImg8bit', "png"),
                      'color_r': ('rightImg8bit', 'rightImg8bit', "png"),
                      'semantic': ('gtFine', 'gtFine_labelTrainIds', "png"),
                      'disp': ('disparity', 'disparity', "png"),
                      'c_params': ('matrix', 'matrix', "npy"),
                      }

    def __init__(self, 
                 dataset_mode,
                 split_file,
                 crop_coords=[0, 0, 768, 2048],
                 full_size=None,
                 patch_size=None,
                 normalize_params=[0.411, 0.432, 0.45],
                 flip_mode=None, # "img", "k", "both", "semantic"(lr, ud, rotation)
                 load_KTmatrix=False,
                 load_disp=False,
                 load_semantic=True,
                 fuse_kitti=False,
                 use_casser_test=True,
                 load_test_gt=True):
        self.init_opts = locals()

        self.dataset_mode = dataset_mode
        self.dataset_dir = Path.get_path_of('cityscapes')
        self.split_file = split_file

        self.crop_coords = crop_coords
        self.full_size = full_size
        self.patch_size = patch_size
        self.flip_mode = flip_mode

        self.load_semantic = load_semantic
        self.load_disp = load_disp
        self.load_KTmatrix = load_KTmatrix

        self.fuse_kitti = fuse_kitti

        self.use_casser_test = use_casser_test
        self.load_test_gt = load_test_gt

        self.file_list = self._get_file_list(split_file)
        
        if self.load_semantic:
            assert self.flip_mode not in ['img', 'both'],\
                "To ensure the alignment of semantic label, image order cannot be swapped."
        if self.load_disp:
            assert self.flip_mode != "semantic",\
                "Images can't br rotated and flipped up and down when reading matching information."
        if self.load_test_gt:
            assert not self.load_disp,\
                "When the ground-truth depths are used, do not load the raw dispairties."
        
        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])
        if dataset_mode == "train":
            # random resize and crop
            if self.patch_size is not None:
                self.crop = BatchRandomCrop(patch_size)
            else:
                self.crop = NoneTransform()
            # just resize
            if self.full_size is not None and self.patch_size is None:
                pass
        else:
            if self.load_test_gt:
                self.gt_depths = os.path.join(self.dataset_dir, "gt_depths")
            if self.use_casser_test:
                self.crop_coords = [0, 0, 768, 2048]
                self.full_size = [192, 512]
            if self.full_size is not None:
                self.color_resize = tf.Resize(self.full_size,
                                              interpolation=Image.ANTIALIAS)
            else:
                self.color_resize = NoneTransform()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, f_idx):
        '''Reture a data dictionary'''
        file_info = self.file_list[f_idx]
        base_path = os.path.join(self.dataset_dir,
                                 file_info.replace("leftImg8bit", "{}")\
                                          .replace("png", "{}"))
       
        # Read data
        inputs = {}
        color_l_path = base_path.format(*self.DATA_NAME_DICT['color_l'])
        inputs['color_s_raw'] = get_input_img(color_l_path)
        # img_W, img_H = inputs['color_s_raw'].size
        img_W = self.crop_coords[3] - self.crop_coords[1]
        img_H = self.crop_coords[2] - self.crop_coords[0]

        if self.load_semantic: # read semantic segmentation label
            semantic_path = base_path.format(*self.DATA_NAME_DICT['semantic'])
            inputs['semantic'] = get_input_img(semantic_path, False)
        if self.load_disp: # read disparity
            disp_path = base_path.format(*self.DATA_NAME_DICT['disp'])
            inputs['disparity'] = get_input_img(disp_path, False)
        
        c_params_path = base_path.format(*self.DATA_NAME_DICT['c_params'])
        K, T = get_camera_params(c_params_path)
        T[0, 3] = -T[0, 3]
        if self.fuse_kitti:
            K = np.array(K_of_KITTI)
        else:
            # pre-process matrix for crop image
            K = self._updata_matrix(K)
            if self.full_size is not None:
                K[0, :] *= self.full_size[1] / img_W
                K[1, :] *= self.full_size[0] / img_H
        
        inputs['disp_k'] = torch.tensor(K[0, 0] * T[0, 3], dtype=torch.float)
        inputs['disp_k'] = torch.abs(inputs['disp_k'])
        if self.load_KTmatrix:
            inputs["K"] = torch.from_numpy(K).to(torch.float)
            inputs["inv_K"] = torch.from_numpy(np.linalg.pinv(K))\
                .to(torch.float)
            inputs["T"] = torch.from_numpy(T).to(torch.float)

        if self.fuse_kitti:
            inputs['direct'] = torch.tensor(0.4074, dtype=torch.float)
        else:
            inputs['direct'] = torch.tensor(1, dtype=torch.float)
        
        if self.dataset_mode == "train":
            color_r_path = base_path.format(*self.DATA_NAME_DICT['color_r'])
            inputs['color_o_raw'] = get_input_img(color_r_path)
        if self.dataset_mode == 'test':
            if self.load_test_gt:
                gt_depth = np.load(os.path.join(self.gt_depths, str(f_idx).zfill(3) + '_depth.npy'))
                inputs['depth'] = gt_depth

            
        # Process data
        # resize crop & color jit & flip(roataion) for train
        if self.dataset_mode == "train":
            # crop for image
            if self.crop_coords is not None:
                self.fix_crop = tf.functional.crop

            else:
                self.fix_crop = NoneTransform()

            # flip
            is_flip = (self.dataset_mode == 'train' and
                       self.flip_mode is not None and
                       random.uniform(0, 1) > 0.5)
            if is_flip:
                if self.flip_mode == "semantic":
                    pass #TODO
                else:
                    if self.flip_mode == "both": # random flip mode
                        flip_img = random.uniform(0, 1) > 0.5
                    else:
                        flip_img = self.flip_mode == "img"

                    if flip_img:
                        inputs['color_o_raw'], inputs['color_s_raw'] =\
                            inputs['color_s_raw'], inputs['color_o_raw']
                    else:  # flip_mode == "k"
                        # inputs['disp_k'] = -inputs['disp_k']
                        inputs['direct'] = -inputs['direct']
                        if self.load_KTmatrix:
                            inputs["T"][0, 3] = -inputs["T"][0, 3]
                
            # resize
            if self.full_size is not None:
                img_size = self.full_size
            else:
                _size = inputs["color_s_raw"].size # (w, h)
                img_size = (_size[1], _size[0])
            scale_factor = random.uniform(0.75, 1.5)\
                if self.patch_size is not None else 1
            if scale_factor != 1 or self.full_size is not None:
                self.color_resize = tf.Resize(tuple(int(s * scale_factor)
                                                    for s in img_size),
                                              interpolation=Image.ANTIALIAS)
                self.map_resize = tf.Resize(tuple(int(s * scale_factor)
                                                  for s in img_size),
                                            interpolation=Image.NEAREST)
            else:
                self.color_resize = NoneTransform()
                self.map_resize = NoneTransform()
            # random crop
            crop_params = self.crop.get_params(img_size,
                                               self.patch_size,
                                               scale_factor)
            
            # color jit
            color_aug = tf.ColorJitter.get_params(
                    (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
            
            for key in list(inputs):
                if "color" in key:
                    raw_img = inputs[key]
                    raw_img = self.fix_crop(raw_img, *self.crop_coords)
                    if is_flip:
                        raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                    img = self.crop(self.color_resize(raw_img), crop_params)
                    img = self.to_tensor(img)
                    aug_img = color_aug(img)
                    inputs[key.replace("_raw", "")] =\
                        self.normalize(img)
                    inputs[key.replace("_raw", "_aug")] =\
                        self.normalize(aug_img)

                elif 'disparity' in key:
                    raw_disp = inputs[key]
                    raw_disp = self.fix_crop(raw_disp, *self.crop_coords)
                    if is_flip:
                        raw_disp = raw_disp.transpose(Image.FLIP_LEFT_RIGHT)
                    disp = self.to_tensor(raw_disp) 
                    disp = self.crop(self.map_resize(disp), crop_params)
                    # disp will be changed when resize
                    inputs[key] = disp * (img_size[1] / img_W) * scale_factor / 256
                    mask = (disp == 0).to(torch.float)
                    inputs['depth'] = inputs['disp_k'] / (inputs[key] + mask)
                    inputs['depth'][disp == 0] = 0
                
                elif "semantc" in key:
                    raw_map = inputs[key]
                    raw_map = self.fix_crop(raw_map, *self.crop_coords)
                    if is_flip:
                        raw_map = np.fliplr(raw_map)
                    s_map = torch.from_numpy(raw_map.copy()).unsqueeze(0)
                    s_map = self.crop(self.map_resize(s_map), crop_params)
                    inputs[key] = s_map

        # resize for other modes
        else:
            if self.crop_coords is not None:
                self.fix_crop = tf.functional.crop
            if self.full_size is not None:
                img_size = self.full_size
            else:
                _size = inputs["color_s_raw"].size # (w, h)
                img_size = (_size[1], _size[0])
            for key in list(inputs):
                if "color" in key:
                    raw_img = inputs[key]
                    raw_img = self.fix_crop(raw_img, *self.crop_coords)
                    img = self.color_resize(raw_img)
                    inputs[key.replace("_raw", "")] =\
                        self.normalize(self.to_tensor(img))
                elif 'disparity' in key:
                    raw_disp = inputs[key]
                    raw_disp = self.fix_crop(raw_disp, *self.crop_coords)
                    # disp will be changed when resize
                    inputs[key] = disp * (img_size[1] / img_W) / 256
                    mask = (disp == 0).to(torch.float)
                    inputs['depth'] = inputs['disp_k'] / (inputs[key] + mask)
                    inputs['depth'][disp == 0] = 0
                    inputs[key] = disp
                elif "semantc" in key:
                    raw_map = inputs[key]
                    raw_map = self.fix_crop(raw_map, *self.crop_coords)
                    s_map = torch.from_numpy(raw_map.copy()).unsqueeze(0)
                    inputs[key] = s_map
                elif 'depth' in key:
                    raw_depth = inputs[key]
                    raw_depth = raw_depth[self.crop_coords[0]: self.crop_coords[2], self.crop_coords[1]: self.crop_coords[3]]
                    depth = self.to_tensor(raw_depth)
                    inputs[key] = depth
        
        # delete raw data
        inputs.pop("color_s_raw")
        if self.dataset_mode == 'train':
            inputs.pop("color_o_raw")      
        if self.load_disp:
            inputs.pop("disparity") 
        inputs["file_info"] = [file_info]

        return inputs

            
    def _get_file_list(self, split_file):
        with open(split_file, 'r') as f:
            files = f.readlines()
            filenames = []
            for f in files:
                file_name = f.replace('\n', '')
                filenames.append(file_name)
        return filenames
    
    def _updata_matrix(self, K):
        K[0, 2] -= self.crop_coords[1]
        K[1, 2] -= self.crop_coords[0]
        return K
    
    @property
    def dataset_info(self):
        infos = []
        infos.append('    -{} Datasets'.format(self.dataset_mode))
        infos.append('      get {} of data'.format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ["self", "__class__"]:
                infos.append("        {}: {}".format(key, val))
        return infos
