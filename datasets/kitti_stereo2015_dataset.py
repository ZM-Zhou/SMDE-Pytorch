import os
import random
from collections import namedtuple


import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data

from datasets.utils.data_reader import (get_input_depth, get_input_img,
                                        get_intrinsic)
from datasets.utils.my_transforms import (BatchRandomCrop, NoneTransform,
                                          do_fal_color_aug)
from path_my import Path
from utils import platform_manager

width_to_baseline = dict()
width_to_baseline[1242] = 0.9982 * 0.54
width_to_baseline[1241] = 0.9848 * 0.54
width_to_baseline[1224] = 1.0144 * 0.54
width_to_baseline[1238] = 0.9847 * 0.54
width_to_baseline[1226] = 0.9765 * 0.54
width_to_baseline[1280] = 0.54

K_of_KITTI = [[721.54, 0, 609.56, 0], [0, 721.54, 172.85, 0], [0, 0, 1, 0]]

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

@platform_manager.DATASETS.add_module
class KITTIColorStereoDataset(data.Dataset):
    """KITTI dataset for depth estimation from images."""

    DATA_NAME_DICT = {
        'color_l': ('image_2', 'png'),
        'color_r': ('image_3', 'png'),
        'depth': ('disp_occ_0', 'png'),
        'semantic': ('semantic', 'png')
    }

    def __init__(
            self,
            dataset_mode,
            split_file,
            full_size=None,
            normalize_params=[0.411, 0.432, 0.45],
            load_semantic=False,
            load_KTmatrix=False,
            stereo_test=False,
            ):
        super().__init__()
        self.init_opts = locals()

        self.dataset_mode = dataset_mode
        self.dataset_dir = Path.get_path_of('kitti_stereo2015')
        self.split_file = split_file
        self.full_size = full_size
        self.load_semantic = load_semantic
        self.load_KTmatrix = load_KTmatrix
        self.stereo_test = stereo_test

        self.file_list = self._get_file_list(split_file)

        # Initialize transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])

        if self.full_size is not None:
            self.color_resize = tf.Resize(full_size,
                                            interpolation=Image.ANTIALIAS)
            self.depth_resize = tf.Resize(full_size,
                                            interpolation=Image.NEAREST)
        else:
            self.color_resize = NoneTransform()
            self.depth_resize = NoneTransform()


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        """Return a data dictionary."""
        file_info = self.file_list[f_idx]
        image_base_path = os.path.join(self.dataset_dir,
                                       'training',
                                       '{}', file_info + '.{}') 

        # Read data
        inputs = {}
        color_side_path = image_base_path.format(
            *self.DATA_NAME_DICT['color_l'])
        inputs['color_s_raw'] = get_input_img(color_side_path)

        disp_path = image_base_path.format(*self.DATA_NAME_DICT[
            'depth'])
        inputs['depth_raw'] = get_input_img(disp_path, False)

        if self.load_semantic:
            semantic_path = image_base_path.format(*self.DATA_NAME_DICT[
                'semantic'])
            seg_s = get_input_img(semantic_path, False)
            seg_copy_s = np.array(seg_s.copy())
            for k in np.unique(seg_s):
                seg_copy_s[seg_copy_s == k] = labels[k].trainId
            inputs['seg_s_raw'] = seg_copy_s
    
        if self.load_KTmatrix:
            intric = np.zeros((4, 4))
            intrinsic = np.array(K_of_KITTI)
            intric[:3, :3] = intrinsic[:, :3]
            intric[3, 3] = 1
            if self.full_size is not None:
                img_W, img_H = 1244, 375
                intric[0, :] *= self.full_size[1] / img_W
                intric[1, :] *= self.full_size[0] / img_H

            baseline = -0.54
            extric = torch.tensor([[1, 0, 0, baseline], [0, 1, 0, 0],
                                    [0, 0, 1, 0], [0, 0, 0, 1]])

            inputs['K'] = torch.from_numpy(intric).to(torch.float)
            inputs['inv_K'] = torch.from_numpy(np.linalg.pinv(intric))\
                .to(torch.float)
            inputs['T'] = extric
        inputs['direct'] = torch.tensor(1, dtype=torch.float)  # used to determine the direction of baseline
        
        if self.stereo_test:
            color_oside_path = image_base_path.format(
                *self.DATA_NAME_DICT['color_r'])
            inputs['color_o_raw'] = get_input_img(color_oside_path)

        for key in list(inputs):
            if 'color' in key:
                raw_img = inputs[key]
                raw_img = self.color_resize(raw_img)
                img = self.to_tensor(raw_img)
                inputs[key.replace('_raw', '')] =\
                    self.normalize(img)

            elif 'depth' in key:
                raw_depth = inputs[key]
                raw_depth = np.asarray(raw_depth, dtype=np.float32) / 256.0
                depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                # raw_w = depth.shape[-1]
                # depth = self.depth_resize(depth)
                invalid_mask = (depth == 0)
                # depth = depth / raw_w * self.full_size[1]
                inputs['disp'] = depth.clone()
                depth = 0.54 * 721.54 / (depth + invalid_mask.to(torch.float))
                depth[invalid_mask] = 0
                inputs[key.replace('_raw', '')] = depth
            
            elif 'seg' in key:
                raw_semantic =  inputs[key]
                raw_semantic = np.asarray(raw_semantic, dtype=np.int)
                semantic = torch.from_numpy(raw_semantic.copy()).unsqueeze(0)
                semantic = self.depth_resize(semantic)
                inputs[key.replace('_raw', '')] = semantic
        
        
        # delete the raw data
        inputs.pop('color_s_raw')
        inputs.pop('depth_raw')
        if self.load_semantic:
            inputs.pop('seg_s_raw')
        if self.stereo_test:
            inputs.pop('color_o_raw')

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

    @property
    def dataset_info(self):
        infos = []
        infos.append('    -{} Datasets'.format(self.dataset_mode))
        infos.append('      get {} of data'.format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ['self', '__class__']:
                infos.append('        {}: {}'.format(key, val))
        return infos


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = KITTIColorFlowDataset('test',
                                    'data_splits/kitti_flow/train_list.txt',
                                    normalize_params=[0, 0, 0],
                                    load_semantic=True)
    
    for i in range(len(dataset)):
        inputs = dataset[i]
        image = inputs['color_s']
        depth = inputs['depth']
        semantic = inputs['semantic']
        depth_mask = (depth <= 10) & (depth > 0)
        masked_image = image * depth_mask

        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.imshow(masked_image.permute(1, 2, 0).numpy())
        ax = fig.add_subplot(312)
        ax.imshow(depth.permute(1, 2, 0).numpy())
        ax = fig.add_subplot(313)
        ax.imshow(semantic.permute(1, 2, 0).numpy())

        plt.savefig('data.png', dpi=300)
        plt.close()
        print(i)
