import os
import random
from collections import namedtuple

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
class KITTIColorDepthDataset(data.Dataset):
    """KITTI dataset for depth estimation from images."""

    DATA_NAME_DICT = {
        'color_l': ('image_02', 'png'),
        'color_r': ('image_03', 'png'),
        'depth': ('velodyne_points', 'bin'),
        'hints_l': ('image_02', 'npy'),
        'hints_r': ('image_03', 'npy'),
        'seg_l': ('image_02', 'png'),
        'seg_r': ('image_03', 'png'),
    }

    def __init__(
            self,
            dataset_mode,
            split_file,
            full_size=None,
            patch_size=None,
            random_resize=True,
            normalize_params=[0.411, 0.432, 0.45],
            flip_mode=None,
            color_aug=True,
            output_frame=['o'],  # -1, 1
            multi_out_scale=None,
            load_KTmatrix=False,
            load_depth=True,
            load_depthhints=False,
            load_semantic=False,
            is_fixK=False,
            stereo_test=False,
            jpg_test=False,
            improved_test=False,):
        super().__init__()
        self.init_opts = locals()

        self.dataset_dir = Path.get_path_of('kitti')
        self.dataset_mode = dataset_mode
        self.split_file = split_file
        self.full_size = full_size
        self.patch_size = patch_size
        self.random_resize = random_resize
        self.flip_mode = flip_mode
        self.color_aug = color_aug
        self.output_frame = output_frame
        self.multi_out_scale = multi_out_scale
        self.load_KTmatrix = load_KTmatrix
        self.load_depth = load_depth
        self.load_depthhints = load_depthhints
        self.load_semantic = load_semantic
        self.is_fixK = is_fixK
        self.improved_test = improved_test
        self.stereo_test = stereo_test
        self.jpg_test = jpg_test
        
        self.file_list = self._get_file_list(split_file)

        if self.load_semantic:
            assert os.path.exists(os.path.join(self.dataset_dir, 'segmentation'))

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

        # Change the root path if use jpg images
        if self.jpg_test:
            if dataset_mode == 'test':
                self.dataset_dir = Path.get_path_of('eigen_kitti_test_jpg')
            else:
                raise NotImplementedError
       
        # Read pre-generated ground-truth depths for testing:
        self.gt_depths = None
        if len(self.file_list) == 697 or len(self.file_list) == 652:
            if self.improved_test:
                gt_path = os.path.join(self.dataset_dir, 'gt_depths_improved.npz')
                self.gt_depths = np.load(gt_path,
                                         fix_imports=True,
                                         encoding='latin1',
                                         allow_pickle=True)['data']
            else:
                gt_path = os.path.join(self.dataset_dir, 'gt_depths_raw.npz')
                self.gt_depths = np.load(gt_path,
                                         fix_imports=True,
                                         encoding='latin1',
                                         allow_pickle=True)['data']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):
        """Return a data dictionary."""
        file_info = self.file_list[f_idx]
        base_path_list, data_side, date_path = self._get_formative_file_base(
            file_info)

        # Read data
        inputs = {}
        target_path = base_path_list[0]
        color_side_path = target_path.format(
            *self.DATA_NAME_DICT['color_{}'.format(data_side)])
        if self.jpg_test:
            color_side_path = color_side_path.replace('.png', '.jpg')
        inputs['color_s_raw'] = get_input_img(color_side_path)

        img_W, img_H = inputs['color_s_raw'].size
        if self.is_fixK:
            intrinsic = np.array(K_of_KITTI)
        else:
            intrinsic = get_intrinsic(date_path)

        if self.is_fixK == 'v2':
            k = intrinsic[0, 0] * 0.54
            inputs['disp_k'] = torch.tensor(k, dtype=torch.float)
            if self.full_size is not None:
                inputs['disp_k'] *= self.full_size[1] / 1244
        else:
            k = intrinsic[0, 0] * width_to_baseline[
                img_W]  # approximate coefficient
            inputs['disp_k'] = torch.tensor(k, dtype=torch.float)
            if self.full_size is not None:
                inputs['disp_k'] *= self.full_size[1] / img_W

        inputs['direct'] = torch.tensor(
            1,
            dtype=torch.float)  # used to determine the direction of baseline
        if data_side == 'r':
            inputs['direct'] = -inputs['direct']

        # take ground-truth depths
        if self.gt_depths is not None:
            inputs['depth'] = self.gt_depths[f_idx]
        else:
            if self.load_depth:
                depth_path = target_path.format(*self.DATA_NAME_DICT['depth'])
                inputs['depth'] = get_input_depth(depth_path, date_path, data_side)

        if self.stereo_test:
            color_path = base_path_list[0].format(*self.DATA_NAME_DICT[
                'color_{}'.format('r' if data_side == 'l' else 'l')])
            if self.jpg_test:
                color_path = color_path.replace('.png', '.jpg')
            inputs['color_o_raw'] = get_input_img(color_path)

        if self.load_KTmatrix:
            intric = np.zeros((4, 4))
            intric[:3, :3] = intrinsic[:, :3]
            intric[3, 3] = 1
            if self.full_size is not None:
                intric[0, :] *= self.full_size[1] / img_W
                intric[1, :] *= self.full_size[0] / img_H
            if self.is_fixK:
                if data_side == 'l':
                    baseline = -0.54
                else:
                    baseline = 0.54
                extric = torch.tensor([[1, 0, 0, baseline], [0, 1, 0, 0],
                                        [0, 0, 1, 0], [0, 0, 0, 1]])
            else:
                if data_side == 'l':
                    factor = -1
                else:
                    factor = 1
                extric = torch.tensor(
                    [[1, 0, 0, factor * width_to_baseline[img_W]],
                        [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            inputs['K'] = torch.from_numpy(intric).to(torch.float)
            inputs['inv_K'] = torch.from_numpy(np.linalg.pinv(intric))\
                .to(torch.float)
            inputs['T'] = extric

        if self.load_semantic:
            base_path = base_path_list[0]
            seg_l_path = base_path.format(
                *self.DATA_NAME_DICT['seg_{}'.format(data_side)])
            seg_l_path = seg_l_path.replace(
                self.dataset_dir,
                self.dataset_dir + '/segmentation').replace('data/', '')
            seg_l = get_input_img(seg_l_path, False)
            seg_copy_l = np.array(seg_l.copy())
            for k in np.unique(seg_l):
                seg_copy_l[seg_copy_l == k] = labels[k].trainId
            inputs['seg_s'] = torch.from_numpy(seg_copy_l).unsqueeze(0)

            if 'o' in self.output_frame:
                oside = 'r' if data_side == 'l' else 'l'
                seg_r_path = base_path.format(
                    *self.DATA_NAME_DICT['seg_{}'.format(oside)])
                seg_r_path = seg_r_path.replace(
                    self.dataset_dir,
                    self.dataset_dir + '/segmentation').replace('data/', '')
                seg_r = get_input_img(seg_r_path, False)
                seg_copy_r = np.array(seg_r.copy())
                for k in np.unique(seg_r):
                    seg_copy_r[seg_copy_r == k] = labels[k].trainId
                inputs['seg_o'] = torch.from_numpy(seg_copy_r).unsqueeze(0)

        if self.dataset_mode == 'train':
            for idx_frame, base_path in enumerate(base_path_list[1:]):
                if self.output_frame[idx_frame] != 'o':
                    color_path = base_path.format(
                        *self.DATA_NAME_DICT['color_{}'.format(data_side)])
                else:
                    color_path = base_path_list[0].format(*self.DATA_NAME_DICT[
                        'color_{}'.format('r' if data_side == 'l' else 'l')])
                if self.jpg_test:
                    color_path = color_path.replace('.png', '.jpg')
                inputs['color_{}_raw'.format(self.output_frame[idx_frame])]\
                    = get_input_img(color_path)

            if self.load_depthhints:
                hints_l_path = base_path.format(
                    *self.DATA_NAME_DICT['hints_{}'.format(data_side)])
                hints_l_path = hints_l_path.replace(
                    self.dataset_dir,
                    self.dataset_dir + '/depth_hints').replace('data/', '')
                hints_depth_l = torch.from_numpy(np.load(hints_l_path))
                # 0.058 = 0.58 (K[0, 0]) * 0.1 (baseline)
                inputs['hints_s'] = 5.4 * hints_depth_l
                # inputs['hints_s'] = 0.058 / (hints_depth_l +
                #                              1e-8) * (hints_depth_l > 0)
                if 'o' in self.output_frame:
                    oside = 'r' if data_side == 'l' else 'l'
                    hints_r_path = base_path.format(
                        *self.DATA_NAME_DICT['hints_{}'.format(oside)])
                    hints_r_path = hints_r_path.replace(
                        self.dataset_dir,
                        self.dataset_dir + '/depth_hints').replace('data/', '')
                    hints_depth_r = torch.from_numpy(np.load(hints_r_path))
                    inputs['hints_o'] = 5.4 * hints_depth_r
                    # inputs['hints_o'] = 0.058 / (hints_depth_r +
                    #                              1e-8) * (hints_depth_r > 0)
            
            if (self.dataset_mode == 'train' and self.flip_mode is not None):
                if self.flip_mode == 'both':  # random flip mode
                    switch_img = random.uniform(0, 1) > 0.5
                    switch_k = random.uniform(0, 1) > 0.5
                    if switch_img and switch_k:
                        is_flip = False
                        inputs['color_o_raw'], inputs['color_s_raw'] =\
                            inputs['color_s_raw'], inputs['color_o_raw']
                        if self.load_depthhints:
                            inputs['hints_s'], inputs['hints_o'] = \
                                inputs['hints_o'], inputs['hints_s']
                        if self.load_semantic:
                            inputs['seg_s'], inputs['seg_o'] = \
                                inputs['seg_o'], inputs['seg_s']    
                        inputs['direct'] = -inputs['direct']
                        if self.load_KTmatrix:
                            inputs['T'][0, 3] = -inputs['T'][0, 3]
                    elif switch_img and not switch_k:
                        is_flip = True
                        inputs['color_o_raw'], inputs['color_s_raw'] =\
                            inputs['color_s_raw'], inputs['color_o_raw']
                        if self.load_depthhints:
                            inputs['hints_s'], inputs['hints_o'] = \
                                inputs['hints_o'], inputs['hints_s']
                        if self.load_semantic:
                            inputs['seg_s'], inputs['seg_o'] = \
                                inputs['seg_o'], inputs['seg_s']    
                    elif switch_img and not switch_k:
                        is_flip = True
                        inputs['direct'] = -inputs['direct']
                        if self.load_KTmatrix:
                            inputs['T'][0, 3] = -inputs['T'][0, 3]
                    else:
                        is_flip = False
                else:
                    is_flip = random.uniform(0, 1) > 0.5
                    if is_flip:
                        flip_img = self.flip_mode == 'img'
                        if flip_img:
                            inputs['color_o_raw'], inputs['color_s_raw'] =\
                                inputs['color_s_raw'], inputs['color_o_raw']
                            if self.load_depthhints:
                                inputs['hints_s'], inputs['hints_o'] = \
                                    inputs['hints_o'], inputs['hints_s']
                            if self.load_semantic:
                                inputs['seg_s'], inputs['seg_o'] = \
                                    inputs['seg_o'], inputs['seg_s']    
                        else:
                            inputs['direct'] = -inputs['direct']
                            if self.load_KTmatrix:
                                inputs['T'][0, 3] = -inputs['T'][0, 3]

        # Process data
        # resize crop & color jit & flip for train
        if self.dataset_mode == 'train':
            # resize
            if self.full_size is not None:
                img_size = self.full_size
            else:
                _size = inputs['color_s_raw'].size  # (w, h)
                img_size = (_size[1], _size[0])
            scale_factor = random.uniform(0.75, 1.5)\
                if self.patch_size is not None and self.random_resize else 1
            if scale_factor != 1 or self.full_size is not None:
                random_size = tuple(int(s * scale_factor) for s in img_size)
                self.color_resize = tf.Resize(random_size,
                                              interpolation=Image.ANTIALIAS)
                if self.multi_out_scale is not None:
                    self.multi_resize = {}
                    if self.patch_size is not None:
                        base_size = self.patch_size
                    else:
                        base_size = img_size
                    for scale in self.multi_out_scale:
                        s = 2 ** scale
                        self.multi_resize[scale] = tf.Resize([x // s for x in base_size],
                                                         interpolation=Image.ANTIALIAS)
                
                self.depth_resize = tf.Resize(random_size,
                                              interpolation=Image.NEAREST)
            else:
                self.color_resize = NoneTransform()
                self.depth_resize = NoneTransform()
            # crop
            crop_params = self.crop.get_params(img_size, self.patch_size,
                                               scale_factor)
            # color jit
            if self.color_aug and random.uniform(0, 1) < 0.5:
                if self.color_aug == 'v2':
                    do_color_aug = tf.ColorJitter.get_params(
                        (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
                else:
                    gamma_param = random.uniform(0.8, 1.2)
                    bright_param = random.uniform(0.5, 2)
                    cbright_param = [random.uniform(0.8, 1.2) for _ in range(3)]
            else:
                if self.color_aug == 'v2':
                    do_color_aug = lambda x: x
                else:
                    gamma_param = 1
                    bright_param = 1
                    cbright_param = [1, 1, 1]

            for key in list(inputs):
                if 'color' in key:
                    raw_img = inputs[key]
                    if is_flip:
                        raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                    raw_img = self.crop(self.color_resize(raw_img), crop_params)
                    img = self.to_tensor(raw_img)
                    if self.color_aug == 'v2':
                        aug_img = do_color_aug(img)
                    else:
                        aug_img = do_fal_color_aug(img, gamma_param,
                                                        bright_param,
                                                        cbright_param)
                    inputs[key.replace('_raw', '')] =\
                        self.normalize(img)
                    inputs[key.replace('_raw', '_aug')] =\
                        self.normalize(aug_img)
                    if self.multi_out_scale is not None:
                        for scale in self.multi_out_scale:
                            scale_img = self.multi_resize[scale](raw_img)
                            scale_img = self.to_tensor(scale_img)
                            if self.color_aug == 'v2':
                                scale_aug_img = do_color_aug(scale_img)
                            else:
                                scale_aug_img = do_fal_color_aug(scale_img,
                                                                 gamma_param,
                                                                 bright_param,
                                                                 cbright_param)
                            inputs[key.replace('_raw', '_{}'.format(scale))] =\
                                self.normalize(scale_img)
                            inputs[key.replace('_raw', '_{}_aug'.format(scale))] =\
                                self.normalize(scale_aug_img)

                elif 'depth' in key:
                    # depth will be changed when resize
                    raw_depth = inputs[key] / scale_factor
                    if is_flip:
                        raw_depth = np.fliplr(raw_depth)
                    depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                    depth = self.crop(self.depth_resize(depth), crop_params)
                    inputs[key] = depth

                elif 'hints' in key:
                    raw_hints = inputs[key] / scale_factor
                    # raw_hints = inputs[key] * random_size[1]
                    if is_flip:
                        raw_hints = torch.flip(raw_hints, dims=[2])
                    hints = self.crop(self.depth_resize(raw_hints),
                                      crop_params)
                    inputs[key] = hints.to(torch.float)
                elif 'seg' in key:
                    raw_seg = inputs[key]
                    if is_flip:
                        raw_seg = torch.flip(raw_seg, dims=[2])
                    seg = self.crop(self.depth_resize(raw_seg),
                                      crop_params)
                    inputs[key] = seg

        # resize for other modes
        else:
            for key in list(inputs):
                if 'color' in key:
                    raw_img = inputs[key]
                    img = self.color_resize(raw_img)
                    inputs[key.replace('_raw', '')] =\
                        self.normalize(self.to_tensor(img))

                elif 'depth' in key:
                    # do not resize ground truth in test
                    raw_depth = inputs[key]
                    depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)  
                    inputs[key] = depth
                elif 'hints' in key:
                    # do not resize hints truth in test
                    raw_hints = inputs[key]
                    inputs[key] = raw_hints.to(torch.float)

        # delete the raw data
        inputs.pop('color_s_raw')
        if self.stereo_test:
            inputs.pop('color_o_raw')
        if self.dataset_mode == 'train':
            for id_frame in self.output_frame:
                inputs.pop('color_{}_raw'.format(id_frame))

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

    def _get_formative_file_base(self, info):
        info_part = info.split(' ')
        data_dir = info_part[0]
        data_idx = info_part[1]
        img_side = info_part[2]

        date_path = os.path.join(self.dataset_dir, data_dir.split('/')[0])
        base_path_list = []
        base_path = os.path.join(self.dataset_dir, data_dir, '{}', 'data',
                                 '{:010d}'.format(int(data_idx)) + '.{}')
        base_path_list.append(base_path)
        for frame_id in self.output_frame:
            if frame_id != 'o':
                base_path = os.path.join(
                    self.dataset_dir, data_dir, '{}', 'data',
                    '{:010d}'.format(int(data_idx) + frame_id) + '.{}')
                base_path_list.append(base_path)
            else:
                base_path = os.path.join(
                    self.dataset_dir, data_dir, '{}', 'data',
                    '{:010d}'.format(int(data_idx)) + '.{}')
                base_path_list.append(base_path)

        return base_path_list, img_side, date_path

    @property
    def dataset_info(self):
        infos = []
        infos.append('    -{} Datasets'.format(self.dataset_mode))
        infos.append('      get {} of data'.format(len(self)))
        for key, val in self.init_opts.items():
            if key not in ['self', '__class__']:
                infos.append('        {}: {}'.format(key, val))
        return infos
