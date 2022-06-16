import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils import data

from datasets.utils.data_reader import get_input_depth_make3d, get_input_img
from datasets.utils.my_transforms import NoneTransform
from path_my import Path
from utils import platform_manager

# ref https://github.com/nianticlabs/monodepth2/issues/392

@platform_manager.DATASETS.add_module
class Make3DDataset(data.Dataset):
    DATA_NAME_DICT = {
        'color': ('Test134', 'img-', 'jpg'),
        'depth': ('Gridlaserdata', 'depth_sph_corr-', 'mat')
    }

    def __init__(self,
                 dataset_mode,
                 split_file,
                 normalize_params=[0.411, 0.432, 0.45],
                 is_godard_crop=True,
                 full_size=None):
        super().__init__()
        self.init_opts = locals()
        self.dataset_mode = dataset_mode
        self.dataset_dir = Path.get_path_of('make3d')
        self.split_file = split_file
        self.is_godard_crop = is_godard_crop
        self.full_size = full_size

        self.file_list = self._get_file_list(split_file)

        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, f_idx):

        file_info = self.file_list[f_idx]
        base_path = os.path.join(self.dataset_dir, '{}',
                                 '{}' + file_info + '.{}')
        inputs = {}
        color_l_path = base_path.format(*self.DATA_NAME_DICT['color'])
        inputs['color_s_raw'] = get_input_img(color_l_path)

        depth_path = base_path.format(*self.DATA_NAME_DICT['depth'])
        inputs['depth'] = get_input_depth_make3d(depth_path)

        for key in list(inputs):
            if 'color' in key:
                raw_img = inputs[key]
                if self.is_godard_crop:
                    raw_img = raw_img.crop((0, 710, 1704, 1562))
                img = self.to_tensor(raw_img)
                if self.full_size is not None:
                    # for outputting the same image with cv2
                    img = img.unsqueeze(0)
                    img = F.interpolate(img, self.full_size, mode='nearest')
                    img = img.squeeze(0)
                inputs[key.replace('_raw', '')] =\
                    self.normalize(img)

            elif 'depth' in key:
                raw_depth = inputs[key]
                if self.is_godard_crop:
                    raw_depth = raw_depth[17:38, :]
                depth = torch.from_numpy(raw_depth.copy()).unsqueeze(0)
                inputs[key] = depth

        # delete raw data
        inputs.pop('color_s_raw')
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
