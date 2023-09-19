import os

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F


def make_output_img(imgs, size):
    """Combine all images into the specified size in order."""
    output_img = []
    for columns in size:
        output_row = []
        for key in columns:
            if isinstance(key, list):
                if key[0] in imgs and key[1] in imgs:
                    base_img = imgs[key[0]]
                    add_img = imgs[key[1]]
                    img = base_img + key[2] * add_img
                    max_value = np.max(img)
                    img = img / max_value * 255
                    output_row.append(img)
                else:
                    output_row.append(
                        np.ones_like(imgs[list(imgs.keys())[0]]) * 123)
            else:
                if key in imgs:
                    output_row.append(imgs[key])
                else:
                    output_row.append(
                        np.ones_like(imgs[list(imgs.keys())[0]]) * 123)
        output_row = np.hstack(output_row)
        output_img.append(output_row)
    output_img = np.vstack(output_img).astype(np.uint8)
    output_img = Image.fromarray(output_img)

    return output_img


class Visualizer(object):
    def __init__(self, output_path, options, rank_id=0):
        self.output_path = output_path
        self.load_dict = options['type']
        self.show_shape = options['shape']
        self.visual_mode_dict = {
            'img': self._visual_rgb,
            'depth': self._visual_depth,
            'disp': self._visual_disp,
            'mask_disp': self._visual_mdisp,
            'error_heat': self._visual_heatjet,
            'error_pn': self._visual_pn,
            'mask': self._visual_mask,
            'mask_raw': self._visual_mask_raw,
            'mask_error_pn': self._visual_mpn,
            'vector': self._visual_vector,
            'normal': self._visual_normal
        }
        self.inter_mode_dict = {
            'img': 'bilinear',
            'depth': 'bilinear',
            'disp': 'bilinear',
            'mask_disp': 'nearest',
            'error_heat': 'bilinear',
            'error_pn': 'bilinear',
            'mask': 'nearest',
            'mask_raw': 'nearest',
            'mask_error_pn': 'nearest',
            'vector': 'nearest',
            'normal': 'nearest'
        }

        self.visual_dict = {}
        self.h = None
        self.w = None
        self.rank_id = rank_id

        os.makedirs(output_path, exist_ok=True)

    def _parallel_mask(func):
        def inner(self, *args, **kwargs):
            if self.rank_id == 0:
                ret = func(self, *args, **kwargs)
                return ret
            else:
                pass

        return inner

    @_parallel_mask
    def update_visual_dict(self, inputs, outputs, losses=None):
        self.visual_dict.clear()
        self.h = None
        self.w = None

        for name, data_type in self.load_dict.items():
            if name in inputs:
                self.visual_dict[name] = (inputs[name], data_type)
                continue
            if name in outputs:
                self.visual_dict[name] = (outputs[name], data_type)
                continue
            if losses is not None and name in losses:
                self.visual_dict[name] = (losses[name], data_type)
                continue
            if name.replace('_s', '_o') in outputs:
                self.visual_dict[name] = (outputs[name.replace('_s', '_o')],
                                          data_type)
                continue
            # if losses is not None and name.replace('_s', '_o') in losses:
            #     self.visual_dict[name] = (losses[name.replace('_s', '_o')],
            #                               data_type)
            #     continue
            if losses is not None and name.replace('/s', '/o') in losses:
                self.visual_dict[name] = (losses[name.replace('/s', '/o')],
                                          data_type)
                continue

    @_parallel_mask
    def do_visualizion(self, name='', t_shape_name=None):
        save_path = os.path.join(self.output_path, name + '.png')
        # determine the output size
        if self.h is None:
            if t_shape_name is None:
                tar_img_key = list(self.visual_dict.keys())[0]
            else:
                tar_img_key = t_shape_name
            tar_img = self.visual_dict[tar_img_key][0]
            _, _, self.h, self.w = tar_img.shape

        for k, v in self.visual_dict.items():
            if isinstance(v, tuple):
                img = v[0].to(torch.float)
                mode = v[1]
                img = F.interpolate(img, [self.h, self.w],
                                    mode=self.inter_mode_dict[mode])
                img = img[0, ...].detach().cpu().permute(1, 2, 0).numpy()
                self.visual_dict[k] = self._do_visualize(img, mode)
        imgs = make_output_img(self.visual_dict, self.show_shape)
        imgs.save(save_path)

    def _do_visualize(self, img, mode):
        output = self.visual_mode_dict[mode](img)
        return output

    def _visual_rgb(self, img):
        if img.min() < 0:
            img += (0.411, 0.432, 0.45)
        return img * 255


    def _visual_depth(self, depth):
        compute_mask = depth != 0
        compute_depth = depth[compute_mask]
        vmax = np.percentile(compute_depth, 95)
        normal_depth = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper_depth = cm.ScalarMappable(norm=normal_depth,
                                         cmap='plasma')  # magma
        depth_color = (mapper_depth.to_rgba(depth[..., 0])[:, :, :3] * 255)
        return depth_color

    def _visual_disp(self, disp):
        compute_mask = disp != 0
        compute_disp = disp[compute_mask]
        normal_disp = mpl.colors.Normalize(vmin=compute_disp.min(),
                                           vmax=disp.max())
        mapper_disp = cm.ScalarMappable(norm=normal_disp,
                                        cmap='plasma')  # magma
        disp_color = (mapper_disp.to_rgba(disp[..., 0])[:, :, :3] * 255)
        return disp_color

    def _visual_heatjet(self, error):
        max_value = np.max(error)
        normal_error = mpl.colors.Normalize(vmin=error.min(), vmax=max_value)
        mapper_error = cm.ScalarMappable(norm=normal_error, cmap='jet')
        error = (mapper_error.to_rgba(error[..., 0])[:, :, :3] * 255)
        return error

    def _visual_pn(self, error):
        min_value = np.abs(np.min(error))
        max_value = np.abs(np.max(error))
        if max_value > min_value:
            normal_value = max_value
        else:
            normal_value = min_value
        normal_error = mpl.colors.Normalize(vmin=-normal_value,
                                            vmax=normal_value)
        mapper_error = cm.ScalarMappable(norm=normal_error, cmap='coolwarm')
        error = (mapper_error.to_rgba(error[..., 0])[:, :, :3] * 255)
        return error

    def _visual_mpn(self, error):
        mask = error > 1e9
        error[mask] = 0
        min_value = np.abs(np.min(error))
        max_value = np.abs(np.max(error))
        if max_value > min_value:
            normal_value = max_value
        else:
            normal_value = min_value
        normal_error = mpl.colors.Normalize(vmin=-normal_value,
                                            vmax=normal_value)
        mapper_error = cm.ScalarMappable(norm=normal_error, cmap='coolwarm')
        error = (mapper_error.to_rgba(error[..., 0])[:, :, :3] * 255) * (1 -
                                                                         mask)
        return error

    def _visual_mdisp(self, disp):
        compute_mask = disp != 0
        compute_disp = disp[compute_mask]
        normal_disp = mpl.colors.Normalize(vmin=compute_disp.min(),
                                           vmax=disp.max())
        mapper_disp = cm.ScalarMappable(norm=normal_disp,
                                        cmap='plasma')  # magma
        disp_color = (mapper_disp.to_rgba(disp[..., 0])[:, :, :3] *
                      255) * compute_mask
        return disp_color

    def _visual_mask(self, mask):
        max_element = mask.max()
        show_mask = (mask / max_element) * 255
        show_mask = np.tile(show_mask, (1, 1, 3))
        return show_mask
    
    def _visual_mask_raw(self, mask):
        show_mask = np.tile(mask, (1, 1, 3))
        return show_mask

    def _visual_vector(self, vector):
        mod = (vector[..., 0]**2 + vector[..., 1]**2)**0.5
        max_mod = mod.max()
        normal_mod = mod[:, :, np.newaxis] / max_mod
        phase = np.arctan2(vector[..., 0], vector[..., 1])
        phase = phase[:, :, np.newaxis]
        normal_phase = mpl.colors.Normalize(vmin=-3.14159, vmax=3.14159)
        mapper_phase = cm.ScalarMappable(norm=normal_phase, cmap='hsv')
        vector = (mapper_phase.to_rgba(phase[..., 0])[:, :, :3] * 255)
        vector = vector * normal_mod + 255 * (1 - normal_mod)
        return vector
    
    def _visual_normal(self, normal):
        show_normal = (1 + normal) / 2 * 255
        return show_normal
