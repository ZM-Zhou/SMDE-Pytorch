import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.get_dataset import get_dataset_with_opts
from metric import Metric
from models.get_models import get_model_with_opts
from saver import load_model_for_evaluate
from utils.platform_loader import read_yaml_options
from visualizer import Visualizer

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='SMDE Evaluation Parser')

parser.add_argument('--out_dir',
                    dest='out_dir',
                    default='./eval_res',
                    help='output dir path')
parser.add_argument('--exp_opts',
                    dest='exp_opts',
                    # required=True,
                    help="the yaml file for model's options")
parser.add_argument('--test_opts',
                    dest='test_opts',
                    default=None,
                    help="the yaml file for test set")
parser.add_argument('--model_path',
                    dest='trained_model',
                    # required=True,
                    help='the path of trained model')

parser.add_argument('--num_workers',
                    dest='num_workers',
                    type=int,
                    default=2,
                    help='# of dataloader')
parser.add_argument('--visual_list',
                    dest='visual_list',
                    default=None,
                    help='list of images which should be visualized')
parser.add_argument('--visual_opts',
                    dest='visual_opts',
                    default='options/_base/visualization/test-l-d.yaml',
                    help='the yaml file for visualization options')
parser.add_argument('--save_visual',
                    action='store_true',
                    default=False,
                    help='Save visualization results')
parser.add_argument('--save_pred',
                    action='store_true',
                    default=False,
                    help='Save predicted depths')
parser.add_argument('--extra_name',
                    '--extra_name',
                    default='',
                    help='save results with extera dir name')

parser.add_argument('-fpp',
                    '--simple_flip_post_process',
                    action='store_true',
                    default=False,
                    help='Simple Flip Post-processing')
parser.add_argument('-gpp',
                    '--godard_post_process',
                    action='store_true',
                    default=False,
                    help='Post-processing as done in Godards paper')
parser.add_argument('-mspp',
                    '--multi_scale_post_process',
                    action='store_true',
                    default=False,
                    help='Post-processing as done in FAL-Net')
parser.add_argument('--metric_name',
                    dest='metric_name',
                    type=str,
                    nargs='+',
                    default=['depth_kitti'],
                    help='metric type')
parser.add_argument('--disable_metric',
                    dest='disable_metric',
                    action='store_true',
                    default= False,
                    help='disable metric')

parser.add_argument('--precompute_path',
                    default=None,
                    help='use pre-compute depth for evaluation')

opts = parser.parse_args()

def flip_post_process_disparity(l_disp, r_disp):
    return (l_disp + r_disp) / 2

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in
    Monodepthv1."""
    _, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    l_mask = torch.from_numpy(l_mask.copy()).unsqueeze(0).to(l_disp)
    r_mask = torch.from_numpy(r_mask.copy()).unsqueeze(0).to(l_disp)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def multi_scale_post_process(l_disp, r_down_disp):
    norm = l_disp / (np.percentile(l_disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * l_disp + norm * r_down_disp


def evaluate():
    # Initialize the random seed and device
    device = torch.device('cuda')
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the options
    opts_dic = read_yaml_options(opts.exp_opts)
    if opts.save_visual:
        visual_dic = read_yaml_options(opts.visual_opts)

    # Initialize the dataset and dataloader
    print('->Load the test dataset')
    if opts.test_opts is not None:
        test_dic = read_yaml_options(opts.test_opts)
        opts_dic['test_dataset'] = test_dic['_dataset']
    if 'photo_rmse' in opts.metric_name:
        opts_dic['test_dataset']['params']['stereo_test'] = True
    test_dataset = get_dataset_with_opts(opts_dic, 'test')
    test_loader = DataLoader(test_dataset,
                             1,
                             num_workers=opts.num_workers,
                             shuffle=False,
                             drop_last=True)

    if not opts.precompute_path:
        # Initialize the network
        print('->Load the pretrained model')
        network = get_model_with_opts(opts_dic, device)
        network = load_model_for_evaluate(opts.trained_model, network)
        network.eval()
    else:
        print('->Use pt files in ' + opts.precompute_path)

    if not opts.disable_metric:
        metric_func = Metric(opts.metric_name, None)

    # Initialize the output folder
    if opts.visual_list is not None:
        with open(opts.visual_list, 'r') as f:
            visual_list = f.readlines()
            for list_idx, line in enumerate(visual_list):
                visual_list[list_idx] = line.replace('\n', '')
    if opts.save_visual or opts.save_pred:
        exp_name = opts.trained_model.split('/')[-3]
        out_dir = os.path.join(
            opts.out_dir, exp_name, opts.extra_name)
        if opts.simple_flip_post_process:
            out_dir += '-fpp'
        elif opts.godard_post_process:
            out_dir += '-gpp'
        elif opts.multi_scale_post_process:
            out_dir += '-mspp'
        else:
            out_dir += '-raw'
        if opts.save_visual:
            os.makedirs(out_dir + '/visual', exist_ok=True)
            visualizer = Visualizer(out_dir + '/visual', visual_dic['visual'])
        if opts.save_pred:
            os.makedirs(out_dir + '/pred', exist_ok=True)

    # Evaluate
    if (opts.simple_flip_post_process or opts.godard_post_process
            or opts.multi_scale_post_process):
        print('->Use the post processing')
    print('->Start Evaluation')
    test_data_num = len(test_loader)
    # all_errors = [0 for _ in range(7)]
    idx = 0
    with torch.no_grad():
        for inputs in test_loader:
            for ipt_key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[ipt_key] = ipt.to(device, non_blocking=True)
            if not opts.precompute_path:
                outputs = network.inference_forward(inputs, is_train=False)
                if opts.godard_post_process or opts.simple_flip_post_process:
                    inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
                    flip_outputs = network.inference_forward(inputs, is_train=False)
                    fflip_depth = torch.flip(flip_outputs[('depth', 's')],
                                             dims=[3])
                    if opts.godard_post_process:
                        pp_depth = batch_post_process_disparity(
                            1 / outputs[('depth', 's')], 1 / fflip_depth)
                    else:
                        pp_depth = flip_post_process_disparity(
                            1 / outputs[('depth', 's')], 1 / fflip_depth)
                    pp_depth = 1 / pp_depth
                    inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
                    outputs[('depth', 's')] = pp_depth.clone()
                elif opts.multi_scale_post_process:
                    inputs['color_s'] = torch.flip(inputs['color_s'], dims=[3])
                    up_fac = 2/3
                    H, W = inputs['color_s'].shape[2:]
                    raw_color = inputs['color_s'].clone()
                    inputs['color_s'] = F.interpolate(inputs['color_s'],
                                             scale_factor=up_fac,
                                             mode='bilinear',
                                             align_corners=True)
                    flip_outputs = network.inference_forward(inputs, is_train=False)
                    flip_depth = flip_outputs[('depth', 's')]
                    flip_depth = up_fac * F.interpolate(flip_depth,
                                                           size=(H, W),
                                                           mode='nearest')
                    fflip_depth = torch.flip(flip_depth,
                                             dims=[3])
                    pp_depth = multi_scale_post_process(
                        1 / outputs[('depth', 's')], 1 / fflip_depth)
                    pp_depth = 1 / pp_depth

                    inputs['color_s'] = torch.flip(raw_color, dims=[3])
                    outputs[('depth', 's')] = pp_depth.clone()
                else:
                    pp_depth = outputs[('depth', 's')].clone()
            else:
                outputs = {}
                pt_path = opts.precompute_path + '/{}.pt'.format(idx)
                outputs[('depth',
                         's')] = torch.load(pt_path).to(inputs['depth'])

            if not opts.disable_metric:
                metric_func.update_metric(outputs, inputs)

            if opts.visual_list is not None and inputs['file_info'][0][
                    0] in visual_list:
                if opts.save_visual:
                    visual_map = {}
                    visual_map['pp_disp'] = 1 / pp_depth
                    visual_map['pp_depth'] = pp_depth

                    visualizer.update_visual_dict(inputs, outputs, visual_map)
                    visualizer.do_visualizion(str(idx))
                if opts.save_pred:
                    # save_path = os.path.join(
                    #     out_dir, 'pred', inputs['file_info'][0][0].replace(
                    #         ' ', '__').replace('/', '-') + '.pt')
                    save_path = os.path.join(
                        out_dir, 'pred', str(idx) + '.pt')
                    torch.save(pp_depth, save_path)
            print('{}/{}'.format(idx, test_data_num), end='\r')
            idx += 1
        print('{}/{}'.format(idx, test_data_num))
    
    if not opts.disable_metric:
        info_line, err_line = metric_func.get_metric_output(test_mode=True)
        print(info_line)
        print(err_line)

if __name__ == '__main__':
    evaluate()
