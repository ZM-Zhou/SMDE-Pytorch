import csv
from math import exp
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def metric_depth(pred, gt, median_scale=False, kitti_mask=False,
                 cityscapes_mask=False, min_depth=0, max_depth=80, in_mask=None):
    _, _, h, w = gt.shape
    pred = torch.nn.functional.interpolate(pred, [h, w],
                                           mode='bilinear',
                                           align_corners=False)

    if kitti_mask:
        # garg crop for kitti
        mask = gt > min_depth
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :,
                int(0.40810811 * h):int(0.99189189 * h),
                int(0.03594771 * w):int(0.96405229 * w)] = 1
        mask = mask * crop_mask
    # mask A in ManyDepth
    elif cityscapes_mask:
        mask = (gt > min_depth) & (gt < max_depth)
        if gt.shape[3] == 2048:
            height_crop = gt.shape[2] - 512
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, height_crop:, 192:1856] = 1
            mask = mask * crop_mask
        elif gt.shape[2] == 512 and gt.shape[3] == 1664:
            pass 
        else:
            raise NotImplementedError
    else:
        mask = (gt > min_depth) & (gt < max_depth)
    
    if in_mask is not None:
        in_mask = torch.nn.functional.interpolate(in_mask, [h, w],
                                           mode='nearest')
        mask = (mask * in_mask).to(torch.bool)

    gt = gt[mask].clamp(min_depth, max_depth)
    pred = pred[mask]
    gt_median = torch.median(gt)
    pred_median = torch.median(pred)
    scale = gt_median / pred_median
    if median_scale:
        pred *= scale
    pred = pred.clamp(min_depth, max_depth)

    # compute errors
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()

    rmse = (gt - pred)**2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred))**2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred)**2 / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, scale]

def metric_disp(pred, gt):
    _, _, h, w = gt.shape
    disp_scale = w / pred.shape[3]
    pred = torch.nn.functional.interpolate(pred, [h, w],
                                           mode='bilinear',
                                           align_corners=False)

    mask = gt > 0
    
    gt = gt[mask]
    pred = pred[mask] * disp_scale
    abs_err = torch.abs(gt - pred)
    rel_err = abs_err / gt
    d1 = ((abs_err > 3) & (rel_err > 0.05)).float().mean()
    epe = torch.mean(abs_err)

    return [epe, d1 * 100]

def metric_synth(pred, gt):
    # PSNR
    m_rgb = torch.ones_like(pred)
    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
    pred = (pred + m_rgb).clamp(0, 1)
    gt = (gt + m_rgb).clamp(0, 1)

    mse_err = (pred - gt).pow(2).mean()
    psnr = 10 * (1 / mse_err).log10()

    # SSIM
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            exp(-((x - window_size // 2)**2) / float(2 * sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(
            _2D_window.expand(channel, 1, window_size,
                              window_size).contiguous())
        return window

    def _ssim(img1,
              img2,
              window,
              window_size,
              channel,
              mask=None,
              size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (F.conv2d(
            img1 * img1, window, padding=window_size // 2, groups=channel) -
                     mu1_sq)
        sigma2_sq = (F.conv2d(
            img2 * img2, window, padding=window_size // 2, groups=channel) -
                     mu2_sq)
        sigma12 = (F.conv2d(
            img1 * img2, window, padding=window_size // 2, groups=channel) -
                   mu1_mu2)

        C1 = (0.01)**2
        C2 = (0.03)**2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        if not (mask is None):
            b = mask.size(0)
            ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask
            ssim_map = ssim_map.view(b, -1).sum(dim=1) / mask.view(
                b, -1).sum(dim=1).clamp(min=1)
            return ssim_map

        import pdb

        pdb.set_trace

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    (_, channel, _, _) = pred.size()
    window = create_window(11, channel)

    if pred.is_cuda:
        window = window.cuda(pred.get_device())
    window = window.type_as(pred)

    ssim = _ssim(pred, gt, window, 11, channel, None, True)

    return [psnr, ssim]


def metric_depth_m3d(pred, gt):
    _, _, h, w = gt.shape
    pred = torch.nn.functional.interpolate(pred,
                                           [h, w],
                                           mode='nearest')

    # C1 metric
    mask = (gt > 0) & (gt < 70)
    gt = gt[mask]
    pred = pred[mask]
    gt_median = torch.median(gt)
    pred_median = torch.median(pred)
    scale = gt_median / pred_median
    pred *= scale
    pred = pred.clamp(1e-3, 70)

    # compute errors

    rmse = (gt - pred)**2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred))**2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred)**2 / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, scale]

def metric_depth_nyu(pred, gt, median_scale=False):
    _, _, h, w = gt.shape
    pred = torch.nn.functional.interpolate(pred, [h, w],
                                           mode='bilinear',
                                           align_corners=False)

    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]
    if median_scale:
        gt_median = torch.median(gt)
        pred_median = torch.median(pred)
        scale = gt_median / pred_median
        pred *= scale
    pred = pred.clamp(1e-1, 10)

    # compute errors
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()

    rmse = (gt - pred)**2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred))**2
    rmse_log = torch.sqrt(rmse_log.mean())

    log10 = torch.mean(torch.abs((torch.log10(gt) - torch.log10(pred))))

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred)**2 / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3]



def metric_photo_rmse(pred, gt):
    # RMSE
    m_rgb = torch.ones_like(pred)
    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
    pred = (pred + m_rgb).clamp(0, 1) * 255
    gt = (gt + m_rgb).clamp(0, 1) * 255
    rmse = (torch.mean((pred - gt)**2))**(1 / 2)
    return [rmse]


class Metric(object):
    M_DIRECT = {
        'abs_rel': 1,
        'sq_rel': 1,
        'rms': 1,
        'log_rms': 1,
        'log_10': 1,
        'a1': -1,
        'a2': -1,
        'a3': -1,
        'psnr': -1,
        'ssim': -1,
        'photo_rmse': 1
    }

    def __init__(self, metric_name, best_compute):
        self.metric_name = metric_name
        self.case_names = []
        self.case_num = 0
        self.now_metric = []
        self.computer = []
        if ('depth_kitti' in metric_name
                or 'depth_kitti_mono' in metric_name
                or 'depth_ddad' in metric_name
                or 'depth_ddad_mono' in metric_name
                or 'depth_cityscapes_mono' in metric_name
                or 'depth_cityscapes' in metric_name
                or 'depth_vkitti2' in metric_name):
            self.case_names += [
                'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3', 'scale'
            ]
            self.case_num += 8
        if 'synth' in metric_name:
            self.case_names += ['psnr', 'ssim']
            self.case_num += 2
        if 'photo_rmse' in metric_name:
            self.case_names += ['photo_rmse']
            self.case_num += 1
        if ( 'depth_m3d' in metric_name
                or 'depth_m3d_mono' in metric_name):
            self.case_names += [
                'abs_rel', 'sq_rel', 'rms', 'log_10', 'scale'
            ]
            self.case_num += 5
        if 'depth_nyu_mono' in metric_name:
            self.case_names += [
                'abs_rel', 'sq_rel', 'rms', 'log_rms', 'log_10', 'a1', 'a2', 'a3'
            ]
            self.case_num += 8
        if 'depth_kitti_stereo2015' in metric_name:
            self.case_names += [
                'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3', 'scale', 'EPE-all', 'D1-all'
            ]
            self.case_num += 10

        if best_compute == 'depth_kitti':
            self.best_names = [
                'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'
            ]
        if best_compute == 'depth_a1':
            self.best_names = ['a1']
        elif best_compute == 'synth':
            self.best_names = ['psnr', 'ssim']
        elif best_compute == 'photo_rmse':
            self.best_names = ['photo_rmse']

        self.now_metric = [0 for _ in range(self.case_num)]
        self.all_metric = []
        self.now_count = 0
        self.best_metric = None

    def get_metric_output(self, test_mode=False, save_csv=None):
        mean_metric = self._get_mean_metric()
        for name in self.metric_name:
            if 'mono' in name:
                std_scale = np.std([m[-1] for m in self.all_metric])
                print('    STD of scale: {:0.3f}'.format(std_scale))
        info_line = '    |'
        metric_line = '    |'
        for c_idx in range(self.case_num):
            info_line += self.case_names[c_idx].center(10) + '|'
            if test_mode:
                metric_line += '{:10.3f}|'.format(mean_metric[c_idx])
            else:
                metric_line += '{:10.4f}|'.format(mean_metric[c_idx])

        if save_csv:
            with open(save_csv, 'w') as f:
                writer = csv.writer(f)
                for data_row in self.all_metric:
                    writer.writerow(data_row)

        return [info_line, metric_line]

    def update_metric(self, outputs, inputs, name=None):
        res = []
        if 'depth_kitti' in self.metric_name:
            if name is None:
                pred = outputs[('depth', 's')]
            else:
                pred = outputs[name]
            gt = inputs['depth']
            res += metric_depth(pred, gt, kitti_mask=True)
        if 'depth_kitti_mono' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth(pred, gt, median_scale=True, kitti_mask=True)
        if 'depth_kitti_stereo2015' in self.metric_name:
            if name is None:
                pred = outputs[('depth', 's')]
            else:
                pred = outputs[name]
            gt = inputs['depth']
            res += metric_depth(pred, gt)
            
            if name is None:
                pred = outputs[('disp', 's')]
            else:
                pred = outputs[name.replace('depth', 'disp')]
            gt = inputs['disp']
            res += metric_disp(pred, gt)
        if 'depth_cityscapes' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth(pred, gt, cityscapes_mask=True)
        if 'depth_cityscapes_mono' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth(pred, gt, cityscapes_mask=True,
                                median_scale=True)
        if 'depth_ddad' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth(pred, gt, min_depth=0, max_depth=200)
        if 'depth_ddad_mono' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth(pred, gt, median_scale=True, 
                                min_depth=0, max_depth=200)
        if 'depth_vkitti2' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth(pred, gt, median_scale=False, 
                                min_depth=0, max_depth=100)
        
        if 'synth' in self.metric_name:
            pred = outputs[('synth', 's')]
            gt = inputs['color_o']
            res += metric_synth(pred, gt)
        if 'photo_rmse' in self.metric_name:
            pred = outputs[('synth', 's')]
            gt = inputs['color_o']
            res += metric_photo_rmse(pred, gt)
        if ( 'depth_m3d' in self.metric_name
                or 'depth_m3d_mono' in self.metric_name):
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth_m3d(pred, gt)
        if 'depth_nyu_mono' in self.metric_name:
            pred = outputs[('depth', 's')]
            gt = inputs['depth']
            res += metric_depth_nyu(pred, gt, True)
        

        self.now_metric = [a + b for (a, b) in zip(self.now_metric, res)]
        self.all_metric.append([e.item() for e in res])
        self.now_count += 1

    def clear_metric(self):
        self.now_metric = [0 for _ in range(self.case_num)]
        self.now_count = 0
        self.all_metric = []

    def compute_best_metric(self):
        mean_metric = self._get_mean_metric()
        if self.best_metric is None:
            self.best_metric = mean_metric
            return True
        else:
            improve_metric = [
                (now - best) / best
                for now, best in zip(mean_metric, self.best_metric)
            ]
            all_improve = 0
            for case_name in self.best_names:
                improve_case = improve_metric[self.case_names.index(case_name)]
                direct_case = self.M_DIRECT[case_name]
                all_improve += improve_case * direct_case

            if all_improve < 0:
                self.best_metric = mean_metric
                return True
            else:
                return False

    def _get_mean_metric(self):
        mean_metric = [m / self.now_count for m in self.now_metric]
        return mean_metric
