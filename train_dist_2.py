import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import get_dataset_with_opts
from logger import Logger
from metric import Metric
from models import get_model_with_opts
from saver import ModelSaver
from utils.env_information import get_env_info
from utils.platform_loader import read_yaml_options
from visualizer import Visualizer

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='SMDE-Pytorch Training Parser')

parser.add_argument('--local_rank', type=int, help='local gpu id', default=0)

parser.add_argument('--name',
                    dest='exp_name',
                    type=str,
                    required=True,
                    help='the name of experiment')
parser.add_argument('--log_dir',
                    dest='log_dir',
                    type=str,
                    default='./train_log',
                    help='log path')
parser.add_argument('--seed',
                    dest='seed',
                    type=int,
                    default=2021,
                    help='the random seed')

parser.add_argument('--batch_size',
                    dest='batch_size',
                    type=int,
                    default=4,
                    help='# images (pair) in batch')
parser.add_argument('--num_workers',
                    dest='num_workers',
                    type=int,
                    default=4,
                    help='# of dataloader')
parser.add_argument('--epoch',
                    dest='epoch',
                    type=int,
                    default=50,
                    help='# of train epochs')

parser.add_argument('--exp_opts',
                    dest='exp_opts',
                    required=True,
                    help="the yaml file for model's options")
parser.add_argument('--pretrained_path',
                    dest='pretrained_path',
                    default=None,
                    help='the path of pretrained model')
parser.add_argument('--start_epoch',
                    dest='start_epoch',
                    type=int,
                    default=None,
                    help='# of training')

parser.add_argument('--disable_compute_flops',
                    dest='compute_flops',
                    action='store_false',
                    default=True,
                    help='compute FLOPs before training')

parser.add_argument('--log_freq',
                    dest='log_freq',
                    type=int,
                    default=100,
                    help='the frequency of text log')
parser.add_argument('--visual_freq',
                    dest='visual_freq',
                    type=int,
                    default=1000,
                    help='the frequency of visualize')
parser.add_argument('--save_freq',
                    dest='save_freq',
                    type=int,
                    default=10,
                    help='the frequency of save')

parser.add_argument('--test_freq',
                    dest='test_freq',
                    type=int,
                    default=1,
                    help='the frequency of test')

parser.add_argument('--metric_name',
                    dest='metric_name',
                    type=str,
                    nargs='+',
                    default=['depth_kitti'],
                    help='metric type')
parser.add_argument('--metric_source',
                    dest='metric_source',
                    type=str,
                    nargs='+',
                    default=[None],
                    help='metric source')
parser.add_argument('--best_compute',
                    dest='best_compute',
                    type=str,
                    default='depth_kitti',
                    help='metric for selecting best model')



opts = parser.parse_args()

# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------

class Trainer(object):
    def __init__(self, env_info):
        self.world_size = env_info['GPU Number']
        if self.world_size == 1:
            torch.set_num_threads(1)
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        # Initialize the experiment logger
        self.logger = Logger(opts.log_dir, opts.exp_name, opts.local_rank)
        self.logger.log_for_env(env_info)
        self.logger.log_for_opts(opts)

        # Initialize the random seed to all the processes
        # TODO: nessesary?
        seed = opts.seed
        if self.world_size != 1:
            if opts.local_rank == 0:
                random_num = torch.tensor(seed,
                                          dtype=torch.int32,
                                          device=self.device)
            else:
                random_num = torch.tensor(0,
                                          dtype=torch.int32,
                                          device=self.device)
            dist.broadcast(random_num, src=0)
            seed = random_num.item()
        
        self._set_all_seed(seed)
        self.seed = seed

        # Initialize the options
        opts_dic = read_yaml_options(opts.exp_opts)

        # Initialize the datasets and dataloaders
        if 'photo_rmse' in opts.metric_name:
            assert 'stereo_test' in  opts_dic['val_dataset']['params']\
                and opts_dic['val_dataset']['params']['stereo_test'] == True
        train_dataset = get_dataset_with_opts(opts_dic, 'train')
        val_dataset = get_dataset_with_opts(opts_dic, 'val')
        self.val_dataset_len = len(val_dataset)

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(train_dataset)
            self.train_loader = DataLoader(train_dataset,
                                           opts.batch_size,
                                           num_workers=opts.num_workers,
                                           shuffle=False,
                                           pin_memory=True,
                                           drop_last=True,
                                           sampler=self.train_sampler)
        else:
            self.train_loader = DataLoader(train_dataset,
                                           opts.batch_size,
                                           num_workers=opts.num_workers,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)
        # Use separate loaders for validation
        self.val_loader = DataLoader(val_dataset,
                                     1,
                                     num_workers=opts.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
        self.logger.log_for_data(train_dataset.dataset_info,
                                 val_dataset.dataset_info,
                                 len(self.train_loader),
                                 len(self.val_loader),
                                 opts.num_workers)

        # Initialize the saver and the metric
        self.saver = ModelSaver(self.logger.get_log_dir,
                                is_parallel=(self.world_size > 1),
                                rank_id=opts.local_rank)
        self.metric = []
        for sou_name in opts.metric_source:
            metric_item = Metric(opts.metric_name, opts.best_compute)
            self.metric.append((sou_name, metric_item))

        # Initialize the network
        self.network = get_model_with_opts(opts_dic, self.device)
        net_info, loss_info = self.network.network_info
        self.logger.log_for_model(opts_dic['model']['type'], 
                                  net_info,
                                  loss_info)

        # Load the pretrained model
        # TODO: Update the saver code
        if opts.pretrained_path is not None:
            self.network = self.saver.load_model(opts.pretrained_path,
                                                 self.network)
            self.logger.print('# Load model in {}'.format(opts.pretrained_path))

        # check the network
        _model_prarms = self.network.state_dict()
        _network_params = self.network._networks.state_dict()
        assert len(_model_prarms) == len(_network_params),\
            'All trainable parameters should ONLY be in the model._network.'

        # put the model into mutli-gpu
        if self.world_size > 1:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.network)
            self.network = DDP(self.network,
                               device_ids=[opts.local_rank],
                               output_device=opts.local_rank,
                               find_unused_parameters=True)
            self.network.ddp_rank = opts.local_rank
            self.network.world_size = self.world_size
            self.network.train_forward = self.network.module.train_forward
            self.network.inference_forward = self.network.module.inference_forward
            self.network.module.ddp_forward = self.network.forward
        
        # Initialize the optimizers and the schedulers
        if self.world_size > 1:
            param_groups = self.network.module.get_parameters()
        else:
            param_groups = self.network.get_parameters()
        
        self.optimizers = {}
        for group_name, (group_settings, st_epoch) in param_groups.items():
            optim_info = opts_dic['losses'][group_name]['optim']
            for setting in group_settings:
                if 'lr' in setting:
                    setting['lr'] *= optim_info['lr']
                
            if optim_info['type'] == 'Adam':
                optimizer = optim.Adam(group_settings,
                                       optim_info['lr'],
                                       **optim_info['params'])
            elif optim_info['type'] == 'AdamW':
                optimizer = optim.AdamW(group_settings,
                                        optim_info['lr'],
                                        **optim_info['params'])
            sched_info = optim_info['sched']
            if sched_info['type'] == 'Step':
                scheduler = sched.MultiStepLR(optimizer, 
                                              **sched_info['params'])
            self.optimizers[group_name] = (optimizer, scheduler, st_epoch)
        
        # Load the optimizers
        if opts.start_epoch is not None:
            self.epoch = opts.start_epoch
            self.batch_step = 1
            temp_epoch = 1
            while temp_epoch < self.epoch:
                temp_epoch += 1
                for group_name, _ in self.optimizers.items():
                    self.optimizers[group_name][1].step()
            
        else:
            if opts.pretrained_path is not None:
                (self.optimizers,
                 self.epoch,
                 self.batch_step) = self.saver.load_optim(opts.pretrained_path,
                                                          self.optimizers)
                self.logger.print('# Load optimizers in {}'
                                  .format(opts.pretrained_path))       
            else:
                self.epoch = 1
                self.batch_step = 1      
        
        # Initialize the visualizer
        if 'visual' in opts_dic:
            self.visualizer = Visualizer(os.path.join(self.logger.get_log_dir,
                                                      'image'),
                                         opts_dic['visual'],
                                         rank_id=opts.local_rank)
        else:
            self.visualizer = False

        # compute the network flops and inference time
        if opts.compute_flops:
            with torch.no_grad():
                input_size = opts_dic['pred_size']
                self.network.eval()
                from thop import profile
                if self.world_size > 1:
                    out_modes = self.network.module.out_mode
                else:
                    out_modes = self.network.out_mode
                for used_mode in out_modes:
                    # generate the input tensor
                    if used_mode == 'Mono' or used_mode == 'Refine':
                        input_tensor = torch.rand(1, 3, *input_size)
                    elif used_mode == 'Stereo':
                        input_tensor = torch.rand(1, 6, *input_size)
                    else:
                        raise NotImplementedError
                    input_tensor = input_tensor.to(self.device)

                    # compute the flops and parameters with thop
                    if self.world_size > 1:
                        self.network.module.used_out_mode = used_mode
                    else:
                        self.network.used_out_mode = used_mode
                    flops, p_nums = profile(self.network,
                                            inputs=(input_tensor, {})) 
                    
                    # compute the fps with no more than 1000 iterations
                    max_iter_num = 1000
                    inferece_time = []
                    for i in range(max_iter_num):
                        st_time = time.time()
                        _ = self.network(input_tensor, {})
                        temp_time = time.time() - st_time
                        inferece_time.append(temp_time)
                        if (i + 1) % 25 == 0 and\
                            len(inferece_time) >=100:
                            _time = inferece_time[-100:]
                            if np.std(_time) / np.mean(_time) < 0.005:
                                break  
                    
                    gpu_name = torch.cuda.get_device_name(self.device)
                    self.logger.log_for_flops_etc(list(input_tensor.shape),
                                                  used_mode,
                                                  flops,
                                                  p_nums,
                                                  gpu_name,
                                                  np.mean(inferece_time),
                                                  np.mean(inferece_time[-100:]),
                                                  i + 1,
                                                  max_iter_num)


    def _set_all_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _stable(self, dataloader, seed):
        self._set_all_seed(seed)
        return dataloader

    def _process_epoch(self):
        st_batch_time = time.time()
        for inputs in self._stable(self.train_loader, self.seed + self.epoch):
            for ipt_key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[ipt_key] = ipt.to(self.device, non_blocking=True)
            data_time = time.time() - st_batch_time
            outputs, losses, times = self.network.train_forward(inputs,
                                                                self.optimizers,
                                                                self.epoch)
            
            # stop the training if loss is nan
            if self.world_size != 1:
                with torch.no_grad():
                    dist.reduce(losses['loss'], dst=0)
                    if opts.local_rank == 0:
                        losses['loss'] /= self.world_size

            show_loss = losses['loss']
            if torch.isnan(show_loss):
                for k, v in losses.items():
                    if '-value' in k:
                        self.logger.print(k, v)
                self.saver.save_model(self.network,
                                      self.optimizers,
                                      self.epoch,
                                      self.batch_step,
                                      None,
                                      name='nan')
                exit()

            if self.batch_step % opts.log_freq == 0:
                self.logger.log_for_batch(self.epoch,
                                          self.batch_step,
                                          show_loss,
                                          data_time,
                                          times['fp_time'],
                                          times['fp_time'],
                                          losses)

            if self.visualizer and (self.batch_step % opts.visual_freq == 0
                                    or self.batch_step == 1):
                self.visualizer.update_visual_dict(inputs, outputs, losses)
                img_name = '{}-{}'.format(self.epoch, self.batch_step)
                self.visualizer.do_visualizion(img_name)

            self.batch_step += 1
            del outputs
            del losses
            st_batch_time = time.time()

    def _test_model(self): 
        self.network.eval()

        test_data_num = self.val_dataset_len
        idx = 0
        for inputs in self.val_loader:
            for ipt_key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[ipt_key] = ipt.to(self.device, non_blocking=True)

            outputs = self.network.inference_forward(inputs)
            for sou_name, metric_item in self.metric:
                metric_item.update_metric(outputs, inputs, name=sou_name)
            idx += 1
            if opts.local_rank == 0:
                print('{}/{}'.format(idx, test_data_num), end='\r')

        for sou_name, metric_item in self.metric:
            is_best = metric_item.compute_best_metric()
            self.saver.save_model(self.network,
                                  self.optimizers,
                                  self.epoch,
                                  self.batch_step,
                                  is_best,
                                  sou_name)

            self.logger.log_for_test(metric_item.get_metric_output(), is_best, sou_name)
            metric_item.clear_metric()
        
        if self.world_size > 1:
            dist.barrier()


    def do_train(self):
        self.logger.log_for_start_testing()
        self.epoch -= 1 # for save the model with correct epoch number
        with torch.no_grad():
            self._test_model()
        self.epoch += 1
        while self.epoch <= opts.epoch:
            st_epoch_time = time.time()
            # start training
            self.logger.log_for_start_training(self.optimizers, self.epoch)
            self.network.train()
            if self.world_size != 1:
                self.train_sampler.set_epoch(self.seed + self.epoch)
            self._process_epoch()
            # save the model
            if opts.save_freq is not None and self.epoch % opts.save_freq == 0:
                self.saver.save_model(self.network,
                                      self.optimizers,
                                      self.epoch,
                                      self.batch_step,
                                      None,
                                      name=self.epoch)
            # start testing
            if self.epoch % opts.test_freq == 0:
                self.logger.log_for_start_testing()
                with torch.no_grad():
                    self._test_model()

            for _, (_, scheduler, _) in self.optimizers.items():
                scheduler.step()

            # do log
            self.logger.log_for_epoch(self.epoch,
                                      time.time() - st_epoch_time, opts.epoch)

            self.network.train()
            self.epoch += 1


if __name__ == '__main__':
    env_info_dict = get_env_info()
    if env_info_dict['GPU Number'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opts.local_rank)

    trainer = Trainer(env_info_dict)
    trainer.do_train()
