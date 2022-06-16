import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

# ----------------------------------------------------------------------------
# Parse
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='SNDE-Pytorch Training Parser')

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

parser.add_argument('--optim_name',
                    dest='optim_name',
                    default='Adam',
                    help='The name of used optimizer')
parser.add_argument('-lr',
                    '--learning_rate',
                    dest='learning_rate',
                    type=float,
                    default=0.0001,
                    help='# of the network')
parser.add_argument('--decay_rate',
                    dest='decay_rate',
                    type=float,
                    default=0.5,
                    help='# of optimizer')
parser.add_argument('--decay_step',
                    dest='decay_step',
                    type=int,
                    nargs='+',
                    default=[30, 40],
                    help='# of optimizer')
parser.add_argument('--beta1',
                    dest='beta1',
                    type=float,
                    default=0.5,
                    help='# of Adam optimizer')

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
                    dest='pre_model',
                    default=None,
                    help='the path of pretrained model')
parser.add_argument('--start_epoch',
                    dest='start_epoch',
                    type=int,
                    default=None,
                    help='# of training')

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

parser.add_argument('--metric_name',
                    dest='metric_name',
                    type=str,
                    nargs='+',
                    default=['depth_kitti'],
                    help='metric type')
parser.add_argument('--best_compute',
                    dest='best_compute',
                    type=str,
                    default='depth_kitti',
                    help='metric for selecting best model')

opts = parser.parse_args()

# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size**2
            return tensor
        else:
            return 0


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

        # Initialize the random seed and device
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

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize the options
        opts_dic = read_yaml_options(opts.exp_opts)

        # Initialize the datasets and dataloaders
        if 'photo_rmse' in opts.metric_name:
            opts_dic['test_dataset']['params']['stereo_test'] = True
        train_dataset = get_dataset_with_opts(opts_dic, 'train')
        val_dataset = get_dataset_with_opts(opts_dic, 'val')
        self.test_dataset_size = len(val_dataset)

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
        self.val_loader = DataLoader(val_dataset,
                                     1,
                                     num_workers=opts.num_workers,
                                     shuffle=False,
                                     pin_memory=True)

        self.logger.log_for_data(train_dataset.dataset_info,
                                 val_dataset.dataset_info,
                                 len(self.train_loader), len(self.val_loader),
                                 opts.num_workers)

        # Initialize the saver and check the network
        self.saver = ModelSaver(self.logger.get_log_dir,
                                is_parallel=(world_size > 1),
                                rank_id=opts.local_rank)
        self.metric = Metric(opts.metric_name, opts.best_compute)

        # Initialize the network
        self.network = get_model_with_opts(opts_dic, self.device)
        net_info, loss_info = self.network.network_info
        self.logger.log_for_model(opts_dic['model']['type'], net_info,
                                  loss_info)

        # Load the pretrained model
        if opts.pre_model is not None:
            (self.network, self.epoch,
             self.batch_step) = self.saver.load_model(opts.pre_model,
                                                      self.network, None, None)
            self.logger.print('# Load model in {}'.format(opts.pre_model))
        else:
            self.epoch = 1
            self.batch_step = 1

        if self.world_size > 1:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.network)
            self.network = DDP(self.network,
                               device_ids=[opts.local_rank],
                               output_device=opts.local_rank,
                               find_unused_parameters=True)
        # check the network
        _model_prarms = self.network.state_dict()
        if self.world_size != 1:
            _network_params = self.network.module._networks.state_dict()
        else:
            _network_params = self.network._networks.state_dict()
        assert len(_model_prarms) == len(_network_params),\
            'All trainable parameters should ONLY be in the model._network.'

        # Initialize the optimizer and the scheduler
        if self.world_size > 1:
            param_groups = self.network.module.get_parameters(
                opts.learning_rate)
        else:
            param_groups = self.network.get_parameters(opts.learning_rate)
        self.optimizer = []
        self.scheduler = []
        for params in param_groups:
            if opts.optim_name == 'Adam':
                optimizer = optim.Adam(params,
                                       opts.learning_rate,
                                       betas=(opts.beta1, 0.999))
            if opts.optim_name == 'AdamW':
                optimizer = optim.AdamW(params,
                                        opts.learning_rate,
                                        betas=(opts.beta1, 0.999))
            scheduler = sched.MultiStepLR(optimizer, opts.decay_step,
                                          opts.decay_rate)
            self.optimizer.append(optimizer)
            self.scheduler.append(scheduler)

        # Load the pretrained model
        if opts.start_epoch is not None:
            self.epoch = opts.start_epoch
            for idx_sched in range(len(self.scheduler)):
                temp_epoch = 1
                while temp_epoch < self.epoch:
                    temp_epoch += 1
                    self.scheduler[idx_sched].step()
        else:
            if opts.pre_model is not None:
                self.optimizer, self.scheduler\
                    = self.saver.load_optim(opts.pre_model,
                                            self.optimizer, self.scheduler)
                self.logger.print('# Load optimizer in {}'.format(
                    opts.pre_model))

        # Initialize the visualizer
        if 'visual' in opts_dic:
            self.visualizer = Visualizer(os.path.join(self.logger.get_log_dir,
                                                      'image'),
                                         opts_dic['visual'],
                                         rank_id=opts.local_rank)
        else:
            self.visualizer = False

    def _process_epoch(self):
        st_batch_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            for ipt_key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[ipt_key] = ipt.to(self.device, non_blocking=True)
            st_fp_time = time.time()
            outputs, losses = self.network(inputs)
            st_bp_time = time.time()
            for optimizer_item in self.optimizer:
                optimizer_item.zero_grad()
            if self.world_size != 1:
                losses['loss'] = losses['loss'] * self.world_size
                reduce_loss(losses['loss'], opts.local_rank, self.world_size)
                show_loss = losses['loss']
            else:
                show_loss = losses['loss']

            if torch.isnan(losses['loss']):
                for k, v in losses.items():
                    if '-value' in k:
                        print(k, v)
                exit()
            for idx_optim, optimizer_item in enumerate(self.optimizer):
                if idx_optim == 0:
                    if len(self.optimizer) == 1:
                        losses['loss'].backward()
                    else:
                        losses['0-loss'].backward(retain_graph=True)
                else:
                    losses['loss'].backward()

                optimizer_item.step()
                for _optimizer_item in self.optimizer:
                    _optimizer_item.zero_grad()
            end_batch_time = time.time()

            # compute the process time
            data_time = st_fp_time - st_batch_time
            fp_time = st_bp_time - st_fp_time
            bp_time = end_batch_time - st_bp_time

            if self.batch_step % opts.log_freq == 0:
                self.logger.log_for_batch(self.epoch, self.batch_step,
                                          show_loss, data_time, fp_time,
                                          bp_time, losses)

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

        test_data_num = self.test_dataset_size
        idx = 0
        for inputs in self.val_loader:
            for ipt_key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[ipt_key] = ipt.to(self.device, non_blocking=True)

            outputs = self.network(inputs, is_train=False)
            self.metric.update_metric(outputs, inputs)
            idx += 1
            if opts.local_rank == 0:
                print('{}/{}'.format(idx, test_data_num - 1), end='\r')

        is_best = self.metric.compute_best_metric()
        self.saver.save_model(self.network, self.optimizer, self.epoch,
                              self.batch_step, is_best)

        self.logger.log_for_test(self.metric.get_metric_output(), is_best)
        self.metric.clear_metric()

    def do_train(self):
        self.logger.log_for_start_testing()
        with torch.no_grad():
            self._test_model()
        while self.epoch <= opts.epoch:
            st_epoch_time = time.time()
            # start training
            self.logger.log_for_start_training(self.optimizer)
            self.network.train()
            if self.world_size != 1:
                self.train_sampler.set_epoch(self.epoch)
            self._process_epoch()
            # save the model
            if opts.save_freq is not None and self.epoch % opts.save_freq == 0:
                self.saver.save_model(self.network,
                                      self.optimizer,
                                      self.epoch,
                                      self.batch_step,
                                      None,
                                      name=str(self.epoch))
            # start testing
            self.logger.log_for_start_testing()
            with torch.no_grad():
                self._test_model()
            # do log
            for scheduler_item in self.scheduler:
                scheduler_item.step()
            self.logger.log_for_epoch(self.epoch,
                                      time.time() - st_epoch_time, opts.epoch)

            self.network.train()
            self.epoch += 1


if __name__ == '__main__':
    env_info_dict = get_env_info()
    world_size = env_info_dict['GPU Number']
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opts.local_rank)
        global_rank = dist.get_rank()

    trainer = Trainer(env_info_dict)
    trainer.do_train()
