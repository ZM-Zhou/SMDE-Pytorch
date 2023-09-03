import os
import time

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Text logger for training.

    Create a log folder and file based on date and experiment's name.
    """
    def __init__(self, base_dir, exp_name, rank_id=0):
        self.rank_id = rank_id

        self.exp_start_time = time.time()
        self.log_dir = os.path.join(base_dir,
                                    self._get_time_now() + '_' + exp_name)
        text_dir = os.path.join(self.log_dir, 'logout')
        os.makedirs(text_dir, exist_ok=True)
        self.meta_file = os.path.join(text_dir, '{}_meta.txt'.format(exp_name))
        self.train_file = os.path.join(text_dir,
                                       '{}_trainlog.txt'.format(exp_name))
        if self.rank_id == 0:
            self.writer = SummaryWriter(
                os.path.join(self.log_dir, 'tensorboard'))
            self._set_log_folder_and_file()
            self._log_initialization(exp_name)

    def _parallel_mask(func):
        def inner(self, *args, **kwargs):
            # print(self.rank_id)
            if self.rank_id == 0:
                ret = func(self, *args, **kwargs)
                return ret
            else:
                pass

        return inner

    def _get_time_now(self):
        stamp = time.time()
        date = time.localtime(stamp)
        format_date = time.strftime('%Y-%m-%d_%Hh%Mm%Ss', date)
        return format_date

    def _set_log_folder_and_file(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _train_print(self, arg, *args, **kargs):
        with open(self.train_file, 'a+') as f:
            if kargs:
                print(arg, end=kargs['end'])
                f.write(arg + kargs['end'])
            else:
                print(arg)
                f.write(str(arg) + '\n')

    def _meta_print(self, arg, *args, **kargs):
        with open(self.meta_file, 'a+') as f:
            if kargs:
                print(arg, end=kargs['end'])
                f.write(arg + kargs['end'])
            else:
                print(arg)
                f.write(str(arg) + '\n')

    def _log_initialization(self, exp_name):
        self._meta_print('#' + '-' * 79)
        self._meta_print('# Monocular depth esitmation with Pytorch')
        self._meta_print('    -Experiment: {}'.format(exp_name))
        self._meta_print('    -Start at: {}'.format(self._get_time_now()))
        self._meta_print('#' + '-' * 79)
        self._meta_print('# Logger Initialization Done!')

    @property
    def get_log_dir(self):
        return self.log_dir

    @_parallel_mask
    def print(self, arg, *args, **kwargs):
        self._meta_print(arg, *args, **kwargs)

    @_parallel_mask
    def log_for_env(self, env_info):
        """Log for the environment information."""
        self._meta_print('#' + '-' * 79)
        self._meta_print('# Environment Information')
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info.items()])
        self._meta_print(env_info)

    @_parallel_mask
    def log_for_opts(self, options):
        """Log for the options."""
        self._meta_print('#' + '-' * 79)
        self._meta_print('# Options')
        options_info = '\n'.join(
            [f'  - {k}: {v}' for k, v in sorted(vars(options).items())])
        self._meta_print(options_info)

    @_parallel_mask
    def log_for_data(self, tra_data, val_data, tra_len, val_len, num_workers):
        """Log for the initializaton of datasets and dataloader."""
        self._meta_print('#' + '-' * 79)
        self._meta_print('# Used Dataset')
        loader_line = '      {} iters with {} workers'
        for line in tra_data:
            self._meta_print(line)
        self._meta_print(loader_line.format(tra_len, num_workers))
        for line in val_data:
            self._meta_print(line)
        self._meta_print(loader_line.format(val_len, num_workers))

        self._meta_print('# Datasets and Dataloaders Initialization Done!')

    @_parallel_mask
    def log_for_model(self, model_name, model_info, loss_info):
        """Log for the initializaton of the model and losses."""
        self._meta_print('#' + '-' * 79)
        self._meta_print('# Model and Losses')
        self._meta_print('    -{}'.format(model_name))
        for line in model_info:
            self._meta_print(line)
        self._meta_print('    -losses')
        for line in loss_info:
            self._meta_print(line)
        self._meta_print('# Model Initialization Done!')
    
    @_parallel_mask
    def log_for_flops_etc(self,
                          in_shape,
                          used_mode,
                          flops,
                          p_nums,
                          gpu_name,
                          total_mean_time,
                          inferece_time,
                          total_compute_iter,
                          max_iter_num):
        self._meta_print('#' + '-' * 79)
        self._meta_print('# FlOPs and Inference fps')
        self._meta_print('  Input size: {}, and {} out mode'.format(in_shape,
                                                                    used_mode))
        self._meta_print('   -FLOPs: {:.3f}G'.format(flops / 1e9))
        self._meta_print('   -Params: {:.3f}M'.format(p_nums / 1e6))
        
        self._meta_print('   -on ' + gpu_name + ':')
        total_num = min(total_compute_iter, max_iter_num)
        total_fps = 1 / total_mean_time
        last_fps = 1 / inferece_time
        self._meta_print(
            '      {:.3f} / fps in total {} images'.format(total_fps,
                                                           total_num))
        self._meta_print(
            '      {:.3f} / fps in last(stable) 100 images'.format(last_fps))


    @_parallel_mask
    def log_for_start_training(self, optimizers, epoch):
        self._train_print('#' + '-' * 79)
        self._train_print('# Training Start!')
        for group_name, (optimizer, _, st_epoch) in optimizers.items():
            params_groups = optimizer.param_groups
            for param_idx, param in enumerate(params_groups):
                self._train_print(
                    'optimizer of {}, sub-group {}: lr {:.7f}{}'.format(
                        group_name, 
                        param_idx, 
                        param['lr'],
                        ' but disabled' if epoch < st_epoch else ''))

    @_parallel_mask
    def log_for_batch(self,
                      epoch,
                      batch_step,
                      loss,
                      data_t,
                      fp_t,
                      bp_t,
                      losses=None):
        """Log for batch processing."""
        info_line = 'epoch: {:3d} step：{:7d}| loss: {:7.4f}|' + \
                    ' data={:.3f} fp={:.3f} bp={:.3f}'
        loss = loss.cpu()
        self._train_print(
            info_line.format(epoch, batch_step, loss, data_t, fp_t, bp_t))
        if losses is not None:
            for k, v in losses.items():
                if 'value' in k or k == 'loss':
                    self.writer.add_scalar('{}'.format(k), v.detach(),
                                           batch_step)

    @_parallel_mask
    def log_for_start_testing(self):
        self._train_print('#' + '-' * 79)
        self._train_print('# Testing Start!')

    def log_for_test(self, metric_outputs, is_best, name=None):
        self._train_print('# Testing Result of {}:'.format(name))
        if is_best:
            self._train_print('    -Best Now!:')
        self._train_print(metric_outputs[0] + '\n' + metric_outputs[1])

    @_parallel_mask
    def log_for_epoch(self, epoch, epoch_time, end_epoch):
        rest_epoch = end_epoch - epoch
        rest_time = epoch_time * rest_epoch
        now_time = time.time()
        run_time = now_time - self.exp_start_time
        end_time = now_time + rest_time

        run_t = int(run_time)
        run_s = run_t % 60
        run_t //= 60
        run_m = run_t % 60
        run_t //= 60
        format_run_date = '{:02d}h{:02d}m{:02d}s'.format(run_t, run_m, run_s)

        date = time.localtime(end_time)
        format_end_date = time.strftime('%Y-%m-%d_%Hh%Mm%Ss', date)

        self._train_print(
            '# The experiment has been running for {}'.format(format_run_date))
        self._train_print(
            '# The experiment will end on {}'.format(format_end_date))
