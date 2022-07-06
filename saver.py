import os

import torch


def load_model_for_evaluate(pre_model_path, model):
    pretrained_dict = torch.load(pre_model_path)
    model._networks.load_state_dict(pretrained_dict['model_params'])
    return model


class ModelSaver(object):
    def __init__(self, log_dir, is_parallel=False, rank_id=0):
        self.log_dir = os.path.join(log_dir, 'model')
        self.is_parallel = is_parallel
        self.rank_id = rank_id

        os.makedirs(self.log_dir, exist_ok=True)

    def _parallel_mask(func):
        def inner(self, *args, **kwargs):
            if self.rank_id == 0:
                ret = func(self, *args, **kwargs)
                return ret
            else:
                pass

        return inner

    @_parallel_mask
    def save_model(self, model, optimizers, epoch, step, is_best, name=''):
        if self.is_parallel:
            save_dict = {'model_params': model.module._networks.state_dict()}
        else:
            save_dict = {'model_params': model._networks.state_dict()}

        if is_best:
            save_path = os.path.join(self.log_dir, 'best_model.pth')
            torch.save(save_dict, save_path)

        save_dict['epoch'] = epoch
        save_dict['step'] = step
        for idx_optim in range(len(optimizers)):
            optimizer = optimizers[idx_optim]
            save_dict[('optim_params', idx_optim)] = optimizer.state_dict()
        save_path = os.path.join(self.log_dir, 'last_model{}.pth'.format(name))
        torch.save(save_dict, save_path)
        return is_best

    def load_model(self,
                   pre_model_path,
                   model,
                   optimizer=None,
                   scheduler=None):
        map_location = torch.device('cpu')
        load_dict = torch.load(pre_model_path, map_location)
        pretrained_dict = load_dict['model_params']
        model_dict = model._networks.state_dict()

        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }

        miss, unexpected = model._networks.load_state_dict(
            pretrained_dict, False)
        if miss is not None:
            print(miss)
        if unexpected is not None:
            print(unexpected)
        if 'epoch' in load_dict:
            start_epoch = load_dict['epoch'] + 1
            start_step = load_dict['step']
        else:
            start_epoch = 1
            start_step = 1
        return model, start_epoch, start_step

    def load_optim(self, pre_model_path, optimizers=None, schedulers=None):
        map_location = torch.device('cpu')
        load_dict = torch.load(pre_model_path, map_location)

        start_epoch = load_dict['epoch'] + 1

        for idx_optim in range(len(optimizers)):
            optimizer = optimizers[idx_optim]
            scheduler = schedulers[idx_optim]
            optimizer.load_state_dict(load_dict[('optim_params', idx_optim)])
            for params in optimizer.param_groups:
                params['lr'] = params['initial_lr']

            temp_epoch = 1
            while temp_epoch < start_epoch:
                temp_epoch += 1
                scheduler.step()
            optimizers[idx_optim] = optimizer
            schedulers[idx_optim] = scheduler

        return optimizers, schedulers
