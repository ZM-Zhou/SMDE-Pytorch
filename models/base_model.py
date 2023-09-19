import torch
import torch.nn as nn
from torch import distributed as dist
import time

from models.get_models import get_losses_with_opts


class Base_of_Model(nn.Module):
    """The vaisc class of the model which takes the input data and compute the
    predictions and losses at training and inference stages."""
    def __init__(self, options, loss_options, device):
        super().__init__()
        self.loss_options = loss_options
        self.device = device
        self.ddp_rank = -1
        self.world_size = 1
        self.is_train = False
        self.inputs = None
        self.out_mode = ['Mono']
        self.clip_grad = -1

        self.forward_st_epochs, self.loss_computer =\
            get_losses_with_opts(loss_options, device)
        self._initialize_model(**options)

    def _initialize_model(self):
        raise NotImplementedError

    def train_forward(self, inputs, optimizers, epoch):
        self.is_train = True
        self.now_group_idx = 0
        self.inputs = inputs
        outputs = {}
        losses = {'loss': 0.0}
        times = {'fp_time': 0.0, 'bp_time': 0.0}
        for group_name, st_epoch in self.forward_st_epochs.items():
            if epoch < st_epoch:
                continue
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = v.detach()
                elif isinstance(v, list):
                    for v_idx, v_item in enumerate(v):
                        if isinstance(v_item, torch.Tensor):
                            outputs[k][v_idx] = v_item.detach()

            self.now_group_name = group_name
            _st_fp = time.time()
            x = self._preprocess_inputs()
            if hasattr(self, 'ddp_forward') and callable(self.ddp_forward):
                _, outputs = self.ddp_forward(x, outputs)
            else:
                _, outputs = self.forward(x, outputs)
            loss_sides, outputs = self._postprocess_outputs(outputs)
            losses = self._compute_losses(loss_sides, outputs, losses)
            _st_bp = time.time()
            if losses[group_name + '-loss'] != 0:
                self._optim_params(optimizers[group_name][0],
                                losses[group_name + '-loss'])
            times['fp_time'] += _st_bp - _st_fp
            times['bp_time'] += time.time() - _st_bp
            self.now_group_idx += 1
        
        return outputs, losses, times
    
    def inference_forward(self, inputs, **kargs):
        self.inputs = inputs
        self.is_train = False
        outputs = {}
        for used_out_mode in self.out_mode:
            self.used_out_mode = used_out_mode
            x = self._preprocess_inputs()
            _, outputs = self.forward(x, outputs)
        return outputs
    
    def forward(self, x, outputs, **kargs):
        # ALL the forward propagation of the model with trainable parameters
        # and record the outputs in a simpliest way
        raise NotImplementedError
    
    def _preprocess_inputs(self, inputs):
        raise NotImplementedError
    
    def _postprocess_outputs(self, inputs, outputs):
        raise NotImplementedError

    def _compute_losses(self, sides, outputs, losses, add_loss=True):
        loss_inputs = {}
        for out_key, out_vlaue in outputs.items():
            loss_inputs[out_key] = out_vlaue
        for in_key, in_vlaue in self.inputs.items():
            loss_inputs[in_key] = in_vlaue

        loss_terms = self.loss_computer[self.now_group_name]
        losses[self.now_group_name + '-loss'] = 0
        for train_side in sides:
            for loss_name, (used_loss, loss_rate, loss_mask) in\
                loss_terms.items():
                loss = used_loss(loss_inputs, train_side)
                if loss_mask is not None:
                    mask = loss_inputs[loss_mask.format(train_side)]
                    mask = mask.to(torch.float)
                    loss = loss * mask
                    record_mean = loss.sum().detach() / mask.sum()
                else:
                    record_mean = loss.mean().detach()

                loss_value = loss_rate * loss.mean()
                if add_loss:
                    losses[self.now_group_name + '-loss'] += loss_value
                    losses['loss'] += loss_value.detach()

                losses['{}/{}'.format(loss_name, train_side)] = loss
                losses['{}/{}-value'.format(train_side, loss_name)] = record_mean
        
        return losses
    
    # def _add_final_losses(self, train_side, losses):
    #     loss_terms = self.loss_computer[self.now_group_name]
    #     losses[self.now_group_name + '-loss'] = 0
    #     for loss_name, (_, _, _) in loss_terms.items():
    #         loss_value = losses['{}/{}-value'.format(train_side, loss_name)]
    #         losses[self.now_group_name + '-loss'] += loss_value
    #         losses['loss'] += loss_value.detach()
    
    def _optim_params(self, optimizer, loss):
        optimizer.zero_grad()
        if self.ddp_rank != -1:
            with torch.no_grad():
                dist.reduce(loss, dst=0)
                if self.ddp_rank == 0:
                    loss /= self.world_size

        loss.backward()
        if self.clip_grad != -1:
            for params in optimizer.param_groups:
                params = params['params']
                torch.nn.utils.clip_grad_norm_(params,
                                               max_norm=self.clip_grad)
        optimizer.step()

    def get_parameters(self):
        return {'param_group': ([{'params': list(self.parameters())}], 0)}

    @property
    def network_info(self):
        net_infos = []
        params_num = sum(x.numel() for x in self._networks.parameters())
        net_infos.append('    -params: {}'.format(params_num / 1000000))
        for key, val in self.init_opts.items():
            if key != 'self':
                net_infos.append('      {}: {}'.format(key, val))
        loss_infos = []
        for group_name, st_epoch in self.forward_st_epochs.items():
            loss_infos.append('      {}: trained from {}'
                              .format(group_name, st_epoch))
            for loss_name, loss_opts in self.loss_computer[group_name].items():
                used_loss, loss_rate, loss_mask = loss_opts
                loss_infos.append('      -{} : rate={:.5f}, {}'.format(
                    loss_name, loss_rate, loss_mask))
                for key, val in used_loss.init_opts.items():
                    if key not in ['self', 'device', '__class__']:
                        loss_infos.append('        {}: {}'.format(key, val))

        return net_infos, loss_infos
