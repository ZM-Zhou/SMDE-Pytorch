import torch
import torch.nn as nn

from models.get_models import get_losses_with_opts


class Base_of_Network(nn.Module):
    """The Base class of networks which takes the input data and compute the
    predictions and losses at training and inference stages."""
    def __init__(self, options, loss_options, device):
        super().__init__()
        self.loss_options = loss_options
        self.device = device

        self.loss_computer = get_losses_with_opts(loss_options, device)
        self._initialize_model(**options)

    def _initialize_model(self):
        raise NotImplementedError

    def _compute_losses(self, outputs, train_side, losses, add_loss=True):
        loss_inputs = {}
        for out_key, out_vlaue in outputs.items():
            loss_inputs[out_key] = out_vlaue
        for in_key, in_vlaue in self.inputs.items():
            loss_inputs[in_key] = in_vlaue

        for used_loss in self.loss_options['types']:
            loss_name = used_loss['name']
            loss_rate = used_loss['rate']
            loss = self.loss_computer[loss_name](loss_inputs, train_side)
            if 'mask' in used_loss:
                mask = loss_inputs[used_loss['mask'].format(train_side)]
                mask = mask.to(torch.float)
                loss = loss * mask

            loss_value = loss_rate * loss.mean()
            if add_loss:
                losses['loss'] += loss_value

            losses['{}/{}'.format(loss_name, train_side)] = loss
            losses['{}/{}-value'.format(train_side, loss_name)] = loss_value

    def _add_final_losses(self, train_side, losses):
        for used_loss in self.loss_options['types']:
            loss_name = used_loss['name']
            loss_value = losses['{}/{}-value'.format(train_side, loss_name)]
            losses['loss'] += loss_value

    def get_parameters(self, base_lr=None):
        return [self.parameters()]

    @property
    def network_info(self):
        net_infos = []
        params_num = sum(x.numel() for x in self._networks.parameters())
        net_infos.append('    -params: {}'.format(params_num / 1000000))
        for key, val in self.init_opts.items():
            if key != 'self':
                net_infos.append('      {}: {}'.format(key, val))
        loss_infos = []
        for used_loss in self.loss_options['types']:
            loss_name = used_loss['name']
            loss_rate = used_loss['rate']
            loss_func = self.loss_computer[loss_name]
            loss_infos.append('      {} : rate={:.5f}'.format(
                loss_name, loss_rate))
            for key, val in loss_func.init_opts.items():
                if key not in ['self', 'device', '__class__']:
                    loss_infos.append('        {}: {}'.format(key, val))
        return net_infos, loss_infos
