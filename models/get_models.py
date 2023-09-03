import utils.platform_manager as manager


def get_model_with_opts(opts_dic, device):
    model_name = opts_dic['model']['type']

    if model_name in manager.MODELS.modules_dict:
        Model = manager.MODELS[model_name]
    else:
        raise NotImplementedError(
            'The model was not found {}'.format(model_name))

    if 'params' in opts_dic['model']:
        model_params = opts_dic['model']['params']
    else:
        model_params = {}
    used_losses = opts_dic['losses']

    return Model(model_params, used_losses, device).to(device)


def get_losses_with_opts(opts_dic, device):
    if opts_dic is None:
        return [], None
    loss_computer = {}
    forward_st_epochs = {}
    for group_name, group_opts in opts_dic.items():
        forward_st_epochs[group_name] = group_opts['st_epoch']
        loss_items = {}
        for loss_name, loss_opts in group_opts['loss_terms'].items():
            loss_type = loss_opts['type']
            if loss_type in manager.LOSSES.modules_dict:
                Loss = manager.LOSSES[loss_type]
            else:
                raise NotImplementedError(
                    'The loss was not found {}'.format(loss_type))
            loss_args = loss_opts['args']
            loss_args['device'] = device
            loss_items[loss_name] = (Loss(**loss_args),
                                     loss_opts['rate'],
                                     loss_opts['mask'] if 'mask' in loss_opts\
                                                       else None)
        
        loss_computer[group_name] = loss_items

    return forward_st_epochs, loss_computer
