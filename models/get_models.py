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
    used_losses = opts_dic['loss']

    return Model(model_params, used_losses, device).to(device)


def get_losses_with_opts(opts_dic, device):
    loss_computer = {}
    if opts_dic['types'] is None:
        return None
    for loss_opts in opts_dic['types']:
        loss_type = loss_opts['type']
        if loss_type in manager.LOSSES.modules_dict:
            Loss = manager.LOSSES[loss_type]
        else:
            raise NotImplementedError(
                'The loss was not found {}'.format(loss_type))
        loss_name = loss_opts['name']
        loss_args = loss_opts['args']
        loss_args['device'] = device
        loss_computer[loss_name] = Loss(**loss_args)

    return loss_computer
