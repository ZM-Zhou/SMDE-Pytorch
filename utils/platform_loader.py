import yaml


# ----------------------------------------------------------------------------
# yaml reader
# ----------------------------------------------------------------------------
def read_yaml_options(path):
    with open(path, 'r', encoding='utf-8') as f:
        opts = f.read()
    opts_dic = yaml.load(opts, Loader=yaml.FullLoader)
    dic_keys = list(opts_dic.keys())

    for dic_key in dic_keys:
        if '_base' in dic_key:
            base_path = opts_dic.pop(dic_key)
            base_dic = read_yaml_options(base_path)
            opts_dic = _update_dic(opts_dic, base_dic)
        if '_train' in dic_key:
            base_path = opts_dic.pop(dic_key)
            base_dic = read_yaml_options(base_path)
            base_dic['train_dataset'] = base_dic.pop('_dataset')
            opts_dic = _update_dic(opts_dic, base_dic)
        if '_val' in dic_key:
            base_path = opts_dic.pop(dic_key)
            base_dic = read_yaml_options(base_path)
            base_dic['val_dataset'] = base_dic.pop('_dataset')
            opts_dic = _update_dic(opts_dic, base_dic)
        if '_test' in dic_key:
            base_path = opts_dic.pop(dic_key)
            base_dic = read_yaml_options(base_path)
            base_dic['test_dataset'] = base_dic.pop('_dataset')
            opts_dic = _update_dic(opts_dic, base_dic)
    
    _check_losses(opts_dic)

    return opts_dic


def _update_dic(dic, base_dic):
    '''
    we assert that all the options value will be replaced if they are not dictionary.
    '''
    base_dic = base_dic.copy()
    for key, value in dic.items():
        if isinstance(value, dict) and key in base_dic:
            # for the 'type' is None, remove it from the dictionary,
            # which should be only appear in the loss terms and datasets
            if 'type' in value and 'type' == None:
                base_dic.pop(key)
            else:
                base_dic[key] = _update_dic(value, base_dic[key])
        else:
            base_dic[key] = value
        dic = base_dic
    return dic

def _check_losses(dic):
    if 'loss_terms' not in dic and 'losses' not in dic:
        return None
    if 'losses' not in dic:
        dic['losses'] = {'param_group':{
                            'st_epoch': 1,
                            '_optim': 'options/_base/runtime/adam.yaml',
                            'loss_terms': dic.pop('loss_terms')}}

    for params_name, setting_group in dic['losses'].items():
        if 'st_epoch' not in setting_group :
            dic['losses'][params_name]['st_epoch'] = 0
        if '_optim' not in setting_group:
            optim_setting = 'options/_base/optimizers/adam.yaml'
        else:
            optim_setting = dic['losses'][params_name].pop('_optim')
 
        opts_optim = read_yaml_options(optim_setting)
        if 'optim' not in dic['losses'][params_name]:
            dic['losses'][params_name]['optim'] = opts_optim
        else:
            dic['losses'][params_name]['optim'] = \
                _update_dic(dic['losses'][params_name]['optim'], opts_optim)

if __name__ == '__main__':
    opts = read_yaml_options('options/TiO-Depth/train/tio_depth_kitti.yaml')
    print(opts)