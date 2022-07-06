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

    return opts_dic


def _update_dic(dic, base_dic):
    base_dic = base_dic.copy()
    for key, val in dic.items():
        if isinstance(val, dict) and key in base_dic:
            # read a sub-dictionary of options
            base_dic[key] = _update_dic(val, base_dic[key])
        elif isinstance(val, list) and key in base_dic:
            # for read losses
            for target_item in val:
                is_exist = False
                is_losses = True
                if not isinstance(base_dic[key], list):
                    is_losses = False
                else:
                    # search all the existing losses
                    for list_idx, list_item in enumerate(base_dic[key]):
                        if isinstance(list_item, dict):
                            # update a existing loss term
                            if list_item['name'] == target_item['name']:
                                if ('type' in target_item
                                        and target_item['type'] == None):
                                    base_dic[key].pop(list_idx)
                                else:
                                    base_dic[key][list_idx] =\
                                        _update_dic(target_item,
                                                    base_dic[key][list_idx])
                                is_exist = True
                        else:
                            # this flag is used to record the options
                            # in list format, but not a loss term.
                            is_losses = False
                            break
                    # add a new loss term if the loss is not exist
                    if isinstance(list_item, dict) and not is_exist:
                        base_dic[key].append(target_item)
                if not is_losses:
                    base_dic[key] = val
                    break

        else:
            base_dic[key] = val
    dic = base_dic
    return dic
