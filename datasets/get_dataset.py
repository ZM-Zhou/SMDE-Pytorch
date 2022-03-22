from torch.utils.data import ConcatDataset

import utils.platform_manager as manager


def get_dataset_with_opts(opts_dic, mode):
    dataset_opts = opts_dic['{}_dataset'.format(mode)]
    if isinstance(dataset_opts, list):
        used_datasets = []
        dataset_info = []
        for dataset_part_opts in dataset_opts:
            dataset_name = dataset_part_opts['type']
            if dataset_name in manager.DATASETS.modules_dict:
                Dataset = manager.DATASETS[dataset_name]
            else:
                raise NotImplementedError(
                    'The dataset was not found {}'.format(dataset_name))
            _params = dataset_part_opts['params']
            _params['dataset_mode'] = mode
            used_datasets.append(Dataset(**_params))
            for info_line in used_datasets[-1].dataset_info:
                dataset_info.append(info_line)

        cat_dataset = ConcatDataset(used_datasets)
        cat_dataset.dataset_info = dataset_info
        return cat_dataset

    else:
        dataset_name = opts_dic['{}_dataset'.format(mode)]['type']
        if dataset_name in manager.DATASETS.modules_dict:
            Dataset = manager.DATASETS[dataset_name]
        else:
            raise NotImplementedError(
                'The dataset was not found {}'.format(dataset_name))

        _params = opts_dic['{}_dataset'.format(mode)]['params']
        _params['dataset_mode'] = mode

    return Dataset(**_params)
