import subprocess
import sys
from collections import defaultdict
from os import path

import torch
import torch.utils.cpp_extension


def get_env_info():
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        gpu_number = torch.cuda.device_count()
        env_info['GPU Number'] = gpu_number
        devices = defaultdict(list)
        for k in range(gpu_number):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and path.isdir(CUDA_HOME):
            try:
                nvcc = path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(f'"{nvcc}" -V | tail -n1',
                                               shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    return env_info


if __name__ == '__main__':
    env_info_dict = get_env_info()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    print(env_info)
