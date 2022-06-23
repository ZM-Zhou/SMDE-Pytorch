import os
import numpy as np
import json

dataset_root = "/zhouzm/Datasets/cityscapes"
split_file = "data_splits/cityscapes/train_extra_stereo_list.txt"

with open(split_file, "r") as f:
    lines = f.readlines()


for line in lines:
    line = line.replace("\n", "")
    json_path = dataset_root + "/"\
              + line.replace("leftImg8bit", "camera").replace("png", "json")
    with open(json_path, "r") as json_file:
        matrix_dic = json.load(json_file)
    
    fuse_matrix = np.zeros((4, 4))
    fuse_matrix[0, 0] = matrix_dic["intrinsic"]["fx"]
    fuse_matrix[0, 2] = matrix_dic["intrinsic"]["u0"]
    fuse_matrix[1, 1] = matrix_dic["intrinsic"]["fy"]
    fuse_matrix[1, 2] = matrix_dic["intrinsic"]["v0"]
    fuse_matrix[0, 3] = matrix_dic["extrinsic"]["baseline"]
    fuse_matrix[2, 2] = 1
    fuse_matrix[3, 3] = 1

    save_path = json_path.replace("camera", "matrix").replace("json", "npy")
    save_dir = save_path.replace(save_path.split("/")[-1], "")
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, fuse_matrix)