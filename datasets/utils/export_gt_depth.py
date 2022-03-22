from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil

import os
import sys
sys.path.append(os.getcwd())
from datasets.utils.data_reader import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    opt = parser.parse_args()

    split_folder = os.path.join(os.getcwd(), "data_splits", "kitti")

    with open(os.path.join(split_folder, "test_list.txt"), "r") as f:
        lines = f.readlines()

    print("Exporting ground truth depths for {}".format("eigen"))

    gt_depths = []
    for line in lines:

        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

       
        calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
        velo_filename = os.path.join(opt.data_path, folder,
                                        "velodyne_points/data", "{:010d}.bin".format(frame_id))
        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(opt.data_path, "gt_depths.npz")

    print("Saving to {}".format("eigen"))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()