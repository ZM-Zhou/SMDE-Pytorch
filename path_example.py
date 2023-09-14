class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == "eigen_kitti_test_jpg":
            return '/zhouzm/Datasets/eigen_kitti_test_jpg'
        elif name == "kitti_stereo2015":
            return '/zhouzm/Datasets/kitti_2015'
        elif name == 'make3d':
            return '/zhouzm/Datasets/Make3D'
        elif name == 'nyuv2':
            return '/zhouzm/Datasets/NYU_v2/nyu_test'
        elif name == 'cityscapes':
            return '/zhouzm/Datasets/NYU_v2/cityscapes'
        elif name == 'swin':
            return '/zhouzm/Download/swin_tiny_patch4_window7_224.pth'
        else:
            raise NotImplementedError