class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == "eigen_kitti_test_jpg":
            return '/zhouzm/Datasets/eigen_kitti_test_jpg'
        elif name == 'nyuv2':
            return '/zhouzm/Datasets/NYU_v2/nyu_test'
        else:
            raise NotImplementedError