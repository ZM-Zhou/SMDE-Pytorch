class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == "kitti":
            return '/zhouzm/Datasets/kitti'
        elif name == "eigen_kitti_test_jpg":
            return '/zhouzm/Datasets/eigen_kitti_test_jpg'
        else:
            raise NotImplementedError