import numpy as np
from solaris.nets.transform import Rotate, RandomScale


class TestRotate(object):
    """Test the rotation transform."""

    def test_rotate(self):
        rot = Rotate()
        arr = np.array([[3, 5, 3],
                        [5, 8, 10],
                        [1, 3, 8]])
        rot_arr = rot.apply(arr, angle=45)

        assert np.array_equal(rot_arr,
                              np.array([[5, 7, 5, 10],
                                        [5, 6, 10, 7],
                                        [5, 5, 4, 9],
                                        [5, 2, 3, 3]]))


class TestRandomScale(object):
    """Test the random scale transform."""

    def test_random_scale(self):
        rs = RandomScale(scale_limit=0.2)
        assert rs.scale_limit == (0.8, 1.2)
        arr = np.array([[3, 5, 3],
                        [5, 8, 10],
                        [1, 3, 8]])
        scaled_arr = rs.apply(arr, scale_x=2, scale_y=2)
        assert np.array_equal(scaled_arr, np.array([[3, 3, 5, 5, 2, 2],
                                                    [3, 4, 5, 6, 4, 4],
                                                    [5, 6, 7, 8, 9, 9],
                                                    [4, 5, 6, 8, 10, 10],
                                                    [2, 2, 3, 5, 8, 9],
                                                    [1, 1, 2, 4, 7, 8]],
                                                   dtype='uint8'))
