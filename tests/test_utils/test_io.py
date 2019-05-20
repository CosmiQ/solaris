import numpy as np
from solaris.utils.io import preprocess_im_arr


class TestPreprocessImArr(object):
    """Test image pre-processing."""

    def test_rescale_auto(self):
        expected_result = np.array([[[  0,   0,   0],
                                     [ 10,  10,  10],
                                     [ 21,  21,  21],
                                     [ 31,  31,  31],
                                     [ 42,  42,  42]],

                                    [[ 53,  53,  53],
                                     [ 63,  63,  63],
                                     [ 74,  74,  74],
                                     [ 85,  85,  85],
                                     [ 95,  95,  95]],

                                    [[106, 106, 106],
                                     [116, 116, 116],
                                     [127, 127, 127],
                                     [138, 138, 138],
                                     [148, 148, 148]],

                                    [[159, 159, 159],
                                     [170, 170, 170],
                                     [180, 180, 180],
                                     [191, 191, 191],
                                     [201, 201, 201]],

                                    [[212, 212, 212],
                                     [223, 223, 223],
                                     [233, 233, 233],
                                     [244, 244, 244],
                                     [255, 255, 255]]], dtype='uint8')
        im_arr = np.arange(5*5*3, 5*5*6).reshape(5, 5, 3).astype('uint16')
        normalized_arr = preprocess_im_arr(im_arr, 'uint16', rescale=True)

        assert np.array_equal(normalized_arr, expected_result)

    def test_rescale_single_vals(self):
        expected_result = np.array([[[ 77,  79,  80],
                                     [ 82,  83,  85],
                                     [ 86,  87,  89],
                                     [ 90,  92,  93],
                                     [ 94,  96,  97]],

                                    [[ 99, 100, 102],
                                     [103, 104, 106],
                                     [107, 109, 110],
                                     [111, 113, 114],
                                     [116, 117, 119]],

                                    [[120, 121, 123],
                                     [124, 126, 127],
                                     [128, 130, 131],
                                     [133, 134, 136],
                                     [137, 138, 140]],

                                    [[141, 143, 144],
                                     [145, 147, 148],
                                     [150, 151, 153],
                                     [154, 155, 157],
                                     [158, 160, 161]],

                                    [[162, 164, 165],
                                     [167, 168, 170],
                                     [171, 172, 174],
                                     [175, 177, 178],
                                     [179, 181, 182]]], dtype='uint8')
        im_arr = np.arange(5*5*3, 5*5*6).reshape(5, 5, 3).astype('uint16')
        normalized_arr = preprocess_im_arr(im_arr, 'uint16', rescale=True,
                                           rescale_min=20, rescale_max=200)

        assert np.array_equal(normalized_arr, expected_result)

    def test_rescale_limit_range(self):
        expected_result = np.array([[[  0,   0,   0],
                                     [  0,   0,   0],
                                     [  0,   0,   0],
                                     [  0,   0,   0],
                                     [  0,   0,   0]],

                                    [[  0,   5,  10],
                                     [ 15,  20,  25],
                                     [ 30,  35,  40],
                                     [ 45,  51,  56],
                                     [ 61,  66,  71]],

                                    [[ 76,  81,  86],
                                     [ 91,  96, 102],
                                     [107, 112, 117],
                                     [122, 127, 132],
                                     [137, 142, 147]],

                                    [[153, 158, 163],
                                     [168, 173, 178],
                                     [183, 188, 193],
                                     [198, 204, 209],
                                     [214, 219, 224]],

                                    [[229, 234, 239],
                                     [244, 249, 255],
                                     [255, 255, 255],
                                     [255, 255, 255],
                                     [255, 255, 255]]], dtype='uint8')
        im_arr = np.arange(5*5*3, 5*5*6).reshape(5, 5, 3).astype('uint16')
        normalized_arr = preprocess_im_arr(im_arr, 'uint16', rescale=True,
                                           rescale_min=90, rescale_max=140)

        assert np.array_equal(normalized_arr, expected_result)

    def test_rescale_channel_lists(self):
        expected_result = np.array([[[100,  83,  67],
                                     [105,  89,  72],
                                     [111,  94,  78],
                                     [116, 100,  83],
                                     [122, 105,  89]],

                                    [[127, 111,  94],
                                     [132, 116, 100],
                                     [138, 122, 105],
                                     [143, 127, 111],
                                     [149, 132, 116]],

                                    [[154, 138, 122],
                                     [160, 143, 127],
                                     [165, 149, 132],
                                     [171, 154, 138],
                                     [176, 160, 143]],

                                    [[182, 165, 149],
                                     [187, 171, 154],
                                     [193, 176, 160],
                                     [198, 182, 165],
                                     [204, 187, 171]],

                                    [[209, 193, 176],
                                     [214, 198, 182],
                                     [220, 204, 187],
                                     [225, 209, 193],
                                     [231, 214, 198]]], dtype='uint8')
        im_arr = np.arange(5*5*3, 5*5*6).reshape(5, 5, 3).astype('uint16')
        normalized_arr = preprocess_im_arr(im_arr, 'uint16', rescale=True,
                                           rescale_min=[20, 30, 40],
                                           rescale_max=[160, 170, 180])

        assert np.array_equal(normalized_arr, expected_result)
