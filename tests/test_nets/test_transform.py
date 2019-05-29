import numpy as np
from solaris.nets.transform import Rotate, RandomScale, build_pipeline
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import HorizontalFlip, Normalize


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


class TestBuildAugPipeline(object):
    """Test sol.nets.transform.build_pipeline()."""

    def test_build_pipeline(self):
        config = {'training_augmentation':
                  {'p': 0.75,
                   'augmentations':
                   {'oneof':
                    {'normalize': {},
                     'rotate':
                     {'border_mode': 'reflect',
                      'limit': 45}
                     },
                    'horizontalflip': {'p': 0.5}
                    }
                   },
                  'validation_augmentation':
                  {'augmentations':
                   {'horizontalflip': {'p': 0.5}
                    }
                   }
                  }
        train_augs, val_augs = build_pipeline(config)

        assert isinstance(train_augs.transforms[0], OneOf)
        assert isinstance(train_augs.transforms[1], HorizontalFlip)
        assert train_augs.p == 0.75
        assert isinstance(train_augs.transforms[0].transforms[0], Normalize)
        assert isinstance(train_augs.transforms[0].transforms[1], Rotate)
        assert train_augs.transforms[0].p == 0.5
        assert train_augs.transforms[0].transforms[0].p == 1
        assert not train_augs.transforms[0].transforms[0].always_apply
        assert train_augs.transforms[0].transforms[1].limit == (-45, 45)
        assert train_augs.transforms[0].transforms[1].border_mode == 'reflect'
        assert isinstance(val_augs, Compose)
        assert isinstance(val_augs.transforms[0], HorizontalFlip)

        # test running the pipeline
        arr = np.array([[3, 5, 3],
                        [5, 8, 10],
                        [1, 3, 8]])
        train_result = train_augs(image=arr)
        # make sure this gave a 2D numpy array out in a dict with key 'image'
        assert len(train_result['image'].shape) == 2
