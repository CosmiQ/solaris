import os
from .. import weights_dir
from .xdxd_sn4 import XDXD_SpaceNet4_UNetVGG16
from .selim_sef_sn4 import SelimSef_SpaceNet4_ResNet34UNet

model_dict = {
    'xdxd_spacenet4': {
        'weight_path': os.path.join(weights_dir,
                                    'xdxd_spacenet4_solaris_weights.pth'),
        'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/xdxd_spacenet4_solaris_weights.pth',
        'arch': XDXD_SpaceNet4_UNetVGG16}}
