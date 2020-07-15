import os
from .. import weights_dir
from .xdxd_sn4 import XDXD_SpaceNet4_UNetVGG16
from .selim_sef_sn4 import SelimSef_SpaceNet4_ResNet34UNet
from .selim_sef_sn4 import SelimSef_SpaceNet4_DenseNet121UNet
from .selim_sef_sn4 import SelimSef_SpaceNet4_DenseNet161UNet
from .woj_sn6 import Woj_SpaceNet6_UNet # rram

model_dict = {
    'xdxd_spacenet4': {
        'weight_path': os.path.join(weights_dir,
                                    'xdxd_spacenet4_solaris_weights.pth'),
        'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/xdxd_spacenet4_solaris_weights.pth',
        'arch': XDXD_SpaceNet4_UNetVGG16
        },
    'selimsef_spacenet4_resnet34unet': {
        'weight_path': os.path.join(
            weights_dir, 'selimsef_spacenet4_resnet34unet_solaris_weights.pth'
            ),
        'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_resnet34unet_solaris_weights.pth',
        'arch': SelimSef_SpaceNet4_ResNet34UNet
        },
    'selimsef_spacenet4_densenet121unet': {
        'weight_path': os.path.join(
            weights_dir, 'selimsef_spacenet4_densenet121unet_solaris_weights.pth'
        ),
        'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_densenet121unet_solaris_weights.pth',
        'arch': SelimSef_SpaceNet4_DenseNet121UNet
        },
    'selimsef_spacenet4_densenet161unet': {
        'weight_path': os.path.join(
            weights_dir, 'selimsef_spacenet4_densenet161unet_solaris_weights.pth'
        ),
        'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_densenet161unet_solaris_weights.pth',
        'arch': SelimSef_SpaceNet4_DenseNet161UNet
        },
    # rram below
    'xdxd_spacenet_test': {
        'weight_path': os.path.join(weights_dir,
                                    'xdxd_spacenet4_solaris_weights.pth'),
        'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/xdxd_spacenet4_solaris_weights.pth',
        'arch': XDXD_SpaceNet4_UNetVGG16
        }
    # 'test_name': {
    #     'weight_path': os.path.join(weights_dir,
    #                                 'woj_spacenet6_solaris_weights.pth'),
    #     # 'weight_url': 'empty for now -- NEED TO FILL THIS WITH AWS-HOSTED WEIGHT FILE LATER',
    #     # 'weight_url': 'https://drive.google.com/file/d/11sspczYR_6GIF9kWvBTYCkp7EU5N-Dk5/view?usp=sharing',
    #     'weight_url': 'https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_densenet161unet_solaris_weights.pth', # temp check
    #     'arch': Woj_SpaceNet6_UNet
    #     }
    }
