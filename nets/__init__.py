import os

weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'weights')

from . import callbacks, datagen, infer, losses, metrics, model_io
from . import optimizers, train, transform, zoo


if not os.path.isdir(weights_dir):
    os.mkdir(weights_dir)
