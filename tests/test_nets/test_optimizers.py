from solaris.nets.optimizers import get_optimizer
from tensorflow import keras
import torch


class TestGetOptimizer(object):
    """Test the sol.nets.optimizers.get_optimizer() function."""

    def test_get_optimizers(self):
        config = {'training':
                  {'optimizer': 'sgd'}}
        keras_sgd = get_optimizer('keras', config)
        assert keras_sgd == keras.optimizers.SGD
        torch_sgd = get_optimizer('torch', config)
        assert torch_sgd == torch.optim.SGD
