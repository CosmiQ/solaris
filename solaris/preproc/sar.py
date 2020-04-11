import numpy as np
import scipy.signal

from .pipesegment import PipeSegment, LoadSegment, MergeSegment
from .image import Image
#from . import image


class Multilook(PipeSegment):
    def __init__(self, kernel_size=5, method='avg'):
        super().__init__()
        self.kernel_size = kernel_size
        self.method = method
    def transform(self, pin):
        pout = Image(np.zeros(pin.data.shape, dtype=pin.data.dtype),
                     pin.name, pin.metadata)
        for i in range(pin.data.shape[0]):
            if self.method =='avg':
                pout.data[i, :, :] = scipy.signal.convolve(
                    pin.data[i, :, :],
                    np.ones((self.kernel_size,
                             self.kernel_size)) / (self.kernel_size**2),
                    mode='same')
            elif self.method == 'med':
                pout.data[i, :, :] = scipy.ndimage.filters.median_filter(
                    pin.data[i, :, :],
                    self.kernel_size,
                    mode='reflect')
            elif self.method == 'max':
                pout.data[i, :, :] = scipy.ndimage.filters.maximum_filter(
                    pin.data[i, :, :],
                    self.kernel_size,
                    mode='reflect')
            elif self.method == 'non':
                pass
            else:
                raise Exception('! Invalid method in Multilook.')
        return pout


class UnequalMultilook(PipeSegment):
    def __init__(self, usize=5, vsize=5):
        super().__init__()
        self.usize = usize
        self.vsize = vsize
    def transform(self, pin):
        raise Exception('! Not written yet')
