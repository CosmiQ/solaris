import numpy as np
import scipy.signal

from .pipesegment import PipeSegment, LoadSegment, MergeSegment
from .image import Image
#from . import image


class Multilook(PipeSegment):
    """
    Multilook filter to reduce speckle in SAR magnitude imagery
    Note: Set kernel_size to a tuple to vary it by direction.
    """
    def __init__(self, kernel_size=5, method='avg'):
        super().__init__()
        self.kernel_size = kernel_size
        self.method = method
    def transform(self, pin):
        if self.method == 'avg':
            filter = scipy.ndimage.filters.uniform_filter
        elif self.method == 'med':
            filter = scipy.ndimage.filters.median_filter
        elif self.method == 'max':
            filter = scipy.ndimage.filters.maximum_filter
        else:
            raise Exception('! Invalid method in Multilook.')
        pout = Image(np.zeros(pin.data.shape, dtype=pin.data.dtype),
                     pin.name, pin.metadata)
        for i in range(pin.data.shape[0]):
            pout.data[i, :, :] = filter(
                pin.data[i, :, :],
                size=self.kernel_size,
                mode='reflect')
        return pout
