import gdal
import json
import math
import numpy as np
import scipy.signal

from .pipesegment import PipeSegment, LoadSegment, MergeSegment
from .image import Image
from . import image


class Amplitude(PipeSegment):
    """
    Convert complex image to amplitude, by taking the magnitude of each pixel
    """
    def transform(self, pin):
        return Image(np.absolute(pin.data), pin.name, pin.metadata)


class Intensity(PipeSegment):
    """
    Convert amplitude (or complex values) to intensity, by squaring each pixel
    """
    def transform(self, pin):
        pout = Image(None, pin.name, pin.metadata)
        if not np.iscomplexobj(pin.data):
            pout.data = np.square(pin.data)
        else:
            pout.data = np.square(np.absolute(pin.data))
        return pout


class Decibels(PipeSegment):
    """
    Express quantity in decibels
    The 'flag' argument indicates how to handle nonpositive inputs:
    'min' outputs the log of the image's smallest positive value,
    'nan' outputs NaN, and any other value is used as the flag value itself.
    """
    def __init__(self, flag='min'):
        super().__init__()
        self.flag = flag
    def transform(self, pin):
        pout = Image(None, pin.name, pin.metadata)
        if isinstance(self.flag, str) and self.flag.lower() == 'min':
            flagval = 10. * np.log10((pin.data)[pin.data>0].min())
        elif isinstance(self.flag, str) and self.flag.lower() == 'nan':
            flagval = math.nan
        else:
            flagval = self.flag / 10.
        pout.data = 10. * np.log10(
            pin.data,
            out=np.full(np.shape(pin.data), flagval).astype(pin.data.dtype),
            where=pin.data>0
        )
        return pout


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


class Orthorectify(PipeSegment):
    """
    Orthorectify an image using its ground control points (GCPs)
    """
    def __init__(self, projection=3857, algorithm='lanczos', row_res=1., col_res=1.):
        super().__init__()
        self.projection = projection
        self.algorithm = algorithm
        self.row_res = row_res
        self.col_res = col_res
    def transform(self, pin):
        file1 = '/home/sol/src/sar/preproc/file1.tif'
        file2 = '/home/sol/src/sar/preproc/file2.tif'
        (pin * image.SaveImage(file1))()
        dataset = gdal.Open(file1)
        gdal.Warp(file2, dataset, dstSRS='epsg:' + str(self.projection), resampleAlg=self.algorithm, xRes=self.row_res, yRes=self.col_res)
        pout = image.LoadImage(file2)()
        return pout
        """
        dataset = (pin * image.SaveImage('', driver='MEM'))()
        #dataset = (pin * image.SaveImage('/home/sol/src/preproc/sar0.tif'))()
        print('before warp')
        gdal.Warp('/home/sol/src/sar/preproc/sar.tif', dataset, dstSRS='epsg:' + str(self.projection), resampleAlg=self.algorithm, xRes=self.row_res, yRes=self.col_res)
        print('after warp')
        #print(type(var2))
        return pin
        """


class CapellaGridFiletoGCPs(PipeSegment):
    """
    Generate ground control points (GCPs) from a Capella grid file
    and save them in a corresponding image's metadata
    """
    pass


class CapellaScaleFactor(PipeSegment):
    """
    Calibrate Capella single-look complex data (or amplitude thereof)
    using the scale factor in the metadata
    """
    def transform(self, pin):
        tiffjson = json.loads(pin.metadata['meta']['TIFFTAG_IMAGEDESCRIPTION'])
        scale_factor = tiffjson['collect']['image']['scale_factor']
        return Image(scale_factor * pin.data, pin.name, pin.metadata)
