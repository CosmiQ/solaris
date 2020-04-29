import gdal
import json
import math
import numpy as np
import os
import scipy.signal
import uuid

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
    Orthorectify an image using its ground control points (GCPs) with GDAL
    """
    def __init__(self, projection=3857, algorithm='lanczos',
                 row_res=1., col_res=1.):
        super().__init__()
        self.projection = projection
        self.algorithm = algorithm
        self.row_res = row_res
        self.col_res = col_res
    def transform(self, pin):
        drivername = 'GTiff'
        srcpath = '/vsimem/orthorectify_input_' + str(uuid.uuid4()) + '.tif'
        dstpath = '/vsimem/orthorectify_output_' + str(uuid.uuid4()) + '.tif'
        (pin * image.SaveImage(srcpath, driver=drivername))()
        gdal.Warp(dstpath, srcpath,
                  dstSRS='epsg:' + str(self.projection),
                  resampleAlg=self.algorithm,
                  xRes=self.row_res, yRes=self.col_res,
                  dstNodata=math.nan)
        pout = image.LoadImage(dstpath)()
        driver = gdal.GetDriverByName(drivername)
        driver.Delete(srcpath)
        driver.Delete(dstpath)
        return pout


class CapellaScaleFactor(PipeSegment):
    """
    Calibrate Capella single-look complex data (or amplitude thereof)
    using the scale factor in the metadata
    """
    def transform(self, pin):
        tiffjson = json.loads(pin.metadata['meta']['TIFFTAG_IMAGEDESCRIPTION'])
        scale_factor = tiffjson['collect']['image']['scale_factor']
        return Image(scale_factor * pin.data, pin.name, pin.metadata)


class CapellaGridToGCPs(PipeSegment):
    """
    Generate ground control points (GCPs) from a Capella grid file
    and save them in a corresponding image's metadata.  Input is a tuple
    with the image in the 0 position and the grid in the 1 position.
    Output is the image with modified metadata.  Spacing between points
    is in pixels.
    """
    def __init__(self, reverse_order=False, row_range=None, col_range=None,
                 spacing=100, row_spacing=None, col_spacing=None):
        super().__init__()
        self.reverse_order = reverse_order
        self.row_range = row_range
        self.col_range = col_range
        self.spacing = spacing
        self.row_spacing = row_spacing
        self.col_spacing = col_spacing
    def transform(self, pin):
        if not self.reverse_order:
            img = pin[0]
            grid = pin[1]
        else:
            img = pin[1]
            grid = pin[0]
        pout = Image(img.data, img.name, img.metadata.copy())
        if self.row_range is None:
            rlo = 0
            rhi = img.data.shape[1] - 1
        else:
            rlo = self.row_range[0]
            rhi = self.row_range[1]
        if self.col_range is None:
            clo = 0
            chi = img.data.shape[2] - 1
        else:
            clo = self.col_range[0]
            chi = self.col_range[1]
        rspace = self.spacing
        cspace = self.spacing
        if self.row_spacing is not None:
            rspace = self.row_spacing
        if self.col_spacing is not None:
            cspace = self.col_spacing
        gcps = []
        for ri in range(rlo, rhi + 1, rspace):
            for ci in range(clo, chi + 1, cspace):
                gcps.append(gdal.GCP(
                    grid.data[1, ri, ci], #y, longitude
                    grid.data[0, ri, ci], #x, latitude
                    grid.data[2, ri, ci], #z, height
                    ci, ri
                ))
        pout.metadata['gcps'] = gcps
        return pout


class CapellaGridCommonWindow(PipeSegment):
    """
    Given an iterable of Capella grid files with equal orientations and
    pixel sizes but translational offsets, find the overlapping region
    and return its indices for each grid file. Also return the subpixel
    offset of each grid file needed for exact alignment.
    """
    def __init__(self, master=0, subpixel=True):
        super().__init__()
        self.master = master
        self.subpixel = subpixel
    def transform(self, pin):
        print('start')
        #Find the point in each grid that's closest to center of master grid
        #Below, 'r' refers to row and 'c' refers to column
        m = self.master
        l = len(pin)
        order = [m] + list(range(l)[:m]) + list(range(l)[m+1:])
        localrefs = [] * len(pin)
        fineoffests = [] * len(pin)
        extents = [] * len(pin)
        windows = [] * len(pin)
        for step, index in enum(order):
            r = pin[index].data[0]
            c = pin[index].data[1]
            if step==0:
                localrefs[index] = (int(0.5 * r.shape[0]),
                                    int(0.5 * r.shape[1]))
                fineoffsets[index] = (0., 0.)
                refr = r[localrefs[index]]
                refc = c[localrefs[index]]
            else:
                localrefs[index] = self.courseoffset(r, c, refr, refc)
                fineoffsets[index] = self.fineoffset(r, c, refr, refc,
                                                     localrefs[index][0],
                                                     localrefs[index][1])


        print('end')
        return windows
    def courseoffset(latgrid, longrid, lattarget, lontarget):
        pass
    def fineoffset(latgrid, longrid, lattarget, lontarget, uidx, vidx):
        pass
