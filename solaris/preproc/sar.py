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


class Crop(PipeSegment):
    """
    Crop an image based on either pixel coordinates
    or georeferenced coordinates
    """
    def __init__(self, xmin, ymin, xmax, ymax, mode='pixel'):
        super().__init__()
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.mode = mode
    def transform(self, pin):
        return self.crop(pin, self.xmin, self.ymin, self.xmax, self.ymax, self.mode)
    def crop(self, pin, xmin, ymin, xmax, ymax, mode):
        if mode in ['pixel', 'p', 0]:
            srcWin = [xmin, ymin, xmax-xmin, ymax-ymin]
            projWin = None
        elif mode in ['geo', 'g', 1]:
            srcWin = None
            projWin = [xmin, ymin, xmax, ymax]
        else:
            raise Exception('! Invalid mode in Crop')
        drivername = 'GTiff'
        srcpath = '/vsimem/crop_input_' + str(uuid.uuid4()) + '.tif'
        dstpath = '/vsimem/crop_output_' + str(uuid.uuid4()) + '.tif'
        (pin * image.SaveImage(srcpath, driver=drivername))()
        gdal.Translate(dstpath, srcpath, srcWin=srcWin, projWin=projWin)
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
        pin = (pin * image.MergeToList())()
        print('start')
        #Find the pixel in each grid that's closest to center of master grid
        #'x' and 'y' are the latitude and longitude bands of the grid files,
        #and (refx, refy) is the (lat, lon) of that center
        m = self.master
        l = len(pin)
        order = [m] + list(range(l)[:m]) + list(range(l)[m+1:])
        localrefs = [[]] * len(pin)
        fineoffsets = [[]] * len(pin)
        extents = [[]] * len(pin)
        windows = [[]] * len(pin)
        for step, index in enumerate(order):
            x = pin[index].data[0]
            y = pin[index].data[1]
            if step==0:
                localrefs[index] = (int(0.5 * x.shape[0]),
                                    int(0.5 * x.shape[1]))
                fineoffsets[index] = (0., 0.)
                refx = x[localrefs[index]]
                refy = y[localrefs[index]]
            else:
                #Find pixel closest to reference point
                localrefs[index] = self.courseoffset(x, y, refx, refy)
                #Find subpixel offset of reference point
                fineoffsets[index] = self.fineoffset(x, y, refx, refy,
                                                     localrefs[index][0],
                                                     localrefs[index][1])
            #Find how far from the reference pixel each grid extends
            extents[index] = [
                localrefs[index][0],
                localrefs[index][1],
                x.shape[0] - localrefs[index][0] - 1,
                x.shape[1] - localrefs[index][1] - 1
            ]
            if step==0:
                minextents = extents[index].copy()
            else:
                for i in range(4):
                    if extents[index][i] < minextents[i]:
                        minextents[i] = extents[index][i]
        for step, index in enumerate(order):
            windows[index] = [
                localrefs[index][0] - minextents[0],
                localrefs[index][1] - minextents[1],
                localrefs[index][0] + minextents[2],
                localrefs[index][1] + minextents[3]
            ]
        #Optionally return subpixel offsets
        if self.subpixel:
            finearray = np.array(fineoffsets)
            windows.append(finearray)
        return windows

    def haversine(self, lat1, lon1, lat2, lon2, rad=False, radius=6.371E6):
        """
        Haversine formula for distance between two points given their
        latitude and longitude, assuming a spherical earth.
        """
        if not rad:
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            lat2 = np.radians(lat2)
            lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * radius * np.arcsin(np.sqrt(a))

    def courseoffset(self, latgrid, longrid, lattarget, lontarget):
        """
        Given a latitude/longitude pair, find the closest point in
        a grid of almost-regularly-spaced latitude/longitude pairs.
        """
        bound0 = np.shape(latgrid)[0] - 1
        bound1 = np.shape(latgrid)[1] - 1
        pos0 = int(bound0 / 2)
        pos1 = int(bound1 / 2)
        def score(pos0, pos1):
            return self.haversine(latgrid[pos0, pos1], longrid[pos0, pos1], lattarget, lontarget)
        while True:
            scorenow = score(pos0, pos1)
            if pos0>0 and score(pos0-1, pos1)<scorenow:
                pos0 -= 1
            elif pos0<bound0 and score(pos0+1, pos1)<scorenow:
                pos0 += 1
            elif pos1>0 and score(pos0, pos1-1)<scorenow:
                pos1 -= 1
            elif pos1<bound1 and score(pos0, pos1+1)<scorenow:
                pos1 += 1
            else:
                return (pos0, pos1)

    def fineoffset(self, latgrid, longrid, lattarget, lontarget, uidx, vidx):
        """
        Given grids of almost-equally-spaced latitude and longitude, and an 
        exact latitude-longitude pair to aim for, and indices of the (ideally)
        nearest point to that lat-long target, returns a first-order estimate
        of the target offset, in pixels, relative to the specified point.
        """
        mlat = lattarget - latgrid[uidx, vidx]
        mlon = lontarget - longrid[uidx, vidx]
        ulat = latgrid[uidx+1, vidx] - latgrid[uidx, vidx]
        ulon = longrid[uidx+1, vidx] - longrid[uidx, vidx]
        vlat = latgrid[uidx, vidx+1] - latgrid[uidx, vidx]
        vlon = longrid[uidx, vidx+1] - longrid[uidx, vidx]
        uoffset = (mlat*ulat + mlon*ulon) / (ulat**2 + ulon**2)
        voffset = (mlat*vlat + mlon*vlon) / (vlat**2 + vlon**2)
        return uoffset, voffset
