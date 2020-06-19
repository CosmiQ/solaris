import gdal
import json
import math
import numpy as np
import os
import osr
import scipy.signal
import uuid
import warnings
import xml.etree.ElementTree as ET

from .pipesegment import PipeSegment, LoadSegment, MergeSegment
from .image import Image
from . import image


class BandMath(PipeSegment):
    """
    Modify the array holding an image's pixel values,
    using a user-supplied function.
    """
    def __init__(self, function, master=0):
        super().__init__()
        self.function = function
        self.master = master
    def transform(self, pin):
        if isinstance(pin, tuple):
            pin = (pin * image.MergeToStack(self.master))()
        data = self.function(pin.data)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        return Image(data, pin.name, pin.metadata)


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


class InPhase(PipeSegment):
    """
    Get in-phase (real) component of complex-valued data
    """
    def transform(self, pin):
        return Image(np.real(pin.data), pin.name, pin.metadata)


class Quadrature(PipeSegment):
    """
    Get quadrature (imaginary) component of complex-valued data
    """
    def transform(self, pin):
        return Image(np.imag(pin.data), pin.name, pin.metadata)


class Phase(PipeSegment):
    """
    Return the phase of the input image
    """
    def transform(self, pin):
        return Image(np.angle(pin.data), pin.name, pin.metadata)


class Conjugate(PipeSegment):
    """
    Return complex conjugate of the input image
    """
    def transform(self, pin):
        return Image(np.conj(pin.data), pin.name, pin.metadata)


class MultiplyConjugate(PipeSegment):
    """
    Given an iterable of two images, multiply the first 
    by the complex conjugate of the second.
    """
    def __init__(self, master=0):
        super().__init__()
        self.master = master
    def transform(self, pin):
        return Image(
            pin[0].data * np.conj(pin[1].data),
            pin[self.master].name,
            pin[self.master].metadata
        )


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


class MultilookComplex(Multilook):
    """
    Like 'Multilook', but supports complex input
    """
    def transform(self, pin):
        mkwargs = {'kernel_size':self.kernel_size, 'method':self.method}
        pout = (pin
               * (InPhase() * Multilook(**mkwargs) * image.Scale(1.+0.j)
                  + Quadrature() * Multilook(**mkwargs) * image.Scale(1.j))
               * image.MergeToSum()
        )()
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
        pout.name = pin.name
        if pin.data.dtype in (bool, np.dtype('bool')):
            pout.data = pout.data.astype('bool')
        driver = gdal.GetDriverByName(drivername)
        driver.Delete(srcpath)
        driver.Delete(dstpath)
        return pout


class DecompositionPauli(PipeSegment):
    """
    Compute the Pauli decomposition of quad-pol SAR data.
    Note: Convention is alpha-->blue, beta-->red, gamma-->green
    """
    def __init__(self, hh_band=0, vv_band=1, xx_band=2):
        super().__init__()
        self.hh_band = hh_band
        self.vv_band = vv_band
        self.xx_band = xx_band
    def transform(self, pin):
        hh = pin.data[self.hh_band]
        vv = pin.data[self.vv_band]
        xx = pin.data[self.xx_band]
        alpha2 = 0.5 * np.square(np.absolute(hh + vv))
        beta2 = 0.5 * np.square(np.absolute(hh - vv))
        gamma2 = 2 * np.square(np.absolute(xx))
        alpha2 = np.expand_dims(alpha2, axis=0)
        beta2 = np.expand_dims(beta2, axis=0)
        gamma2 = np.expand_dims(gamma2, axis=0)
        pout = Image(np.concatenate((alpha2, beta2, gamma2), axis=0),
                     pin.name,
                     pin.metadata)
        return pout


class DecompositionFreemanDurden(PipeSegment):
    """
    Compute the three-component polarimetric decomposition of quad-pol SAR data
    proposed by Freeman and Durden.
    Note: Convention is Ps-->blue, Pd-->red, Pv-->green
    """
    def __init__(self, hh_band=0, vv_band=1, xx_band=2, kernel_size=5):
        super().__init__()
        self.hh_band = hh_band
        self.vv_band = vv_band
        self.xx_band = xx_band
        self.kernel_size = kernel_size
    def transform(self, pin):
        # Scattering matrix terms
        hh = pin * image.SelectBands(self.hh_band)
        vv = pin * image.SelectBands(self.vv_band)
        xx = pin * image.SelectBands(self.xx_band)
        # Covariance matrix terms
        C11 = hh * Intensity()
        C22 = vv * Intensity()
        C33 = xx * Intensity()
        C12 = (hh + vv) * MultiplyConjugate()
        mkwargs = {'kernel_size':self.kernel_size, 'method':'avg'}
        C11 = C11 * Multilook(**mkwargs)
        C22 = C22 * Multilook(**mkwargs)
        C33 = C33 * Multilook(**mkwargs)
        C12 = C12 * MultilookComplex(**mkwargs)
        # Volume amplitude, and volume-subtracted matrix terms
        fv = C33 * image.Scale(1.5)
        c11 = (C11 + fv) * BandMath(lambda x: x[0] - x[1])
        c22 = (C22 + fv) * BandMath(lambda x: x[0] - x[1])
        c12 = (C12 + fv) * BandMath(lambda x: x[0] - x[1] / 3.)
        c12 = (c11 + c22 + c12) * BandMath(lambda x: np.where(x[0]*x[1]
            < np.square(np.abs(x[2])), np.sqrt(x[0]*x[1]) \
            * x[2]/np.abs(x[2]), x[2]))#
        # Surface and dihedral amplitudes
        surfacedominates = c12 * BandMath(lambda x: np.real(x) >= 0)
        term1 = (c11 + c22 + c12 * InPhase() + c12 * Quadrature()
            + surfacedominates) * BandMath(lambda x:
            (x[0]*x[1] - (x[2])**2 - (x[3])**2) /
            (x[0] + x[1] + 2*x[2]*np.where(x[4], 1, -1)))
        term1 = term1 * Amplitude()#
        term2 = (c22 + term1) * BandMath(lambda x: x[0] - x[1])
        term2 = term2 * Amplitude()#
        term3 = (term1 + term2 + c12 * InPhase() + c12 * Quadrature()
            + surfacedominates) * BandMath(lambda x:
            (x[2] + np.where(x[4], 1, -1) * x[0] + x[3] * 1.j) / x[1])
        fs = ((term2 + surfacedominates) * image.SetMask(flag=0) + (term1
            + surfacedominates * image.InvertMask()) * image.SetMask(flag=0)) \
            * image.MergeToSum()
        fd = ((term1 + surfacedominates) * image.SetMask(flag=0) + (term2
            + surfacedominates * image.InvertMask()) * image.SetMask(flag=0)) \
            * image.MergeToSum()
        alpha = (surfacedominates * image.Scale(-1.) + (term3
            + surfacedominates * image.InvertMask()) * image.SetMask(flag=0)) \
            * BandMath(lambda x: x[0] + x[1])
        beta = ((term3 + surfacedominates) * image.SetMask(flag=0) \
            + surfacedominates * image.InvertMask() * image.Scale(1.)) \
            * BandMath(lambda x: x[0] + x[1])
        # Power
        Ps = (fs + beta * Intensity()) * BandMath(lambda x: x[0] * (1. + x[1]))
        Pd = (fd + alpha *Intensity()) * BandMath(lambda x: x[0] * (1. + x[1]))
        Pv = fv
        Pmask = (c11 + c22) * BandMath(lambda x: np.logical_and(
            x[0]==0, x[1]==0)) * image.InvertMask()
        Ps = (Ps + Pmask) * image.SetMask(flag=0)#
        Pd = (Pd + Pmask) * image.SetMask(flag=0)#
        Pstack = (Ps + Pd + Pv) * image.MergeToStack()
        return Pstack()


class DecompositionHAlpha(PipeSegment):
    """
    Compute H-Alpha (Entropy-alpha) dual-polarization decomposition
    """
    def __init__(self, band0=0, band1=1, kernel_size=5):
        super().__init__()
        self.band0 = band0
        self.band1 = band1
        self.kernel_size = kernel_size
    def transform(self, pin):
        mkwargs = {'kernel_size':self.kernel_size, 'method':'avg'}
        image0 = pin * image.SelectBands(self.band0)
        image1 = pin * image.SelectBands(self.band1)
        # Coherence matrix terms
        c00 = image0 * Intensity() * Multilook(**mkwargs)
        c11 = image1 * Intensity() * Multilook(**mkwargs)
        c01 = (image0 + image1) * MultiplyConjugate() \
            * MultilookComplex(**mkwargs)
        c01sq = c01 * Intensity()
        # Calculate eigenvalues and some eigenvector terms (assumes c01 != 0)
        # tr=trace; det=determinant; l1,l2=eigenvalues; v..=eigenvector terms
        tr = (c00 + c11) * BandMath(lambda x: x[0] + x[1])
        det = (c00 + c11 + c01sq) * BandMath(lambda x: x[0]*x[1] - x[2])
        l1 = (tr + det) * BandMath(lambda x:
                                   0.5*x[0] + np.sqrt(0.25*x[0]**2-x[1]))
        l2 = (tr + det) * BandMath(lambda x:
                                   0.5*x[0] - np.sqrt(0.25*x[0]**2-x[1]))
        absv11 = (c00 + c01 + l1) * BandMath(lambda x: np.abs(x[1]) / np.sqrt(np.abs(x[1])**2 + np.abs(x[2] - x[0])**2))
        absv12 = (c00 + c01 + l2) * BandMath(lambda x: np.abs(x[1]) / np.sqrt(np.abs(x[1])**2 + np.abs(x[2] - x[0])**2))
        # Calculate entropy (H) and alpha
        P1 = (l1 + l2) * BandMath(lambda x: x[0] / (x[0] + x[1]))
        P2 = (l1 + l2) * BandMath(lambda x: x[1] / (x[0] + x[1]))
        H = (P1 + P2) * BandMath(lambda x: -x[0] * np.log(x[0])
                                 - x[1] * np.log(x[1]))
        alpha = (P1 + P2 + absv11 + absv12) * BandMath(lambda x: x[0] * np.arccos(x[2]) + x[1] * np.arccos(x[3]))
        outputs = (H + alpha) * image.MergeToStack()
        return outputs()


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
                 spacing=150, row_spacing=None, col_spacing=None):
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
                    grid.data[1, ri, ci],  #longitude
                    grid.data[0, ri, ci],  #latitude
                    grid.data[2, ri, ci],  #altitude
                    ci, ri  #pixel=column=x, line=row=y
                ))
        if len(gcps) > 10922:
            warnings.warn('! Many GCPs generated in CapellaGridToGCPs.')
        pout.metadata['gcps'] = gcps
        return pout


class CapellaGridToPolygon(PipeSegment):
    """
    Given a Capella grid file, return a GeoJSON string indicating its boundary.
    'step' is number of pixels between each recorded point.
    """
    def __init__(self, step=100, flags=False):
        super().__init__()
        self.step = step
        self.flags = flags
    def transform(self, pin):
        # Get indices of selected points along the edges of the grid file
        nrows = pin.data.shape[1]
        ncols = pin.data.shape[2]
        step = self.step
        allri = []
        allci = []
        cornerri = []
        cornerci = []
        for edge in range(4):
            if edge == 0:
                ri = list(range(0, nrows - 1, step))
                ci = [0] * len(ri)
            elif edge == 1:
                ci = list(range(0, ncols - 1, step))
                ri = [nrows - 1] * len(ci)
            elif edge == 2:
                ri = list(range(nrows - 1, 0, -step))
                ci = [ncols - 1] * len(ri)
            elif edge == 3:
                ci = list(range(ncols - 1, 0, -step))
                ri = [0] * len(ci)
            allri.extend(ri)
            allci.extend(ci)
            cornerri.append(ri[0])
            cornerci.append(ci[0])
        allri.append(allri[0])
        allci.append(allci[0])
        # Get latitude/longitude values, and ensure they're counterclockwise
        lats = [pin.data[0, ri, ci] for ri, ci in zip(allri, allci)]
        lons = [pin.data[1, ri, ci] for ri, ci in zip(allri, allci)]
        cornerlats = [pin.data[0, ri, ci] for ri,ci in zip(cornerri, cornerci)]
        cornerlons = [pin.data[1, ri, ci] for ri,ci in zip(cornerri, cornerci)]
        vi = (cornerlons[1] - cornerlons[0], cornerlats[1] - cornerlats[0])
        vf = (cornerlons[0] - cornerlons[3], cornerlats[0] - cornerlats[3])
        counterclockwise = vf[0] * vi[1] - vf[1] * vi[0] > 0
        if not counterclockwise:
            lats.reverse()
            lons.reverse()
        northlooking = cornerlats[3] > cornerlats[0]
        eastlooking = cornerlons[3] > cornerlons[0]
        flags = (counterclockwise, northlooking, eastlooking)
        # Write latitudes & longitudes of the selected points to a JSON string
        jsonstring = '{\n' \
                     '"type": "FeatureCollection",\n' \
                     '"name": "region_' + pin.name + '",\n' \
                     '"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::4326" } },\n' \
                     '"features": [\n' \
                     '{ "type": "Feature", "properties": { }, "geometry": { "type": "Polygon", "coordinates": [ [ '
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            if i>0:
                jsonstring += ', '
            jsonstring += '[ ' + str(lon) + ', ' + str(lat) + ', 0.0 ]'
        jsonstring += '] ] } }\n]\n}'
        if self.flags:
            return (jsonstring,) + flags
        else:
            return jsonstring


class CapellaGridCommonWindow(PipeSegment):
    """
    Given an iterable of Capella grid files with equal orientations and pixel
    sizes but translational offsets, find the overlapping region and return
    its array indices for each grid file. Optionally, also return the subpixel
    offset of each grid file needed for exact alignment.
    """
    def __init__(self, master=0, subpixel=True):
        super().__init__()
        self.master = master
        self.subpixel = subpixel
    def transform(self, pin):
        # Find the pixel in each grid that's closest to center of master grid.
        # 'x' and 'y' are the latitude and longitude bands of the grid files,
        # and (refx, refy) is the (lat, lon) of that center.
        m = self.master
        l = len(pin)
        order = [m] + list(range(m)) + list(range(m+1, l))
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
                # Get latitude and longitude of the reference point
                refx = x[localrefs[index]]
                refy = y[localrefs[index]]
            else:
                # Find pixel closest to reference point
                localrefs[index] = self.courseoffset(x, y, refx, refy)
                # Find subpixel offset of reference point
                fineoffsets[index] = self.fineoffset(x, y, refx, refy,
                                                     localrefs[index][0],
                                                     localrefs[index][1])
            # Find how far from the reference pixel each grid extends
            # Convention is [left, bottom, right, top]
            extents[index] = [
                localrefs[index][1],
                x.shape[0] - localrefs[index][0] - 1,
                x.shape[1] - localrefs[index][1] - 1,
                localrefs[index][0]
            ]
            if step==0:
                minextents = extents[index].copy()
            else:
                for i in range(4):
                    if extents[index][i] < minextents[i]:
                        minextents[i] = extents[index][i]
        # Calculate col_min, row_max, col_max, row_min of overlapping window
        for step, index in enumerate(order):
            windows[index] = [
                localrefs[index][1] - minextents[0],
                localrefs[index][0] + minextents[1],
                localrefs[index][1] + minextents[2],
                localrefs[index][0] - minextents[3]
            ]
        # Optionally return subpixel offsets
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


class TerraSARXScaleFactor(PipeSegment):
    """
    Calibrate TerraSAR-X complex data using the scale factor in the
    accompanying xml file.
    """
    def __init__(self, reverse_order=False):
        super().__init__()
        self.reverse_order = reverse_order
    def transform(self, pin):
        if not self.reverse_order:
            img = pin[0]
            info = pin[1]
        else:
            img = pin[1]
            info = pin[0]
        root = ET.fromstring(info)
        scale_factor = float(list(root.iter('calFactor'))[0].text)
        return Image(math.sqrt(scale_factor) * img.data, img.name, img.metadata)


class TerraSARXGeorefToGCPs(PipeSegment):
    """
    Generate ground control points (GCPs) from a TerraSAR-X GEOREF.xml file
    and save them in a corresponding image's metadata.  Input is a tuple
    with the image in the 0 position and the georef file in the 1 position.
    Output is the image with modified metadata.
    """
    def __init__(self, reverse_order=False):
        super().__init__()
        self.reverse_order = reverse_order
    def transform(self, pin):
        # Define output image
        if not self.reverse_order:
            img = pin[0]
            georef = pin[1]
        else:
            img = pin[1]
            georef = pin[0]
        pout = Image(img.data, img.name, img.metadata.copy())
        # Set GCP values
        gcps = []
        root = ET.fromstring(georef)
        gcpentries = root.findall('./geolocationGrid/gridPoint')
        for gcpentry in gcpentries:
            gcps.append(gdal.GCP(
                float(gcpentry.find('lon').text),     #longitude
                float(gcpentry.find('lat').text),     #latitude
                float(gcpentry.find('height').text),  #altitude
                float(gcpentry.find('col').text),     #pixel=column=x
                float(gcpentry.find('row').text)      #line=row=y
            ))
        pout.metadata['gcps'] = gcps
        # Set GCP projection
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(4326)
        pout.metadata['gcp_projection'] = crs.ExportToWkt()
        return pout
