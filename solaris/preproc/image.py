import gdal
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal_array
import pandas as pd
import uuid
import warnings

from .pipesegment import PipeSegment, LoadSegment, MergeSegment


class Image:
    def __init__(self, data, name='image', metadata={}):
        self.name = name
        self.metadata = metadata
        self.set_data(data)
    def set_data(self, data):
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        self.data = data
    def __str__(self):
        if self.data.ndim < 3:
            raise Exception('! Image data has too few dimensions.')
        metastring = str(self.metadata)
        if len(metastring)>400:
            metastring = metastring[:360] + '...'
        return '%s: %d bands, %dx%d, %s, %s' % (self.name,
                                                *np.shape(self.data),
                                                str(self.data.dtype),
                                                metastring)


class Identity(PipeSegment):
    """
    This class is an alias for the PipeSegment base class to emphasize
    its role as the identity element.
    """
    pass


class LoadImageFromDisk(LoadSegment):
    """
    Load an image from the file system using GDAL, so it can be fed
    into subsequent PipeSegments.
    """
    def __init__(self, pathstring, name=None, verbose=False):
        super().__init__()
        self.pathstring = pathstring
        self.name = name
        self.verbose = verbose
    def load(self):
        return self.load_from_disk(self.pathstring, self.name, self.verbose)
    def load_from_disk(self, pathstring, name=None, verbose=False):
        # Use GDAL to open image file
        dataset = gdal.Open(pathstring)
        if dataset is None:
            raise Exception('! Image file ' + pathstring + ' not found.')
        data = dataset.ReadAsArray()
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        metadata = {
            'geotransform': dataset.GetGeoTransform(),
            'projection_ref': dataset.GetProjectionRef(),
            'gcps': dataset.GetGCPs(),
            'gcp_projection': dataset.GetGCPProjection(),
            'meta': dataset.GetMetadata()
        }
        metadata['band_meta'] = [dataset.GetRasterBand(band).GetMetadata()
                                 for band in range(1, dataset.RasterCount+1)]
        if name is None:
            name = os.path.splitext(os.path.split(pathstring)[1])[0]
        dataset = None
        # Create an Image-class object, and return it
        imageobj = Image(data, name, metadata)
        if verbose:
            print(imageobj)
        return imageobj


class LoadImageFromMemory(LoadSegment):
    """
    Points to an 'Image'-class image so it can be fed
    into subsequent PipeSegments.
    """
    def __init__(self, imageobj, name=None, verbose=False):
        super().__init__()
        self.imageobj = imageobj
        self.name = name
        self.verbose = verbose
    def load(self):
        return self.load_from_memory(self.imageobj, self.name, self.verbose)
    def load_from_memory(self, imageobj, name=None, verbose=False):
        if type(imageobj) is not Image:
            raise Exception('! Invalid input type in LoadImageFromMemory.')
        if name is not None:
            imageobj.name = name
        if verbose:
            print(imageobj)
        return(imageobj)


class LoadImage(LoadImageFromDisk, LoadImageFromMemory):
    """
    Makes an image available to subsequent PipeSegments, whether the image
    is in the filesystem (in which case 'imageinput' is the path) or an
    Image-class variable (in which case 'imageinput' is the variable name).
    """
    def __init__(self, imageinput, name=None, verbose=False):
        PipeSegment.__init__(self)
        self.imageinput = imageinput
        self.name = name
        self.verbose = verbose
    def load(self):
        if type(self.imageinput) is Image:
            return self.load_from_memory(self.imageinput, self.name, self.verbose)
        elif type(self.imageinput) in (str, np.str_):
            return self.load_from_disk(self.imageinput, self.name, self.verbose)
        else:
            raise Exception('! Invalid input type in LoadImage.')


class SaveImage(PipeSegment):
    """
    Save an image to disk using GDAL.
    """
    def __init__(self, pathstring, driver='GTiff', return_image=True,
                 save_projection=True, save_metadata=True, no_data_value=None):
        super().__init__()
        self.pathstring = pathstring
        self.driver = driver
        self.return_image = return_image
        self.save_projection = save_projection
        self.save_metadata = save_metadata
        self.no_data_value = no_data_value
    def transform(self, pin):
        # Save image to disk
        driver = gdal.GetDriverByName(self.driver)
        datatype = gdal_array.NumericTypeCodeToGDALTypeCode(pin.data.dtype)
        if datatype is None:
            if pin.data.dtype in (bool, np.dtype('bool')):
                datatype = gdal.GDT_Byte
            else:
                warnings.warn('! SaveImage did not find data type match; saving as float.')
                datatype = gdal.GDT_Float32
        dataset = driver.Create(self.pathstring, pin.data.shape[2], pin.data.shape[1], pin.data.shape[0], datatype)
        for band in range(pin.data.shape[0]):
            bandptr = dataset.GetRasterBand(band+1)
            bandptr.WriteArray(pin.data[band, :, :])
            if isinstance(self.no_data_value, str) \
               and self.no_data_value.lower() == 'nan':
                bandptr.SetNoDataValue(math.nan)
            elif self.no_data_value is not None:
                bandptr.SetNoDataValue(self.no_data_value)
            bandptr.FlushCache()
        if self.save_projection:
            #First determine which projection system, if any, is used
            proj_lens = [0, 0]
            proj_keys = ['projection_ref', 'gcp_projection']
            for i, proj_key in enumerate(proj_keys):
                if proj_key in pin.metadata.keys():
                    proj_lens[i] = len(pin.metadata[proj_key])
            if proj_lens[0] > 0 and proj_lens[0] >= proj_lens[1]:
                dataset.SetGeoTransform(pin.metadata['geotransform'])
                dataset.SetProjection(pin.metadata['projection_ref'])
            elif proj_lens[1] > 0 and proj_lens[1] >= proj_lens[0]:
                dataset.SetGCPs(pin.metadata['gcps'],
                                pin.metadata['gcp_projection'])
        if self.save_metadata and 'meta' in pin.metadata.keys():
            dataset.SetMetadata(pin.metadata['meta'])
        dataset.FlushCache()
        # Optionally return image
        if self.driver.lower() == 'mem':
            return dataset
        elif self.return_image:
            return pin
        else:
            return None


class ShowImage(PipeSegment):
    """
    Display an image using matplotlib.
    """
    def __init__(self, show_text=False, show_image=True, cmap='gray',
                 vmin=None, vmax=None, bands=None, caption=None,
                 width=None, height=None):
        super().__init__()
        self.show_text = show_text
        self.show_image = show_image
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.bands = bands
        self.caption = caption
        self.width = width
        self.height = height
    def transform(self, pin):
        if self.caption is not None:
            print(self.caption)
        if self.show_text:
            print(pin)
        if self.show_image:
            # Select data, and format it for matplotlib
            if self.bands is None:
                image_formatted = pin.data
            else:
                image_formatted = pin.data[self.bands]
            pyplot_formatted = np.squeeze(np.moveaxis(image_formatted, 0, -1))
            if np.ndim(pyplot_formatted)==3 and self.vmin is not None and self.vmax is not None:
                pyplot_formatted = np.clip((pyplot_formatted - self.vmin) / (self.vmax - self.vmin), 0., 1.)
            # Select image size
            if self.height is None and self.width is None:
                rc = {}
            elif self.height is None and self.width is not None:
                rc = {'figure.figsize': [self.width, self.width]}
            elif self.height is not None and self.width is None:
                rc = {'figure.figsize': [self.height, self.height]}
            else:
                rc = {'figure.figsize': [self.width, self.height]}
            # Show image
            with plt.rc_context(rc):
                plt.imshow(pyplot_formatted, cmap=self.cmap,
                           vmin=self.vmin, vmax=self.vmax)
                plt.show()
        return pin


class ImageStats(PipeSegment):
    """
    Calculate descriptive statististics about an image
    """
    def __init__(self, print_desc=True, print_props=True, return_image=True, return_props=False, median=True, caption=None):
        super().__init__()
        self.print_desc = print_desc
        self.print_props = print_props
        self.return_image = return_image
        self.return_props = return_props
        self.median = median
        self.caption = caption
    def transform(self, pin):
        if self.caption is not None:
            print(self.caption)
        if self.print_desc:
            print(pin)
            print()
        props = pd.DataFrame({
            'min': np.nanmin(pin.data, (1,2)),
            'max': np.nanmax(pin.data, (1,2)),
            'mean': np.nanmean(pin.data, (1,2)),
            'std': np.nanstd(pin.data, (1,2)),
            'pos': np.count_nonzero(np.nan_to_num(pin.data, nan=-1.)>0, (1,2)),
            'zero': np.count_nonzero(pin.data==0, (1,2)),
            'neg': np.count_nonzero(np.nan_to_num(pin.data, nan=1.)<0, (1,2)),
            'nan': np.count_nonzero(np.isnan(pin.data), (1,2)),
        })
        if self.median:
            props.insert(3, 'median', np.nanmedian(pin.data, (1,2)))
        if self.print_props:
            print(props)
            print()
        if self.return_image and self.return_props:
            return (pin, props)
        elif self.return_image:
            return pin
        elif self.return_props:
            return props
        else:
            return None


class MergeToStack(PipeSegment):
    """
    Given an iterable of equal-sized images, combine
    all of their bands into a single image.
    """
    def __init__(self, master=0):
        super().__init__()
        self.master = master
    def transform(self, pin):
        # Make list of all the input bands
        datalist = [imageobj.data for imageobj in pin]
        # Create output image, using name and metadata from designated source
        pout = Image(None, pin[self.master].name, pin[self.master].metadata)
        pout.data = np.concatenate(datalist, axis=0)
        return pout


class MergeToSum(PipeSegment):
    """
    Combine an iterable of images by summing the corresponding bands.
    Assumes that images are of equal size and have equal numbers of bands.
    """
    def __init__(self, master=0):
        super().__init__()
        self.master = master
    def transform(self, pin):
        total = pin[self.master].data.copy()
        for i in range(len(pin)):
            if not i == self.master:
                total += pin[i].data
        return Image(total, pin[self.master].name, pin[self.master].metadata)


class MergeToProduct(PipeSegment):
    """
    Combine an iterable of images by multiplying the corresponding bands.
    Assumes that images are of equal size and have equal numbers of bands.
    """
    def __init__(self, master=0):
        super().__init__()
        self.master = master
    def transform(self, pin):
        product = pin[self.master].data.copy()
        for i in range(len(pin)):
            if not i == self.master:
                product *= pin[i].data
        return Image(product, pin[self.master].name, pin[self.master].metadata)


class SelectItem(PipeSegment):
    """
    Given an iterable, return one of its items.  This is useful when passing
    a list of items into, or out of, a custom class.
    """
    def __init__(self, index=0):
        super().__init__()
        self.index = index
    def transform(self, pin):
        return pin[self.index]


class SelectBands(PipeSegment):
    """
    Reorganize the bands in an image.  This class can be used to
    select, delete, duplicate, or reorder bands.
    """
    def __init__(self, bands=[0]):
        super().__init__()
        if not hasattr(bands, '__iter__'):
            bands = [bands]
        self.bands = bands
    def transform(self, pin):
        return Image(pin.data[self.bands, :, :], pin.name, pin.metadata)


class Bounds(PipeSegment):
    """
    Output the boundary coordinates [xmin, ymin, xmax, ymax] of an image.
    Note: Requires the image to have an affine geotransform, not GCPs.
    Note: Only works for a north-up image without rotation or shearing
    """
    def transform(self, pin):
        gt = pin.metadata['geotransform']
        numrows = pin.data.shape[1]
        numcols = pin.data.shape[2]
        bounds = [gt[0], gt[3] + gt[5]*numrows, gt[0] + gt[1]*numcols, gt[3]]
        return bounds


class Scale(PipeSegment):
    """
    Scale data by a multiplicative factor.
    """
    def __init__(self, factor=1.):
        super().__init__()
        self.factor = factor
    def transform(self, pin):
        return Image(self.factor * pin.data, pin.name, pin.metadata)


class Crop(PipeSegment):
    """
    Crop image based on either pixel coordinates or georeferenced coordinates.
    'bounds' is a list specifying the edges: [left, bottom, right, top]
    """
    def __init__(self, bounds, mode='pixel'):
        super().__init__()
        self.bounds = bounds
        self.mode = mode
    def transform(self, pin):
        row_min = self.bounds[3]
        row_max = self.bounds[1]
        col_min = self.bounds[0]
        col_max = self.bounds[2]
        if self.mode in ['pixel', 'p', 0]:
            srcWin = [col_min, row_min,
                      col_max - col_min + 1, row_max - row_min + 1]
            projWin = None
        elif self.mode in ['geo', 'g', 1]:
            srcWin = None
            projWin = [col_min, row_min, col_max, row_max]
        else:
            raise Exception('! Invalid mode in Crop')
        drivername = 'GTiff'
        srcpath = '/vsimem/crop_input_' + str(uuid.uuid4()) + '.tif'
        dstpath = '/vsimem/crop_output_' + str(uuid.uuid4()) + '.tif'
        (pin * SaveImage(srcpath, driver=drivername))()
        gdal.Translate(dstpath, srcpath, srcWin=srcWin, projWin=projWin)
        pout = LoadImage(dstpath)()
        pout.name = pin.name
        if pin.data.dtype in (bool, np.dtype('bool')):
            pout.data = pout.data.astype('bool')
        driver = gdal.GetDriverByName(drivername)
        driver.Delete(srcpath)
        driver.Delete(dstpath)
        return pout


class CropVariable(Crop):
    """
    Like 'Crop', but window coordinates are accepted from another
    PipeSegment at runtime instead of via initialization arguments.
    """
    def __init__(self, mode='pixel'):
        PipeSegment.__init__(self)
        self.mode = mode
    def transform(self, pin):
        imagetocrop = pin[0]
        self.bounds = pin[1]
        return super().transform(imagetocrop)


class Resize(PipeSegment):
    """
    Resize an image to the requested number of pixels
    """
    def __init__(self, rows, cols):
        super().__init__()
        self.rows = rows
        self.cols = cols
    def transform(self, pin):
        return self.resize(pin, self.rows, self.cols)
    def resize(self, pin, rows, cols):
        drivername = 'GTiff'
        srcpath = '/vsimem/resize_input_' + str(uuid.uuid4()) + '.tif'
        dstpath = '/vsimem/resize_output_' + str(uuid.uuid4()) + '.tif'
        (pin * SaveImage(srcpath, driver=drivername))()
        gdal.Translate(dstpath, srcpath, width=cols, height=rows)
        pout = LoadImage(dstpath)()
        pout.name = pin.name
        if pin.data.dtype in (bool, np.dtype('bool')):
            pout.data = pout.data.astype('bool')
        driver = gdal.GetDriverByName(drivername)
        driver.Delete(srcpath)
        driver.Delete(dstpath)
        return pout


class GetMask(PipeSegment):
    """
    Extract a Boolean mask from an image band.  NaN is assumed to be the
    mask value, unless otherwise specified.
    """
    def __init__(self, band=0, flag='nan'):
        super().__init__()
        self.band = band
        self.flag = flag
    def transform(self, pin):
        if self.flag == 'nan':
            data = np.expand_dims(np.invert(np.isnan(pin.data[self.band])), axis=0)
        else:
            data = np.expand_dims(pin.data[self.band]==self.flag, axis=0)
        return Image(data, pin.name, pin.metadata)


class SetMask(PipeSegment):
    """
    Given an image and a mask, apply the mask to the image.
    More specifically, set the image's pixel value to NaN
    (or other specified value) for every pixel where the 
    mask value is False.
    """
    def __init__(self, flag=math.nan, band=None, reverse_order=False):
        super().__init__()
        self.flag = flag
        self.band = band
        self.reverse_order = reverse_order
    def transform(self, pin):
        if not self.reverse_order:
            img = pin[0]
            mask = pin[1]
        else:
            img = pin[1]
            mask = pin[0]
        mark = np.invert(np.squeeze(mask.data))
        data = np.copy(img.data)
        if self.band is None:
            data[:, mark] = self.flag
        else:
            data[self.band, mark] = self.flag
        return Image(data, img.name, img.metadata)


class InvertMask(PipeSegment):
    """
    Sets all True values in a mask to False and vice versa.
    """
    def transform(self, pin):
        return Image(np.invert(pin.data), pin.name, pin.metadata)
