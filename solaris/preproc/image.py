import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from .pipesegment import PipeSegment, LoadSegment, MergeSegment


class Image:
    def __init__(self, data, name='image', metadata={}):
        self.name = name
        self.metadata = metadata
        self.data = data
    def __str__(self):
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
    def process(self):
        return self.load_from_disk(self.pathstring, self.name, self.verbose)
    def load_from_disk(self, pathstring, name=None, verbose=False):
        #Use GDAL to open image file
        dataset = gdal.Open(pathstring)
        if dataset is None:
            raise Exception('! Image file not found.')
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
        #Create an Image-class object, and return it
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
    def process(self):
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
    def process(self):
        if type(self.imageinput) is Image:
            return self.load_from_memory(self.imageinput, self.name, self.verbose)
        elif type(self.imageinput) is str:
            return self.load_from_disk(self.imageinput, self.name, self.verbose)
        else:
            raise Exception('! Invalid input type in LoadImage.')


class SaveImage(PipeSegment):
    """
    Save an image to disk using GDAL.
    """
    def __init__(self, pathstring, driver='GTiff', return_image=True,
                 save_projection=True, save_metadata=True):
        super().__init__()
        self.pathstring = pathstring
        self.return_image = return_image
        self.save_projection = save_projection
        self.save_metadata = save_metadata
        self.driver = driver
    def transform(self, pin):
        #Save image to disk
        driver = gdal.GetDriverByName(self.driver)
        dataset = driver.Create(self.pathstring, pin.data.shape[2], pin.data.shape[1], pin.data.shape[0], gdal.GDT_Float32)
        for band in range(pin.data.shape[0]):
            dataset.GetRasterBand(band+1).WriteArray(pin.data[band, :, :])
        if self.save_projection:
            if len(pin.metadata['projection_ref']) >= len(pin.metadata['gcp_projection']):
                dataset.SetGeoTransform(pin.metadata['geotransform'])
                dataset.SetProjection(pin.metadata['projection_ref'])
            else:
                dataset.SetGCPs(pin.metadata['gcps'],
                                pin.metadata['gcp_projection'])
        if self.save_metadata:
            dataset.SetMetadata(pin.metadata['meta'])
        dataset.FlushCache()
        #Optionally return image
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
    def __init__(self, show_text=True, show_image=True, cmap='gray',
                 vmin=None, vmax=None):
        super().__init__()
        self.show_text = show_text
        self.show_image = show_image
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
    def transform(self, pin):
        if self.show_text:
            print(pin)
        if self.show_image:
            pyplot_format = np.squeeze(np.moveaxis(pin.data, 0, -1))
            plt.imshow(pyplot_format, cmap=self.cmap,
                       vmin=self.vmin, vmax=self.vmax)
            plt.show()
        return pin


class ImageStats(PipeSegment):
    """
    Calculate descriptive statististics about an image
    """
    def __init__(self, print_desc=True, print_props=True, return_image=True, return_props=False):
        super().__init__()
        self.print_desc = print_desc
        self.print_props = print_props
        self.return_image = return_image
        self.return_props = return_props
    def transform(self, pin):
        if self.print_desc:
            print(pin)
            print()
        props = pd.DataFrame({
            'min': np.nanmin(pin.data, (1,2)),
            'max': np.nanmax(pin.data, (1,2)),
            'mean': np.mean(pin.data, (1,2)),
            'median': np.median(pin.data, (1,2)),
            'pos': np.count_nonzero(np.nan_to_num(pin.data, nan=-1.)>0, (1,2)),
            'zero': np.count_nonzero(pin.data==0, (1,2)),
            'neg': np.count_nonzero(np.nan_to_num(pin.data, nan=1.)<0, (1,2)),
            'nan': np.count_nonzero(np.isnan(pin.data), (1,2)),
        })
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


class MergeToList(PipeSegment):
    """
    Flatten a nested tuple into a single list.  This makes the output
    of repeated merges (i.e., additions) easier to work with.
    """
    def transform(self, pin):
        pout = []
        def recurse(t):
            for item in t:
                if type(item) == tuple:
                    recurse(item)
                else:
                    pout.append(item)
        recurse(pin)
        return pout


class MergeToStack(PipeSegment):
    """
    Given an iterable or nested tuple of equal-sized images, combine
    all of their bands into a single image.
    """
    def __init__(self, master=0):
        super().__init__()
        self.master = master
    def transform(self, pin):
        #Make list of all the input bands
        pmid = (pin * MergeToList())()
        datalist = [imageobj.data for imageobj in pmid]
        #Create output image, using name and metadata from designated source
        pout = Image(None, pmid[self.master].name, pmid[self.master].metadata)
        pout.data = np.concatenate(datalist, axis=0)
        return pout


class SelectItem(PipeSegment):
    """
    Given an iterable, return one of its items.  This is useful when passing
    a list of items into, or out of, a custom class.
    """
    def __init__(self, index=0):
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
