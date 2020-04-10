import gdal
import matplotlib.pyplot as plt
import numpy as np
import os

from .pipesegment import PipeSegment, LoadSegment, MergeSegment


class Image:
    def __init__(self, data, name='image', metadata={}):
        self.name = name
        self.metadata = metadata
        self.data = data
    def __str__(self):
        return '%s: %d bands, %dx%d, %s' % (self.name,
                                            *np.shape(self.data),
                                            self.metadata)


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
        metadata = {'projection': dataset.GetGCPProjection(),
                    'meta':dataset.GetMetadata()}
        if name is None:
            name = os.path.splitext(os.path.split(pathstring)[1])[0]
        dataset = None
        #Create an Image-class object, and return it
        image = Image(data, name, metadata)
        if verbose:
            print(image)
        return image


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
    def __init__(self, pathstring, return_image=True):
        super().__init__()
        self.pathstring = pathstring
        self.return_image = return_image
    def transform(self, pin):
        #Save image to disk
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(self.pathstring, pin.data.shape[2], pin.data.shape[1], pin.data.shape[0], gdal.GDT_Float32)
        for band in range(pin.data.shape[0]):
            dataset.GetRasterBand(band+1).WriteArray(pin.data[band, :, :])
        dataset.SetProjection(pin.metadata['projection'])
        dataset.FlushCache()
        #Optionally return image
        if self.return_image:
            return pin
        else:
            return None


class ShowImage(PipeSegment):
    """
    Display an RGB image using matplotlib.
    """
    def __init__(self, show_text=True, show_image=True):
        super().__init__()
        self.show_text = show_text
        self.show_image = show_image
    def transform(self, pin):
        if self.show_text:
            print(pin)
        if self.show_image:
            pyplot_order = np.moveaxis(pin.data, 0, -1).astype(int)
            plt.imshow(pyplot_order)
            plt.show()
        return pin


class MergeIntoList(PipeSegment):
    """
    Flatten a nested tuple into a single list.  This makes the output
    of repeated merges easier to work with.
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


class MergeIntoStack(PipeSegment):
    """
    Given an iterable or nested tuple of equal-sized images, combine
    all of their bands into a single image.
    """
    def transform(self, pin, master=0):
        #Make list of all the input bands
        pmid = (pin * MergeIntoList())()
        datalist = [image.data for image in pmid]
        #Create output image, using name and metadata from designated source
        pout = Image(None, pmid[master].name, pmid[master].metadata)
        pout.data = np.concatenate(datalist, axis=0)
        return pout
