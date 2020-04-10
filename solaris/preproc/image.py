import gdal
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


class LoadImageFromDisk(LoadSegment):
    def __init__(self, pathstring, name=None, verbose=False):
        super().__init__()
        self.load_from_disk(pathstring, name, verbose)
    def load_from_disk(self, pathstring, name=None, verbose=False):
        #Use GDAL to open image file
        dataset = gdal.Open(pathstring)
        if dataset is None:
            raise Exception('! Image file not found.')
        data = dataset.ReadAsArray()
        metadata = {'projection': dataset.GetGCPProjection()}
        if name is None:
            name = os.path.splitext(os.path.split(pathstring)[1])[0]
        #Create an Image-class object, and set it as the source
        self.source = Image(data, name, metadata)
        if verbose:
            print(self.source)


class LoadImageFromMemory(LoadSegment):
    def __init__(self, imageinput, name=None, verbose=False):
        super().__init__()
        self.load_from_memory(imageinput, name, verbose)
    def load_from_memory(self, imageinput, name=None, verbose=False):
        if type(imageinput) is Image:
            self.source = imageinput
        else:
            raise Exception('! Invalid input type in LoadImageFromMemory.')
        if verbose:
            print(self.source)


class LoadImage(LoadImageFromDisk, LoadImageFromMemory):
    def __init__(self, imageinput, name=None, verbose=False):
        LoadSegment.__init__(self)
        if type(imageinput) is Image:
            self.load_from_memory(imageinput, name, verbose)
        elif type(imageinput) is str:
            self.load_from_disk(imageinput, name, verbose)
        else:
            raise Exception('! Invalid input type in LoadImage.')
