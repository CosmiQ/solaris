import colorsys
import numpy as np

from .pipesegment import PipeSegment, LoadSegment, MergeSegment
from .image import Image
from . import image


class RGBToHSL(PipeSegment):
    """
    Convert an RGB image into an HSL (hue/saturation/lightness) image
    using colorsys.
    """
    def __init__(self, rband=0, gband=1, bband=2, rgbmax=255.):
        super().__init__()
        self.rband = rband
        self.gband = gband
        self.bband = bband
        self.rgbmax = rgbmax
    def transform(self, pin):
        m = self.rgbmax
        convertarray = np.vectorize(colorsys.rgb_to_hls)
        outbands = convertarray(np.clip(pin.data[self.rband] / m, 0, 1),
                                np.clip(pin.data[self.gband] / m, 0, 1),
                                np.clip(pin.data[self.bband] / m, 0, 1))
        pout = Image(None, pin.name, pin.metadata)
        pout.data = np.stack((outbands[0], outbands[2], outbands[1]), axis=0)
        return pout


class HSLToRGB(PipeSegment):
    """
    Convert an HSL (hue/saturation/lightness) image into an RGB image
    using colorsys.
    """
    def __init__(self, hband=0, sband=1, lband=2, rgbmax=255.):
        super().__init__()
        self.hband = hband
        self.sband = sband
        self.lband = lband
        self.rgbmax = rgbmax
    def transform(self, pin):
        convertarray = np.vectorize(colorsys.hls_to_rgb)
        outbands = convertarray(np.clip(pin.data[self.hband], 0, 1),
                                np.clip(pin.data[self.lband], 0, 1),
                                np.clip(pin.data[self.sband], 0, 1))
        pout = Image(None, pin.name, pin.metadata)
        pout.data = self.rgbmax * np.stack(outbands, axis=0)
        return pout


class RGBToHSV(PipeSegment):
    """
    Convert an RGB image into an HSV (hue/saturation/value) image
    using colorsys.
    """
    def __init__(self, rband=0, gband=1, bband=2, rgbmax=255.):
        super().__init__()
        self.rband = rband
        self.gband = gband
        self.bband = bband
        self.rgbmax = rgbmax
    def transform(self, pin):
        m = self.rgbmax
        convertarray = np.vectorize(colorsys.rgb_to_hsv)
        outbands = convertarray(np.clip(pin.data[self.rband] / m, 0, 1),
                                np.clip(pin.data[self.gband] / m, 0, 1),
                                np.clip(pin.data[self.bband] / m, 0, 1))
        pout = Image(None, pin.name, pin.metadata)
        pout.data = np.stack(outbands, axis=0)
        return pout


class HSVToRGB(PipeSegment):
    """
    Convert an HSV (hue/saturation/value) image into an RGB image
    using colorsys.
    """
    def __init__(self, hband=0, sband=1, vband=2, rgbmax=255.):
        super().__init__()
        self.hband = hband
        self.sband = sband
        self.vband = vband
        self.rgbmax = rgbmax
    def transform(self, pin):
        convertarray = np.vectorize(colorsys.hsv_to_rgb)
        outbands = convertarray(np.clip(pin.data[self.hband], 0, 1),
                                np.clip(pin.data[self.sband], 0, 1),
                                np.clip(pin.data[self.vband], 0, 1))
        pout = Image(None, pin.name, pin.metadata)
        pout.data = self.rgbmax * np.stack(outbands, axis=0)
        return pout
