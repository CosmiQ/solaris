class TestImports(object):

    def test_imports(self):
        import torch  # workaround for TLS error
        from solaris.utils import core, geo, config, tile, cli
        from solaris import data
        from solaris.vector import polygon, graph, mask
        from solaris.tile import raster_tile, vector_tile
        from solaris.raster import image
        from solaris.nets import callbacks, datagen, infer, model_io, losses
        from solaris.nets import train, transform, zoo
        from solaris.eval import base, iou, challenges, pixel
        import solaris
