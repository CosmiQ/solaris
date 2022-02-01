# flake8: noqa: F401
class TestImports(object):
    def test_imports(self):
        import solaris
        from solaris import data
        from solaris.eval import base, iou, pixel
        from solaris.raster import image
        from solaris.tile import raster_tile, vector_tile
        from solaris.utils import cli, core, geo, tile
        from solaris.vector import graph, mask, polygon
