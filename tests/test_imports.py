class TestImports(object):

    def test_imports(self):
        from solaris.utils import core, geo, tile, cli
        from solaris import data
        from solaris.vector import polygon, graph, mask
        from solaris.tile import raster_tile, vector_tile
        from solaris.raster import image
        from solaris.eval import base, iou, pixel
        import solaris
