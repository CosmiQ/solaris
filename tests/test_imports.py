import sys


class TestImports(object):
    def test_imports(self):
        from cw_geodata.utils import core, geo
        from cw_geodata.raster_image import image
        from cw_geodata.vector_label import polygon, graph, mask
