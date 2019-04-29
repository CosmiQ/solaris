class TestImports(object):
    def test_imports(self):
        from solaris.utils import core, geo, config, tile, cli
        from solaris import data
        from solaris.vector import polygon, graph, mask
        from solaris.tile import main, vector_utils
        from solaris.raster import image
        from solaris.nets.models import callbacks, datagen, infer, io, losses
        from solaris.nets.models import setup, train, transform, zoo
        from solaris.eval import baseeval, evalfunctions
        from solaris.eval.challenges import off_nadir_dataset
        from solaris.eval.challenges import spacenet_buildings2_dataset
        import solaris
