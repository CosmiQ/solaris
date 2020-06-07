import geopandas as gpd
import shapely.geometry

from .pipesegment import PipeSegment, LoadSegment, MergeSegment


class LoadString(LoadSegment):
    """
    Load a string from a file.
    """
    def __init__(self, pathstring):
        super().__init__()
        self.pathstring = pathstring
    def load(self):
        infile = open(self.pathstring, 'r')
        content = infile.read()
        infile.close()
        return content


class SaveString(PipeSegment):
    """
    Write a string to a file.
    """
    def __init__(self, pathstring, append=False):
        super().__init__()
        self.pathstring = pathstring
        self.append = append
    def transform(self, pin):
        mode = 'a' if self.append else 'w'
        outfile = open(self.pathstring, mode)
        outfile.write(str(pin))
        outfile.close()
        return pin


class ShowString(PipeSegment):
    """
    Print a string to the screen.
    """
    def transform(self, pin):
        print(pin)
        return pin


class LoadDataFrame(LoadSegment):
    """
    Load a GeoPandas GeoDataFrame from a file.
    """
    def __init__(self, pathstring):
        super().__init__()
        self.pathstring = pathstring
    def load(self):
        return gpd.read_file(self.pathstring)


class SaveDataFrame(PipeSegment):
    """
    Save a GeoPandas GeoDataFrame to disk.
    """
    def __init__(self, pathstring, driver='GeoJSON'):
        super().__init__()
        self.pathstring = pathstring
        self.driver = driver
    def transform(self, pin):
        pin.to_file(self.pathstring, driver=self.driver)
        return pin


class ShowString(PipeSegment):
    """
    Print a GeoPandas GeoDataFrame to the screen.
    """
    def transform(self, pin):
        print(pin)
        return pin


class ReprojectDataFrame(PipeSegment):
    """
    Reproject a GeoPandas GeoDataFrame.
    """
    def __init__(self, projection=3857):
        super().__init__()
        self.projection = projection
    def transform(self, pin):
        return pin.to_crs('epsg:' + str(self.projection))


class ExplodeDataFrame(PipeSegment):
    """
    Given a GeoPandas GeoDataFrame, break multi-part geometries
    into multiple lines.
    """
    def transform(self, pin):
        return pin.explode().reset_index()


class IntersectDataFrames(PipeSegment):
    """
    Given an iterable of GeoPandas GeoDataFrames, returns their intersection
    """
    def __init__(self, master=0):
        super().__init__()
        self.master = master
    def transform(self, pin):
        result = pin[self.master]
        for i, gdf in enumerate(pin):
            if not i==self.master:
                result = gpd.overlay(result, gdf)
                result.crs = pin[self.master].crs
        return result


class DataFrameToString(PipeSegment):
    """
    Given a GeoPandas GeoDataFrame, convert it into a GeoJSON string.
    Caveat emptor: This follows the GeoJSON 2016 standard, which does
    not include any coordinate reference system information.
    """
    def __init__(self, crs=True, **kwargs):
        super().__init__()
        self.crs = crs
        self.kwargs = kwargs
    def transform(self, pin):
        geojson = pin.to_json(**(self.kwargs))
        if self.crs:
            geojson = '{"type": "FeatureCollection", "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::' \
                      + str(pin.crs.to_epsg()) \
                      + '" } }, ' \
                      + geojson[30:]
        return geojson


class BoundsToDataFrame(PipeSegment):
    """
    Given a set of tile bounds [left, lower, right, upper],
    convert it to a GeoPandas GeoDataFrame.  Note: User must
    specify projection, since a simple set of bounds doesn't
    include that.
    """
    def __init__(self, projection=None):
        super().__init__()
        self.projection = projection
    def transform(self, pin):
        gdf = gpd.GeoDataFrame()
        if self.projection is not None:
            gdf.crs = 'epsg:' + str(self.projection)
        gdf.loc[0, 'geometry'] = shapely.geometry.box(*pin)
        return gdf
