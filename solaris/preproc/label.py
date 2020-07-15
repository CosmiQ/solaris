import geopandas as gpd
import pandas as pd
import shapely.geometry
import shapely.wkt

from .pipesegment import PipeSegment, LoadSegment, MergeSegment
from ..vector.polygon import convert_poly_coords


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
    def __init__(self, pathstring, geom_col='geometry', projection=None):
        super().__init__()
        self.pathstring = pathstring
        self.geom_col = geom_col
        self.projection = projection
    def load(self):
        if self.pathstring.lower()[-4:] == '.csv':
            df = pd.read_csv(self.pathstring)
            geometry = df.apply(lambda row:
                shapely.wkt.loads(row[self.geom_col]), axis=1)
            df.drop(columns=[self.geom_col])
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            if self.projection is not None:
                gdf.crs = 'epsg:' + str(self.projection)
            return gdf
        else:
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


class ShowDataFrame(PipeSegment):
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


#class DataFrameToMask(PipeSegment):
#    """
#    Given a GeoPandas GeoDataFrame and an Image-class image,
#    convert the DataFrame to the corresponding Boolean mask
#    """
#    pass
#
#
#class MaskToDataFrame(PipeSegment):
#    """
#    Given a boolean mask, convert it to a GeoPandas GeoDataFrame of polygons.
#    """
#    pass


class DataFramePixelCoords(PipeSegment):
    """
    Given a GeoPandas GeoDataFrame, converts between georeferenced
    coordinates and pixel coordinates.  Assumes image has affine geotransform.
    """
    def __init__(self, inverse=False, reverse_order=False, *args, **kwargs):
        super().__init__()
        self.inverse = inverse
        self.reverse_order = reverse_order
        self.args = args
        self.kwargs = kwargs
    def transform(self, pin):
        if not self.reverse_order:
            gdf = pin[0]
            img = pin[1]
        else:
            gdf = pin[1]
            img = pin[0]
        affine = img.metadata['geotransform']
        gdf = gdf.copy()
        newgeoms = gdf.apply(lambda row: convert_poly_coords(
            row.geometry, affine_obj=affine, inverse=self.inverse,
            *self.args, **self.kwargs
        ), axis=1)
        gdf.geometry = newgeoms
        return gdf


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
