import geopandas as gpd

from .pipesegment import PipeSegment, LoadSegment, MergeSegment


class LoadString(LoadSegment):
    """
    Load a string from a file.
    """
    def __init__(self, pathstring):
        super().__init__()
        self.pathstring = pathstring
    def process(self):
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
    def process(self):
        return gpd.read_file(self.pathstring)


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
        return result


class DataFrameToString(PipeSegment):
    """
    Given a GeoPandas GeoDataFrame, convert it into a GeoJSON string.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    def transform(self, pin):
        return pin.to_json(**(self.kwargs))
