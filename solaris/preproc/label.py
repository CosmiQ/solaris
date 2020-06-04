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
    Load a GeoPandas dataframe from a file.
    """
    def __init__(self, pathstring):
        super().__init__()
        self.pathstring = pathstring
    def process(self):
        return gpd.read_file(self.pathstring)


class ExplodeDataFrame(PipeSegment):
    """
    Given a GeoPandas DataFrame, break multi-part geometries
    into multiple lines.
    """
    def transform(self, pin):
        return pin.explode().reset_index()


class DataFrameToString(PipeSegment):
    """
    Given a GeoPandas DataFrame, convert it into a GeoJSON string.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    def transform(self, pin):
        return pin.to_json(**(self.kwargs))
