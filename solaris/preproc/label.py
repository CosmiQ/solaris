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
