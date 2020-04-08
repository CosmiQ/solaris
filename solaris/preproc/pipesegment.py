from multiprocessing import Pool
def _parallel_compute_function(x):
    return (x[0])(*(x[1]))()


class PipeSegment:
    def __init__(self):
        self.feeder = None
        self.procout = None
        self.procstart = False
        self.procfinish = False
    def __call__(self):
        if self.procstart and not self.procfinish:
            raise Exception('(!) Circular dependency in workflow.')
        if not self.procfinish:
            self.procstart = True
            self.procout = self.process()
            self.procfinish = True
        return self.procout
    def process(self):
        return self.transform(self.feeder())
    def transform(self, pin):
        return pin
    def reset(self):
        self.procout = None
        self.procstart = False
        self.procfinish = False
        self.feeder.reset()
    def selfstring(self, offset=0):
        return ' '*2*offset + type(self).__name__ + '\n'
    def __str__(self, offset=0):
        return self.selfstring(offset) + self.feeder.__str__(offset+1)
    def attach(self, ps):
        if self.feeder is None:
            self.feeder = ps
        else:
            self.feeder.attach(ps)
    def __mul__(self, other):
        other.attach(self)
        return other
    def __or__(self, other):
        other.attach(self)
        return other
    def __add__(self, other):
        return MergeSegment(self, other)
    def __rmul__(self, other):
        return LoadSegment(other) * self
    def __ror__(self, other):
        return LoadSegment(other) * self
    @classmethod
    def parallel(cls, inputargs, processes=None):
        allinputs = list(zip([cls]*len(inputargs),inputargs))
        with Pool(processes) as pool:
            return pool.map(_parallel_compute_function, allinputs)


class LoadSegment(PipeSegment):
    def __init__(self, source=None):
        super().__init__()
        self.source = source
    def process(self):
        return self.source
    def reset(self):
        self.procout = None
        self.procstart = False
        self.procfinish = False
    def __str__(self, offset=0):
        return self.selfstring(offset)
    def attach(self, ps):
        pass


class MergeSegment(PipeSegment):
    def __init__(self, feeder1, feeder2):
        super().__init__()
        self.feeder1 = feeder1
        self.feeder2 = feeder2
    def process(self):
        return (self.feeder1(), self.feeder2())
    def reset(self):
        self.procout = None
        self.procstart = False
        self.procfinish = False
        self.feeder1.reset()
        self.feeder2.reset()
    def __str__(self, offset=0):
        return self.selfstring(offset) \
            + self.feeder1.__str__(offset+1) \
            + self.feeder2.__str__(offset+1)
    def attach(self, ps):
        if self.feeder1 is None:
            self.feeder1 = ps
        else:
            self.feeder1.attach(ps)
        if self.feeder2 is None:
            self.feeder2 = ps
        else:
            self.feeder2.attach(ps)
