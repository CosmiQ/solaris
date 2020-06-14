from multiprocessing import Pool
def _parallel_compute_function(x):
    return (x[0])(*(x[1]))()


class PipeSegment:
    def __init__(self):
        self.feeder = None
        self.procout = None
        self.procstart = False
        self.procfinish = False
        self.citedby = 0
        self.usedby = 0
    def __call__(self, saveall=0, verbose=0):
        if self.procstart and not self.procfinish:
            raise Exception('(!) Circular dependency in workflow.')
        if not self.procfinish:
            self.procstart = True
            self.procout = self.process(saveall, verbose)
            self.procfinish = True
        return self.procout
    def process(self, saveall=0, verbose=0):
        pin = self.feeder(saveall, verbose)
        self.feeder.usedby += 1
        if saveall == 0 and self.feeder.usedby == self.feeder.citedby:
            self.feeder.reset(recursive=False)
        if verbose > 0:
            self.printout(verbose, pin)
        return self.transform(pin)
    def transform(self, pin):
        return pin
    def reset(self, recursive=True):
        self.procout = None
        self.procstart = False
        self.procfinish = False
        if recursive:
            self.feeder.reset(recursive=True)
    def printout(self, verbose, *args):
        if verbose >= 1:
            print(type(self))
        if verbose >= 2:
            print(vars(self))
        if verbose >= 3:
            for x in args:
                print(x)
        if verbose >= 2:
            print()
    def selfstring(self, offset=0):
        return ' '*2*offset + type(self).__name__ + '\n'
    def __str__(self, offset=0):
        return self.selfstring(offset) + self.feeder.__str__(offset+1)
    def attach_check(self, ps):
        if not self.attach(ps):
            raise Exception('(!) ' + type(self).__name__
                            + ' has no free input at which to attach '
                            + type(ps).__name__ + '.')
    def attach(self, ps):
        if self.feeder is None:
            self.feeder = ps
            self.feeder.citedby += 1
            return True
        else:
            return self.feeder.attach(ps) or ps is self
    def __mul__(self, other):
        other.attach_check(self)
        return other
    def __or__(self, other):
        other.attach_check(self)
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
    def process(self, saveall=0, verbose=0):
        if verbose > 0:
            self.printout(verbose)
        return self.load()
    def load(self):
        return self.source
    def reset(self, recursive=True):
        self.procout = None
        self.procstart = False
        self.procfinish = False
    def __str__(self, offset=0):
        return self.selfstring(offset)
    def attach(self, ps):
        return ps is self


class MergeSegment(PipeSegment):
    def __init__(self, feeder1, feeder2):
        super().__init__()
        self.feeder1 = feeder1
        self.feeder1.citedby += 1
        self.feeder2 = feeder2
        self.feeder2.citedby += 1
    def process(self, saveall=0, verbose=0):
        p1 = self.feeder1(saveall, verbose)
        p2 = self.feeder2(saveall, verbose)
        self.feeder1.usedby += 1
        if saveall == 0 and self.feeder1.usedby == self.feeder1.citedby:
            self.feeder1.reset(recursive=False)
        self.feeder2.usedby += 1
        if saveall == 0 and self.feeder2.usedby == self.feeder2.citedby:
            self.feeder2.reset(recursive=False)
        if verbose > 0:
            self.printout(verbose, p1, p2)
        if not isinstance(p1, tuple):
            p1 = (p1,)
        if not isinstance(p2, tuple):
            p2 = (p2,)
        return p1 + p2
    def reset(self, recursive=True):
        self.procout = None
        self.procstart = False
        self.procfinish = False
        if recursive:
            self.feeder1.reset(recursive=True)
            self.feeder2.reset(recursive=True)
    def __str__(self, offset=0):
        return self.selfstring(offset) \
            + self.feeder1.__str__(offset+1) \
            + self.feeder2.__str__(offset+1)
    def attach(self, ps):
        if self.feeder1 is None:
            self.feeder1 = ps
            self.feeder.citedby += 1
            flag1 = True
        else:
            flag1 = self.feeder1.attach(ps)
        if self.feeder2 is None:
            self.feeder2 = ps
            self.feeder.citedby += 1
            flag2 = True
        else:
            flag2 = self.feeder2.attach(ps)
        return flag1 or flag2 or ps is self


class SelectItem(PipeSegment):
    """
    Given an iterable, return one of its items.  This can be used to select
    a single output from a class that returns a tuple of outputs.
    """
    def __init__(self, index=0):
        super().__init__()
        self.index = index
    def transform(self, pin):
        return pin[self.index]


class Identity(PipeSegment):
    """
    This class is an alias for the PipeSegment base class, to
    emphasize its property of passing data through unchanged.
    Formally, this is the identity element for the '*' operation.
    """
    pass


class ReturnEmpty(PipeSegment):
    """
    Regardless of input, returns an empty tuple.
    This can be useful in Map and Conditional classes.
    Formally, this is the identity element for the '+' operation.
    """
    def transform(self, pin):
        return ()


class Conditional(PipeSegment):
    """
    This is the pipesegment version of an if statement.
    The piped input is fed into an object of the 'condition_class' class.
    If 'True' is returned, then the input is fed through an 'if_class' object.
    Otherwise, the input is fed through an 'else_class' object.
    """
    def __init__(self, condition_class,
                 if_class=Identity, else_class=ReturnEmpty,
                 condition_args=[], condition_kwargs={},
                 if_args=[], if_kwargs={}, else_args=[], else_kwargs={}):
        super().__init__()
        self.condition_class = condition_class
        self.if_class = if_class
        self.else_class = else_class
        self.condition_args = condition_args
        self.condition_kwargs = condition_kwargs
        self.if_args = if_args
        self.if_kwargs = if_kwargs
        self.else_args = else_args
        self.else_kwargs = else_kwargs
        if issubclass(self.condition_class, LoadSegment) \
           and issubclass(self.if_class, LoadSegment) \
           and issubclass(self.else_class, LoadSegment):
            self.feeder = LoadSegment(None)
    def __call__(self, saveall=0, verbose=0):
        self.saveall = saveall
        self.verbose = verbose
        super().__call__(saveall, verbose)
    def transform(self, pin):
        condition_obj = self.condition_class(*condition_args,
                                             **condition_kwargs)
        if not issubclass(self.condition_class, LoadSegment):
            condition_obj = pin * condition_obj
        if condition_obj(self.saveall, self.verbose):
            inner_obj = self.if_class(*if_args, **if_kwargs)
        else:
            inner_obj = self.else_class(*else_args, **else_kwargs)
        if not issubclass(self.condition_class, LoadSegment):
            inner_obj = pin * inner_obj
        return inner_obj(self.saveall, self.verbose)


class PipeArgs(PipeSegment):
    """
    Wrapper for any PipeSegment subclass which enables it to accept
    initialization arguments from piped input.
    """
    def __init__(self, inner_class, *args, **kwargs):
        super().__init__()
        self.inner_class = inner_class
        self.args = args
        self.kwargs = kwargs
    def transform(self, pin):
        if issubclass(self.inner_class, LoadSegment):
            isloadsegment = True
            argstart = 0
        else:
            isloadsegment = False
            argstart = 1
            inner_pin = pin[0]
        # Gather all initialization arguments
        args = self.args
        kwargs = self.kwargs.copy()
        pargs = (pin if isinstance(pin, tuple) else (pin,))[argstart:]
        for p in pargs:
            if isinstance(p, dict):
                kwargs.update(p)
            else:
                args = args + (p,)
        #Initialize and call object
        obj = self.inner_class(*args, **kwargs)
        if isloadsegment:
            return obj()
        else:
            return (inner_pin * obj)()


class FunctionPipe(PipeSegment):
    """
    Turns a user-supplied function into a PipeSegment
    """
    def __init__(self, function):
        super().__init__()
        self.function = function
    def transform(self, pin):
        return self.function(pin)


def PipeFunction(inner_class=PipeSegment, pin=(), *args, **kwargs):
    """
    Turns a PipeSegment into a standalone function.
    inner_class is the PipeSegment class, pin is the input to pipe into it,
    and *args and **kwargs are sent to the PipeSegment's constructor.
    """
    psobject = inner_class(*args, **kwargs)
    if issubclass(self.inner_class, LoadSegment):
        return psobject()
    else:
        return (pin * psobject)()
