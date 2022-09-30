# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Module containing proxy objects"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "02/10/2017"


import functools


class Proxy(object):
    """Create a proxy of an object.

    Provides default methods and property using :meth:`__getattr__` and special
    method by redefining them one by one.
    Special methods are defined as properties, as a result if the `obj` method
    is not defined, the property code fail and the special method will not be
    visible.
    """

    __slots__ = ["__obj", "__weakref__"]

    def __init__(self, obj):
        object.__setattr__(self, "_Proxy__obj", obj)

    __class__ = property(lambda self: self.__obj.__class__)

    def __getattr__(self, name):
        return getattr(self.__obj, name)

    __setattr__ = property(lambda self: self.__obj.__setattr__)
    __delattr__ = property(lambda self: self.__obj.__delattr__)

    # binary comparator methods

    __lt__ = property(lambda self: self.__obj.__lt__)
    __le__ = property(lambda self: self.__obj.__le__)
    __eq__ = property(lambda self: self.__obj.__eq__)
    __ne__ = property(lambda self: self.__obj.__ne__)
    __gt__ = property(lambda self: self.__obj.__gt__)
    __ge__ = property(lambda self: self.__obj.__ge__)

    # binary numeric methods

    __add__ = property(lambda self: self.__obj.__add__)
    __radd__ = property(lambda self: self.__obj.__radd__)
    __iadd__ = property(lambda self: self.__obj.__iadd__)
    __sub__ = property(lambda self: self.__obj.__sub__)
    __rsub__ = property(lambda self: self.__obj.__rsub__)
    __isub__ = property(lambda self: self.__obj.__isub__)
    __mul__ = property(lambda self: self.__obj.__mul__)
    __rmul__ = property(lambda self: self.__obj.__rmul__)
    __imul__ = property(lambda self: self.__obj.__imul__)

    __truediv__ = property(lambda self: self.__obj.__truediv__)
    __rtruediv__ = property(lambda self: self.__obj.__rtruediv__)
    __itruediv__ = property(lambda self: self.__obj.__itruediv__)
    __floordiv__ = property(lambda self: self.__obj.__floordiv__)
    __rfloordiv__ = property(lambda self: self.__obj.__rfloordiv__)
    __ifloordiv__ = property(lambda self: self.__obj.__ifloordiv__)
    __mod__ = property(lambda self: self.__obj.__mod__)
    __rmod__ = property(lambda self: self.__obj.__rmod__)
    __imod__ = property(lambda self: self.__obj.__imod__)
    __divmod__ = property(lambda self: self.__obj.__divmod__)
    __rdivmod__ = property(lambda self: self.__obj.__rdivmod__)
    __pow__ = property(lambda self: self.__obj.__pow__)
    __rpow__ = property(lambda self: self.__obj.__rpow__)
    __ipow__ = property(lambda self: self.__obj.__ipow__)
    __lshift__ = property(lambda self: self.__obj.__lshift__)
    __rlshift__ = property(lambda self: self.__obj.__rlshift__)
    __ilshift__ = property(lambda self: self.__obj.__ilshift__)
    __rshift__ = property(lambda self: self.__obj.__rshift__)
    __rrshift__ = property(lambda self: self.__obj.__rrshift__)
    __irshift__ = property(lambda self: self.__obj.__irshift__)

    # binary logical methods

    __and__ = property(lambda self: self.__obj.__and__)
    __rand__ = property(lambda self: self.__obj.__rand__)
    __iand__ = property(lambda self: self.__obj.__iand__)
    __xor__ = property(lambda self: self.__obj.__xor__)
    __rxor__ = property(lambda self: self.__obj.__rxor__)
    __ixor__ = property(lambda self: self.__obj.__ixor__)
    __or__ = property(lambda self: self.__obj.__or__)
    __ror__ = property(lambda self: self.__obj.__ror__)
    __ior__ = property(lambda self: self.__obj.__ior__)

    # unary methods

    __neg__ = property(lambda self: self.__obj.__neg__)
    __pos__ = property(lambda self: self.__obj.__pos__)
    __abs__ = property(lambda self: self.__obj.__abs__)
    __invert__ = property(lambda self: self.__obj.__invert__)
    __floor__ = property(lambda self: self.__obj.__floor__)
    __ceil__ = property(lambda self: self.__obj.__ceil__)
    __round__ = property(lambda self: self.__obj.__round__)

    # cast

    __repr__ = property(lambda self: self.__obj.__repr__)
    __str__ = property(lambda self: self.__obj.__str__)
    __complex__ = property(lambda self: self.__obj.__complex__)
    __int__ = property(lambda self: self.__obj.__int__)
    __float__ = property(lambda self: self.__obj.__float__)
    __hash__ = property(lambda self: self.__obj.__hash__)
    __bytes__ = property(lambda self: self.__obj.__bytes__)
    __bool__ = property(lambda self: lambda: bool(self.__obj))
    __format__ = property(lambda self: self.__obj.__format__)

    # container

    __len__ = property(lambda self: self.__obj.__len__)
    __length_hint__ = property(lambda self: self.__obj.__length_hint__)
    __getitem__ = property(lambda self: self.__obj.__getitem__)
    __missing__ = property(lambda self: self.__obj.__missing__)
    __setitem__ = property(lambda self: self.__obj.__setitem__)
    __delitem__ = property(lambda self: self.__obj.__delitem__)
    __iter__ = property(lambda self: self.__obj.__iter__)
    __reversed__ = property(lambda self: self.__obj.__reversed__)
    __contains__ = property(lambda self: self.__obj.__contains__)

    # pickle

    __reduce__ = property(lambda self: self.__obj.__reduce__)
    __reduce_ex__ = property(lambda self: self.__obj.__reduce_ex__)

    # async

    __await__ = property(lambda self: self.__obj.__await__)
    __aiter__ = property(lambda self: self.__obj.__aiter__)
    __anext__ = property(lambda self: self.__obj.__anext__)
    __aenter__ = property(lambda self: self.__obj.__aenter__)
    __aexit__ = property(lambda self: self.__obj.__aexit__)

    # other

    __index__ = property(lambda self: self.__obj.__index__)

    __next__ = property(lambda self: self.__obj.__next__)

    __enter__ = property(lambda self: self.__obj.__enter__)
    __exit__ = property(lambda self: self.__obj.__exit__)

    __concat__ = property(lambda self: self.__obj.__concat__)
    __iconcat__ = property(lambda self: self.__obj.__iconcat__)

    __call__ = property(lambda self: self.__obj.__call__)


def _docstring(dest, origin):
    """Implementation of docstring decorator.

    It patches dest.__doc__.
    """
    if not isinstance(dest, type) and isinstance(origin, type):
        # func is not a class, but origin is, get the method with the same name
        try:
            origin = getattr(origin, dest.__name__)
        except AttributeError:
            raise ValueError(
                "origin class has no %s method" % dest.__name__)

    dest.__doc__ = origin.__doc__
    return dest


def docstring(origin):
    """Decorator to initialize the docstring from another source.

    This is useful to duplicate a docstring for inheritance and composition.

    If origin is a method or a function, it copies its docstring.
    If origin is a class, the docstring is copied from the method
    of that class which has the same name as the method/function
    being decorated.

    :param origin:
        The method, function or class from which to get the docstring
    :raises ValueError:
        If the origin class has not method n case the
    """
    return functools.partial(_docstring, origin=origin)
