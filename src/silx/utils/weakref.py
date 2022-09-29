# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""Weakref utils for compatibility between Python 2 and Python 3 or for
extended features.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "15/09/2016"


import weakref
import types
import inspect


def ref(object, callback=None):
    """Returns a weak reference to object. The original object can be retrieved
    by calling the reference object if the referent is still alive. If the
    referent is no longer alive, calling the reference object will cause None
    to be returned.

    The signature is the same as the standard `weakref` library, but it returns
    `WeakMethod` if the object is a bound method.

    :param object: An object
    :param func callback: If provided, and the returned weakref object is
        still alive, the callback will be called when the object is about to
        be finalized. The weak reference object will be passed as the only
        parameter to the callback. Then the referent will no longer be
        available.
    :return: A weak reference to the object
    """
    if inspect.ismethod(object):
        return WeakMethod(object, callback)
    else:
        return weakref.ref(object, callback)


def proxy(object, callback=None):
    """Return a proxy to object which uses a weak reference. This supports use
    of the proxy in most contexts instead of requiring the explicit
    dereferencing used with weak reference objects.

    The signature is the same as the standard `weakref` library, but it returns
    `WeakMethodProxy` if the object is a bound method.

    :param object: An object
    :param func callback: If provided, and the returned weakref object is
        still alive, the callback will be called when the object is about to
        be finalized. The weak reference object will be passed as the only
        parameter to the callback. Then the referent will no longer be
        available.
    :return: A proxy to a weak reference of the object
    """
    if inspect.ismethod(object):
        return WeakMethodProxy(object, callback)
    else:
        return weakref.proxy(object, callback)


class WeakMethod(object):
    """Wraps a callable object like a function or a bound method.
    Feature callback when the object is about to be finalized.
    Provids the same interface as a normal weak reference.
    """

    def __init__(self, function, callback=None):
        """
        Constructor
        :param function: Function/method to be called
        :param callback: If callback is provided and not None,
            and the returned weakref object is still alive, the
            callback will be called when the object is about to
            be finalized; the weak reference object will be passed
            as the only parameter to the callback; the referent will
            no longer be available
        """
        self.__callback = callback

        if inspect.ismethod(function):
            # it is a bound method
            self.__obj = weakref.ref(function.__self__, self.__call_callback)
            self.__method = weakref.ref(function.__func__, self.__call_callback)
        else:
            self.__obj = None
            self.__method = weakref.ref(function, self.__call_callback)

    def __call_callback(self, ref):
        """Called when the object is about to be finalized"""
        if not self.is_alive():
            return
        self.__obj = None
        self.__method = None
        if self.__callback is not None:
            self.__callback(self)

    def __call__(self):
        """Return a callable function or None if the WeakMethod is dead."""
        if self.__obj is not None:
            method = self.__method()
            obj = self.__obj()
            if method is None or obj is None:
                return None
            return types.MethodType(method, obj)
        elif self.__method is not None:
            return self.__method()
        else:
            return None

    def is_alive(self):
        """True if the WeakMethod is still alive"""
        return self.__method is not None

    def __eq__(self, other):
        """Check it another obect is equal to this.

        :param object other: Object to compare with
        """
        if isinstance(other, WeakMethod):
            if not self.is_alive():
                return False
            return self.__obj == other.__obj and self.__method == other.__method
        return False

    def __ne__(self, other):
        """Check it another obect is not equal to this.

        :param object other: Object to compare with
        """
        if isinstance(other, WeakMethod):
            if not self.is_alive():
                return False
            return self.__obj != other.__obj or self.__method != other.__method
        return True

    def __hash__(self):
        """Returns the hash for the object."""
        return self.__obj.__hash__() ^ self.__method.__hash__()


class WeakMethodProxy(WeakMethod):
    """Wraps a callable object like a function or a bound method
    with a weakref proxy.
    """
    def __call__(self, *args, **kwargs):
        """Dereference the method and call it if the method is still alive.
        Else raises an ReferenceError.

        :raises: ReferenceError, if the method is not alive
        """
        fn = super(WeakMethodProxy, self).__call__()
        if fn is None:
            raise ReferenceError("weakly-referenced object no longer exists")
        return fn(*args, **kwargs)


class WeakList(list):
    """Manage a list of weaked references.
    When an object is dead, the list is flaged as invalid.
    If expected the list is cleaned up to remove dead objects.
    """

    def __init__(self, enumerator=()):
        """Create a WeakList

        :param iterator enumerator: A list of object to initialize the
            list
        """
        list.__init__(self)
        self.__list = []
        self.__is_valid = True
        for obj in enumerator:
            self.append(obj)

    def __invalidate(self, ref):
        """Flag the list as invalidated. The list contains dead references."""
        self.__is_valid = False

    def __create_ref(self, obj):
        """Create a weakref from an object. It uses the `ref` module function.
        """
        return ref(obj, self.__invalidate)

    def __clean(self):
        """Clean the list from dead references"""
        if self.__is_valid:
            return
        self.__list = [ref for ref in self.__list if ref() is not None]
        self.__is_valid = True

    def __iter__(self):
        """Iterate over objects of the list"""
        for ref in self.__list:
            obj = ref()
            if obj is not None:
                yield obj

    def __len__(self):
        """Count item on the list"""
        self.__clean()
        return len(self.__list)

    def __getitem__(self, key):
        """Returns the object at the requested index

        :param key: Indexes to get
        :type key: int or slice
        """
        self.__clean()
        data = self.__list[key]
        if isinstance(data, list):
            result = [ref() for ref in data]
        else:
            result = data()
        return result

    def __setitem__(self, key, obj):
        """Set an item at an index

        :param key: Indexes to set
        :type key: int or slice
        """
        self.__clean()
        if isinstance(key, slice):
            objs = [self.__create_ref(o) for o in obj]
            self.__list[key] = objs
        else:
            obj_ref = self.__create_ref(obj)
            self.__list[key] = obj_ref

    def __delitem__(self, key):
        """Delete an Indexes item of this list

        :param key: Index to delete
        :type key: int or slice
         """
        self.__clean()
        del self.__list[key]

    def __delslice__(self, i, j):
        """Looks to be used in Python 2.7"""
        self.__delitem__(slice(i, j, None))

    def __setslice__(self, i, j, sequence):
        """Looks to be used in Python 2.7"""
        self.__setitem__(slice(i, j, None), sequence)

    def __getslice__(self, i, j):
        """Looks to be used in Python 2.7"""
        return self.__getitem__(slice(i, j, None))

    def __reversed__(self):
        """Returns a copy of the reverted list"""
        reversed_list = reversed(list(self))
        return WeakList(reversed_list)

    def __contains__(self, obj):
        """Returns true if the object is in the list"""
        ref = self.__create_ref(obj)
        return ref in self.__list

    def __add__(self, other):
        """Returns a WeakList containing this list an the other"""
        l = WeakList(self)
        l.extend(other)
        return l

    def __iadd__(self, other):
        """Add objects to this list inplace"""
        self.extend(other)
        return self

    def __mul__(self, n):
        """Returns a WeakList containing n-duplication object of this list"""
        return WeakList(list(self) * n)

    def __imul__(self, n):
        """N-duplication of the objects to this list inplace"""
        self.__list *= n
        return self

    def append(self, obj):
        """Add an object at the end of the list"""
        ref = self.__create_ref(obj)
        self.__list.append(ref)

    def count(self, obj):
        """Returns the number of occurencies of an object"""
        ref = self.__create_ref(obj)
        return self.__list.count(ref)

    def extend(self, other):
        """Append the list with all objects from another list"""
        for obj in other:
            self.append(obj)

    def index(self, obj):
        """Returns the index of an object"""
        ref = self.__create_ref(obj)
        return self.__list.index(ref)

    def insert(self, index, obj):
        """Insert an object at the requested index"""
        ref = self.__create_ref(obj)
        self.__list.insert(index, ref)

    def pop(self, index=-1):
        """Remove and return an object at the requested index"""
        self.__clean()
        obj = self.__list.pop(index)()
        return obj

    def remove(self, obj):
        """Remove an object from the list"""
        ref = self.__create_ref(obj)
        self.__list.remove(ref)

    def reverse(self):
        """Reverse the list inplace"""
        self.__list.reverse()

    def sort(self, key=None, reverse=False):
        """Sort the list inplace.
        Not very efficient.
        """
        sorted_list = list(self)
        sorted_list.sort(key=key, reverse=reverse)
        self.__list = []
        self.extend(sorted_list)

    def __str__(self):
        unref_list = list(self)
        return "WeakList(%s)" % str(unref_list)

    def __repr__(self):
        unref_list = list(self)
        return "WeakList(%s)" % repr(unref_list)
