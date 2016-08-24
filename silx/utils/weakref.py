# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/08/2016"


import weakref
import types
import inspect


class WeakMethod(object):
    """Wraps a callable object like a function or a bound method.
    Feature callback when the object is about to be finalized.
    Provids the same interface as a normal weak reference.
    """

    def __init__(self, callable, callback=None):
        """
        Constructor
        :param callable: Function/method to be called
        :param callback: If callback is provided and not None,
            and the returned weakref object is still alive, the
            callback will be called when the object is about to
            be finalized; the weak reference object will be passed
            as the only parameter to the callback; the referent will
            no longer be available
        """
        self.__callback = callback
        weakref_callback = self.__call_callback if callback is not None else None

        if inspect.ismethod(callable):
            # it is a bound method
            self.__obj = weakref.ref(callable.__self__, weakref_callback)
            self.__method = weakref.ref(callable.__func__, weakref_callback)
        else:
            self.__obj = None
            self.__method = weakref.ref(callable, weakref_callback)

    def __call_callback(self, ref):
        """Called when the object is about to be finalized"""
        if self.__obj is None and self.__method is None:
            return
        self.__obj = None
        self.__method = None
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
