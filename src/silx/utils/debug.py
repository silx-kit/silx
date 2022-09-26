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


import inspect
import types
import logging


debug_logger = logging.getLogger("silx.DEBUG")

_indent = 0


def log_method(func, class_name=None):
    """Decorator to inject a warning log before an after any function/method.

    .. code-block:: python

        @log_method
        def foo():
            return None

    :param callable func: The function to patch
    :param str class_name: In case a method, provide the class name
    """
    def wrapper(*args, **kwargs):
        global _indent

        indent = "  " * _indent
        if class_name is not None:
            name = "%s.%s" % (class_name, func.__name__)
        else:
            name = "%s" % func.__name__

        debug_logger.warning("%s%s" % (indent, name))
        _indent += 1
        result = func(*args, **kwargs)
        _indent -= 1
        debug_logger.warning("%sreturn  (%s)" % (indent, name))
        return result
    return wrapper


def log_all_methods(base_class):
    """Decorator to inject a warning log before an after any method provided by
    a class.

    .. code-block:: python

        @log_all_methods
        class Foo(object):

            def a(self):
                return None

            def b(self):
                return self.a()

    Here is the output when calling the `b` method.

    .. code-block::

        WARNING:silx.DEBUG:_Foobar.b
        WARNING:silx.DEBUG:  _Foobar.a
        WARNING:silx.DEBUG:  return  (_Foobar.a)
        WARNING:silx.DEBUG:return  (_Foobar.b)

    :param class base_class: The class to patch
    """
    methodTypes = (types.MethodType, types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType)
    for name, func in inspect.getmembers(base_class):
        if isinstance(func, methodTypes):
            if func.__name__ not in ["__subclasshook__", "__new__"]:
                # patching __new__ in Python2 break the object, then we skip it
                setattr(base_class, name, log_method(func, base_class.__name__))

    return base_class
