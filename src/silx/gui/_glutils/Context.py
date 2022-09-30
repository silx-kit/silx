# /*##########################################################################
#
# Copyright (c) 2014-2019 European Synchrotron Radiation Facility
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
"""Abstraction of OpenGL context.

It defines a way to get current OpenGL context to support multiple
OpenGL contexts.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"

import contextlib


class _DEFAULT_CONTEXT(object):
    """The default value for OpenGL context"""
    pass

_context = _DEFAULT_CONTEXT
"""The current OpenGL context"""


def getCurrent():
    """Returns platform dependent object of current OpenGL context.

    This is useful to associate OpenGL resources with the context they are
    created in.

    :return: Platform specific OpenGL context
    """
    return _context


def setCurrent(context=_DEFAULT_CONTEXT):
    """Set a platform dependent OpenGL context

    :param context: Platform dependent GL context
    """
    global _context
    _context = context


@contextlib.contextmanager
def current(context):
    """Context manager setting the platform-dependent GL context

    :param context: Platform dependent GL context
    """
    previous_context = getCurrent()
    setCurrent(context)
    yield
    setCurrent(previous_context)
