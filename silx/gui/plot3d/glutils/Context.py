# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2017 European Synchrotron Radiation Facility
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


# context #####################################################################


def _defaultGLContextGetter():
    return None

_glContextGetter = _defaultGLContextGetter


def getGLContext():
    """Returns platform dependent object of current OpenGL context.

    This is useful to associate OpenGL resources with the context they are
    created in.

    :return: Platform specific OpenGL context
    :rtype: None by default or a platform dependent object"""
    return _glContextGetter()


def setGLContextGetter(getter=_defaultGLContextGetter):
    """Set a platform dependent function to retrieve the current OpenGL context

    :param getter: Platform dependent GL context getter
    :type getter: Function with no args returning the current OpenGL context
    """
    global _glContextGetter
    _glContextGetter = getter
