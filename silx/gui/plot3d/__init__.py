# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""
This package provides widgets displaying 3D content based on OpenGL.
"""
from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/01/2017"


import logging as _logging

from .. import qt as _qt

try:
    import OpenGL as _OpenGL
except ImportError:
    _OpenGL = None


_logger = _logging.getLogger(__name__)


if not _qt.HAS_OPENGL:
    _logger.warning(
        'Qt.QtOpenGL is not available: silx.gui.plot3d modules will fail.')

if _OpenGL is None:
    _logger.warning(
        'PyOpenGL is not installed: silx.gui.plot3d modules will fail')


def isAvailable():
    """Returns True if plot3d functionality is available, False otherwise.

    This function checks for PyOpenGL and QtOpenGL availability.
    The availability of OpenGL 2.1 (required by plot3d) is not checked here.

    :rtype: bool"""
    return _OpenGL is not None and _qt.HAS_OPENGL
