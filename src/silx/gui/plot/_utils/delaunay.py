# /*##########################################################################
#
# Copyright (c) 2019-2023 European Synchrotron Radiation Facility
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
"""Wrapper over Delaunay implementation"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2019"


import functools
import logging
import sys

import numpy


_logger = logging.getLogger(__name__)


@functools.lru_cache()
def _get_delaunay_class():
    """Lazy-loading of Delaunay from scipy an warn once if missing"""
    try:
        from scipy.spatial import Delaunay  # noqa
    except ImportError:
        _logger.warning("scipy is missing: No Delaunay tesselation available")
        return None
    return Delaunay


def is_delaunay_available() -> bool:
    """Returns True if Delaunay tesselation is available"""
    return _get_delaunay_class() is not None


def delaunay(x, y):
    """Returns Delaunay instance for x, y points

    :param numpy.ndarray x:
    :param numpy.ndarray y:
    :rtype: Union[None,scipy.spatial.Delaunay]
    """
    delaunay_class = _get_delaunay_class()
    if delaunay_class is None:
        return None

    points = numpy.array((x, y)).T
    try:
        return delaunay_class(points)
    except (RuntimeError, ValueError):
        _logger.debug("Delaunay tesselation failed: %s", sys.exc_info()[1])
        return None
