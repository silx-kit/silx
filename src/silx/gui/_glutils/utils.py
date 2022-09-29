# /*##########################################################################
#
# Copyright (c) 2014-2021 European Synchrotron Radiation Facility
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
"""This module provides conversion functions between OpenGL and numpy types.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"

import numpy

from OpenGL.constants import BYTE_SIZES as _BYTE_SIZES
from OpenGL.constants import ARRAY_TO_GL_TYPE_MAPPING as _ARRAY_TO_GL_TYPE_MAPPING


def sizeofGLType(type_):
    """Returns the size in bytes of an element of type `type_`"""
    return _BYTE_SIZES[type_]


def isSupportedGLType(type_):
    """Test if a numpy type or dtype can be converted to a GL type."""
    return numpy.dtype(type_).char in _ARRAY_TO_GL_TYPE_MAPPING


def numpyToGLType(type_):
    """Returns the GL type corresponding the provided numpy type or dtype."""
    return _ARRAY_TO_GL_TYPE_MAPPING[numpy.dtype(type_).char]


def segmentTrianglesIntersection(segment, triangles):
    """Check for segment/triangles intersection.

    This is based on signed tetrahedron volume comparison.

    See A. Kensler, A., Shirley, P.
    Optimizing Ray-Triangle Intersection via Automated Search.
    Symposium on Interactive Ray Tracing, vol. 0, p33-38 (2006)

    :param numpy.ndarray segment:
        Segment end points as a 2x3 array of coordinates
    :param numpy.ndarray triangles:
        Nx3x3 array of triangles
    :return: (triangle indices, segment parameter, barycentric coord)
        Indices of intersected triangles, "depth" along the segment
        of the intersection point and barycentric coordinates of intersection
        point in the triangle.
    :rtype: List[numpy.ndarray]
    """
    # TODO triangles from vertices + indices
    # TODO early rejection? e.g., check segment bbox vs triangle bbox
    segment = numpy.asarray(segment)
    assert segment.ndim == 2
    assert segment.shape == (2, 3)

    triangles = numpy.asarray(triangles)
    assert triangles.ndim == 3
    assert triangles.shape[1] == 3

    # Test line/triangles intersection
    d = segment[1] - segment[0]
    t0s0 = segment[0] - triangles[:, 0, :]
    edge01 = triangles[:, 1, :] - triangles[:, 0, :]
    edge02 = triangles[:, 2, :] - triangles[:, 0, :]

    dCrossEdge02 = numpy.cross(d, edge02)
    t0s0CrossEdge01 = numpy.cross(t0s0, edge01)
    volume = numpy.sum(dCrossEdge02 * edge01, axis=1)
    del edge01
    subVolumes = numpy.empty((len(triangles), 3), dtype=triangles.dtype)
    subVolumes[:, 1] = numpy.sum(dCrossEdge02 * t0s0, axis=1)
    del dCrossEdge02
    subVolumes[:, 2] = numpy.sum(t0s0CrossEdge01 * d, axis=1)
    subVolumes[:, 0] = volume - subVolumes[:, 1] - subVolumes[:, 2]
    intersect = numpy.logical_or(
        numpy.all(subVolumes >= 0., axis=1),  # All positive
        numpy.all(subVolumes <= 0., axis=1))  # All negative
    intersect = numpy.where(intersect)[0]  # Indices of intersected triangles

    # Get barycentric coordinates
    with numpy.errstate(invalid="ignore"):
        barycentric = subVolumes[intersect] / volume[intersect].reshape(-1, 1)
    del subVolumes

    # Test segment/triangles intersection
    volAlpha = numpy.sum(t0s0CrossEdge01[intersect] * edge02[intersect], axis=1)
    with numpy.errstate(invalid="ignore"):
        t = volAlpha / volume[intersect]  # segment parameter of intersected triangles
    del t0s0CrossEdge01
    del edge02
    del volAlpha
    del volume

    inSegmentMask = numpy.logical_and(t >= 0., t <= 1.)
    intersect = intersect[inSegmentMask]
    t = t[inSegmentMask]
    barycentric = barycentric[inSegmentMask]

    # Sort intersecting triangles by t
    indices = numpy.argsort(t)
    return intersect[indices], t[indices], barycentric[indices]
