# /*##########################################################################
#
# Copyright (c) 2015-2020 European Synchrotron Radiation Facility
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
This module provides functions to generate indices, to check intersection
and to handle planes.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import logging
import numpy

from . import event


_logger = logging.getLogger(__name__)


# numpy #######################################################################

def _uniqueAlongLastAxis(a):
    """Numpy unique on the last axis of a 2D array

    Implemented here as not in numpy as of writing.

    See adding axis parameter to numpy.unique:
    https://github.com/numpy/numpy/pull/3584/files#r6225452

    :param array_like a: Input array.
    :return: Unique elements along the last axis.
    :rtype: numpy.ndarray
    """
    assert len(a.shape) == 2

    # Construct a type over last array dimension to run unique on a 1D array
    if a.dtype.char in numpy.typecodes['AllInteger']:
        # Bit-wise comparison of the 2 indices of a line at once
        # Expect a C contiguous array of shape N, 2
        uniquedt = numpy.dtype((numpy.void, a.itemsize * a.shape[-1]))
    elif a.dtype.char in numpy.typecodes['Float']:
        uniquedt = [('f{i}'.format(i=i), a.dtype) for i in range(a.shape[-1])]
    else:
        raise TypeError("Unsupported type {dtype}".format(dtype=a.dtype))

    uniquearray = numpy.unique(numpy.ascontiguousarray(a).view(uniquedt))
    return uniquearray.view(a.dtype).reshape((-1, a.shape[-1]))


# conversions #################################################################

def triangleToLineIndices(triangleIndices, unicity=False):
    """Generates lines indices from triangle indices.

    This is generating lines indices for the edges of the triangles.

    :param triangleIndices: The indices to draw a set of vertices as triangles.
    :type triangleIndices: numpy.ndarray
    :param bool unicity: If True remove duplicated lines,
                         else (the default) returns all lines.
    :return: The indices to draw the edges of the triangles as lines.
    :rtype: 1D numpy.ndarray of uint16 or uint32.
    """
    # Makes sure indices ar packed by triangle
    triangleIndices = triangleIndices.reshape(-1, 3)

    # Pack line indices by triangle and by edge
    lineindices = numpy.empty((len(triangleIndices), 3, 2),
                              dtype=triangleIndices.dtype)
    lineindices[:, 0] = triangleIndices[:, :2]  # edge = t0, t1
    lineindices[:, 1] = triangleIndices[:, 1:]  # edge =t1, t2
    lineindices[:, 2] = triangleIndices[:, ::2]  # edge = t0, t2

    if unicity:
        lineindices = _uniqueAlongLastAxis(lineindices.reshape(-1, 2))

    # Make sure it is 1D
    lineindices.shape = -1

    return lineindices


def verticesNormalsToLines(vertices, normals, scale=1.):
    """Return vertices of lines representing normals at given positions.

    :param vertices: Positions of the points.
    :type vertices: numpy.ndarray with shape: (nbPoints, 3)
    :param normals: Corresponding normals at the points.
    :type normals: numpy.ndarray with shape: (nbPoints, 3)
    :param float scale: The scale factor to apply to normals.
    :returns: Array of vertices to draw corresponding lines.
    :rtype: numpy.ndarray with shape: (nbPoints * 2, 3)
    """
    linevertices = numpy.empty((len(vertices) * 2, 3), dtype=vertices.dtype)
    linevertices[0::2] = vertices
    linevertices[1::2] = vertices + scale * normals
    return linevertices


def unindexArrays(mode, indices, *arrays):
    """Convert indexed GL primitives to unindexed ones.

    Given indices in arrays and the OpenGL primitive they represent,
    return the unindexed equivalent.

    :param str mode:
       Kind of primitive represented by indices.
       In: points, lines, line_strip, loop, triangles, triangle_strip, fan.
    :param indices: Indices in other arrays
    :type indices: numpy.ndarray of dimension 1.
    :param arrays: Remaining arguments are arrays to convert
    :return: Converted arrays
    :rtype: tuple of numpy.ndarray
    """
    indices = numpy.array(indices, copy=False)

    assert mode in ('points',
                    'lines', 'line_strip', 'loop',
                    'triangles', 'triangle_strip', 'fan')

    if mode in ('lines', 'line_strip', 'loop'):
        assert len(indices) >= 2
    elif mode in ('triangles', 'triangle_strip', 'fan'):
        assert len(indices) >= 3

    assert indices.min() >= 0
    max_index = indices.max()
    for data in arrays:
        assert len(data) >= max_index

    if mode == 'line_strip':
        unpacked = numpy.empty((2 * (len(indices) - 1),), dtype=indices.dtype)
        unpacked[0::2] = indices[:-1]
        unpacked[1::2] = indices[1:]
        indices = unpacked

    elif mode == 'loop':
        unpacked = numpy.empty((2 * len(indices),), dtype=indices.dtype)
        unpacked[0::2] = indices
        unpacked[1:-1:2] = indices[1:]
        unpacked[-1] = indices[0]
        indices = unpacked

    elif mode == 'triangle_strip':
        unpacked = numpy.empty((3 * (len(indices) - 2),), dtype=indices.dtype)
        unpacked[0::3] = indices[:-2]
        unpacked[1::3] = indices[1:-1]
        unpacked[2::3] = indices[2:]
        indices = unpacked

    elif mode == 'fan':
        unpacked = numpy.empty((3 * (len(indices) - 2),), dtype=indices.dtype)
        unpacked[0::3] = indices[0]
        unpacked[1::3] = indices[1:-1]
        unpacked[2::3] = indices[2:]
        indices = unpacked

    return tuple(numpy.ascontiguousarray(data[indices]) for data in arrays)


def triangleStripToTriangles(strip):
    """Convert a triangle strip to a set of triangles.

    The order of the corners is inverted for odd triangles.

    :param numpy.ndarray strip:
        Array of triangle corners of shape (N, 3).
        N must be at least 3.
    :return: Equivalent triangles corner as an array of shape (N, 3, 3)
    :rtype: numpy.ndarray
    """
    strip = numpy.array(strip).reshape(-1, 3)
    assert len(strip) >= 3

    triangles = numpy.empty((len(strip) - 2, 3, 3), dtype=strip.dtype)
    triangles[0::2, 0] = strip[0:-2:2]
    triangles[0::2, 1] = strip[1:-1:2]
    triangles[0::2, 2] = strip[2::2]

    triangles[1::2, 0] = strip[3::2]
    triangles[1::2, 1] = strip[2:-1:2]
    triangles[1::2, 2] = strip[1:-2:2]

    return triangles


def trianglesNormal(positions):
    """Return normal for each triangle.

    :param positions: Serie of triangle's corners
    :type positions: numpy.ndarray of shape (NbTriangles*3, 3)
    :return: Normals corresponding to each position.
    :rtype: numpy.ndarray of shape (NbTriangles, 3)
    """
    assert positions.ndim == 2
    assert positions.shape[1] == 3

    positions = numpy.array(positions, copy=False).reshape(-1, 3, 3)

    normals = numpy.cross(positions[:, 1] - positions[:, 0],
                          positions[:, 2] - positions[:, 0])

    # Normalize normals
    norms = numpy.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1

    return normals / norms.reshape(-1, 1)


# grid ########################################################################

def gridVertices(dim0Array, dim1Array, dtype):
    """Generate an array of 2D positions from 2 arrays of 1D coordinates.

    :param dim0Array: 1D array-like of coordinates along the first dimension.
    :param dim1Array: 1D array-like of coordinates along the second dimension.
    :param numpy.dtype dtype: Data type of the output array.
    :return: Array of grid coordinates.
    :rtype: numpy.ndarray with shape: (len(dim0Array), len(dim1Array), 2)
    """
    grid = numpy.empty((len(dim0Array), len(dim1Array), 2), dtype=dtype)
    grid.T[0, :, :] = dim0Array
    grid.T[1, :, :] = numpy.array(dim1Array, copy=False)[:, None]
    return grid


def triangleStripGridIndices(dim0, dim1):
    """Generate indices to draw a grid of vertices as a triangle strip.

    Vertices are expected to be stored as row-major (i.e., C contiguous).

    :param int dim0: The number of rows of vertices.
    :param int dim1: The number of columns of vertices.
    :return: The vertex indices
    :rtype: 1D numpy.ndarray of uint32
    """
    assert dim0 >= 2
    assert dim1 >= 2

    # Filling a row of squares +
    # an index before and one after for degenerated triangles
    indices = numpy.empty((dim0 - 1, 2 * (dim1 + 1)), dtype=numpy.uint32)

    # Init indices with minimum indices for each row of squares
    indices[:] = (dim1 * numpy.arange(dim0 - 1, dtype=numpy.uint32))[:, None]

    # Update indices with offset per row of squares
    offset = numpy.arange(dim1, dtype=numpy.uint32)
    indices[:, 1:-1:2] += offset
    offset += dim1
    indices[:, 2::2] += offset
    indices[:, -1] += offset[-1]

    # Remove extra indices for degenerated triangles before returning
    return indices.ravel()[1:-1]

    # Alternative:
    # indices = numpy.zeros(2 * dim1 * (dim0 - 1) + 2 * (dim0 - 2),
    #                      dtype=numpy.uint32)
    #
    # offset = numpy.arange(dim1, dtype=numpy.uint32)
    # for d0Index in range(dim0 - 1):
    #    start = 2 * d0Index * (dim1 + 1)
    #    end = start + 2 * dim1
    #    if d0Index != 0:
    #        indices[start - 2] = offset[-1]
    #        indices[start - 1] = offset[0]
    #    indices[start:end:2] = offset
    #    offset += dim1
    #    indices[start + 1:end:2] = offset
    # return indices


def linesGridIndices(dim0, dim1):
    """Generate indices to draw a grid of vertices as lines.

    Vertices are expected to be stored as row-major (i.e., C contiguous).

    :param int dim0: The number of rows of vertices.
    :param int dim1: The number of columns of vertices.
    :return: The vertex indices.
    :rtype: 1D numpy.ndarray of uint32
    """
    # Horizontal and vertical lines
    nbsegmentalongdim1 = 2 * (dim1 - 1)
    nbsegmentalongdim0 = 2 * (dim0 - 1)

    indices = numpy.empty(nbsegmentalongdim1 * dim0 +
                          nbsegmentalongdim0 * dim1,
                          dtype=numpy.uint32)

    # Line indices over dim0
    onedim1line = (numpy.arange(nbsegmentalongdim1,
                                dtype=numpy.uint32) + 1) // 2
    indices[:dim0 * nbsegmentalongdim1] = \
        (dim1 * numpy.arange(dim0, dtype=numpy.uint32)[:, None] +
         onedim1line[None, :]).ravel()

    # Line indices over dim1
    onedim0line = (numpy.arange(nbsegmentalongdim0,
                                dtype=numpy.uint32) + 1) // 2
    indices[dim0 * nbsegmentalongdim1:] = \
        (numpy.arange(dim1, dtype=numpy.uint32)[:, None] +
         dim1 * onedim0line[None, :]).ravel()

    return indices


# intersection ################################################################

def angleBetweenVectors(refVector, vectors, norm=None):
    """Return the angle between 2 vectors.

    :param refVector: Coordinates of the reference vector.
    :type refVector: numpy.ndarray of shape: (NCoords,)
    :param vectors: Coordinates of the vector(s) to get angle from reference.
    :type vectors: numpy.ndarray of shape: (NCoords,) or (NbVector, NCoords)
    :param norm: A direction vector giving an orientation to the angles
                 or None.
    :returns: The angles in radians in [0, pi] if norm is None
              else in [0, 2pi].
    :rtype: float or numpy.ndarray of shape (NbVectors,)
    """
    singlevector = len(vectors.shape) == 1
    if singlevector:  # Make it a 2D array for the computation
        vectors = vectors.reshape(1, -1)

    assert len(refVector.shape) == 1
    assert len(vectors.shape) == 2
    assert len(refVector) == vectors.shape[1]

    # Normalize vectors
    refVector /= numpy.linalg.norm(refVector)
    vectors = numpy.array([v / numpy.linalg.norm(v) for v in vectors])

    dots = numpy.sum(refVector * vectors, axis=-1)
    angles = numpy.arccos(numpy.clip(dots, -1., 1.))
    if norm is not None:
        signs = numpy.sum(norm * numpy.cross(refVector, vectors), axis=-1) < 0.
        angles[signs] = numpy.pi * 2. - angles[signs]

    return angles[0] if singlevector else angles


def segmentPlaneIntersect(s0, s1, planeNorm, planePt):
    """Compute the intersection of a segment with a plane.

    :param s0: First end of the segment
    :type s0: 1D numpy.ndarray-like of length 3
    :param s1: Second end of the segment
    :type s1: 1D numpy.ndarray-like of length 3
    :param planeNorm: Normal vector of the plane.
    :type planeNorm: numpy.ndarray of shape: (3,)
    :param planePt: A point of the plane.
    :type planePt: numpy.ndarray of shape: (3,)
    :return: The intersection points. The number of points goes
             from 0 (no intersection) to 2 (segment in the plane)
    :rtype: list of numpy.ndarray
    """
    s0, s1 = numpy.asarray(s0), numpy.asarray(s1)

    segdir = s1 - s0
    dotnormseg = numpy.dot(planeNorm, segdir)
    if dotnormseg == 0:
        # line and plane are parallels
        if numpy.dot(planeNorm, planePt - s0) == 0:  # segment is in plane
            return [s0, s1]
        else:  # No intersection
            return []

    alpha = - numpy.dot(planeNorm, s0 - planePt) / dotnormseg
    if 0. <= alpha <= 1.:  # Intersection with segment
        return [s0 + alpha * segdir]
    else:  # intersection outside segment
        return []


def boxPlaneIntersect(boxVertices, boxLineIndices, planeNorm, planePt):
    """Return intersection points between a box and a plane.

    :param boxVertices: Position of the corners of the box.
    :type boxVertices: numpy.ndarray with shape: (8, 3)
    :param boxLineIndices: Indices of the box edges.
    :type boxLineIndices: numpy.ndarray-like with shape: (12, 2)
    :param planeNorm: Normal vector of the plane.
    :type planeNorm: numpy.ndarray of shape: (3,)
    :param planePt: A point of the plane.
    :type planePt: numpy.ndarray of shape: (3,)
    :return: The found intersection points
    :rtype: numpy.ndarray with 2 dimensions
    """
    segments = numpy.take(boxVertices, boxLineIndices, axis=0)

    points = set()  # Gather unique intersection points
    for seg in segments:
        for point in segmentPlaneIntersect(seg[0], seg[1], planeNorm, planePt):
            points.add(tuple(point))
    points = numpy.array(list(points))

    if len(points) <= 2:
        return numpy.array(())
    elif len(points) == 3:
        return points
    else:  # len(points) > 3
        # Order point to have a polyline lying on the unit cube's faces
        vectors = points - numpy.mean(points, axis=0)
        angles = angleBetweenVectors(vectors[0], vectors, planeNorm)
        points = numpy.take(points, numpy.argsort(angles), axis=0)
        return points


def clipSegmentToBounds(segment, bounds):
    """Clip segment to volume aligned with axes.

    :param numpy.ndarray segment: (p0, p1)
    :param numpy.ndarray bounds: (lower corner, upper corner)
    :return: Either clipped (p0, p1) or None if outside volume
    :rtype: Union[None,List[numpy.ndarray]]
    """
    segment = numpy.array(segment, copy=False)
    bounds = numpy.array(bounds, copy=False)

    p0, p1 = segment
    # Get intersection points of ray with volume boundary planes
    # Line equation: P = offset * delta + p0
    delta = p1 - p0
    deltaNotZero = numpy.array(delta, copy=True)
    deltaNotZero[deltaNotZero == 0] = numpy.nan  # Invalidated to avoid division by zero
    offsets = ((bounds - p0) / deltaNotZero).reshape(-1)
    points = offsets.reshape(-1, 1) * delta + p0

    # Avoid precision errors by using bounds value
    points.shape = 2, 3, 3  # Reshape 1 point per bound value
    for dim in range(3):
        points[:, dim, dim] = bounds[:, dim]
    points.shape = -1, 3  # Set back to 2D array

    # Find intersection points that are included in the volume
    mask = numpy.logical_and(numpy.all(bounds[0] <= points, axis=1),
                             numpy.all(points <= bounds[1], axis=1))
    intersections = numpy.unique(offsets[mask])
    if len(intersections) != 2:
        return None

    intersections.sort()
    # Do p1 first as p0 is need to compute it
    if intersections[1] < 1:  # clip p1
        segment[1] = intersections[1] * delta + p0
    if intersections[0] > 0:  # clip p0
        segment[0] = intersections[0] * delta + p0
    return segment


def segmentVolumeIntersect(segment, nbins):
    """Get bin indices intersecting with segment

    It should work with N dimensions.
    Coordinate convention (z, y, x) or (x, y, z) should not matter
    as long as segment and nbins are consistent.

    :param numpy.ndarray segment:
        Segment end points as a 2xN array of coordinates
    :param numpy.ndarray nbins:
        Shape of the volume with same coordinates order as segment
    :return: List of bins indices as a 2D array or None if no bins
    :rtype: Union[None,numpy.ndarray]
    """
    segment = numpy.asarray(segment)
    nbins = numpy.asarray(nbins)

    assert segment.ndim == 2
    assert segment.shape[0] == 2
    assert nbins.ndim == 1
    assert segment.shape[1] == nbins.size

    dim = len(nbins)

    bounds = numpy.array((numpy.zeros_like(nbins), nbins))
    segment = clipSegmentToBounds(segment, bounds)
    if segment is None:
        return None  # Segment outside volume
    p0, p1 = segment

    # Get intersections

    # Get coordinates of bin edges crossing the segment
    clipped = numpy.ceil(numpy.clip(segment, 0, nbins))
    start = numpy.min(clipped, axis=0)
    stop = numpy.max(clipped, axis=0)  # stop is NOT included
    edgesByDim = [numpy.arange(start[i], stop[i]) for i in range(dim)]

    # Line equation: P = t * delta + p0
    delta = p1 - p0

    # Get bin edge/line intersections as sorted points along the line
    # Get corresponding line parameters
    t = []
    if numpy.all(0 <= p0) and numpy.all(p0 <= nbins):
        t.append([0.])  # p0 within volume, add it
    t += [(edgesByDim[i] - p0[i]) / delta[i] for i in range(dim) if delta[i] != 0]
    if numpy.all(0 <= p1) and numpy.all(p1 <= nbins):
        t.append([1.])  # p1 within volume, add it
    t = numpy.concatenate(t)
    t.sort(kind='mergesort')

    # Remove duplicates
    unique = numpy.ones((len(t),), dtype=bool)
    numpy.not_equal(t[1:], t[:-1], out=unique[1:])
    t = t[unique]

    if len(t) < 2:
        return None  # Not enough intersection points

    # bin edges/line intersection points
    points = t.reshape(-1, 1) * delta + p0
    centers = (points[:-1] + points[1:]) / 2.
    bins = numpy.floor(centers).astype(numpy.int64)
    return bins


# Plane #######################################################################

class Plane(event.Notifier):
    """Object handling a plane and notifying plane changes.

    :param point: A point on the plane.
    :type point: 3-tuple of float.
    :param normal: Normal of the plane.
    :type normal: 3-tuple of float.
    """

    def __init__(self, point=(0., 0., 0.), normal=(0., 0., 1.)):
        super(Plane, self).__init__()

        assert len(point) == 3
        self._point = numpy.array(point, copy=True, dtype=numpy.float32)
        assert len(normal) == 3
        self._normal = numpy.array(normal, copy=True, dtype=numpy.float32)
        self.notify()

    def setPlane(self, point=None, normal=None):
        """Set plane point and normal and notify.

        :param point: A point on the plane.
        :type point: 3-tuple of float or None.
        :param normal: Normal of the plane.
        :type normal: 3-tuple of float or None.
        """
        planechanged = False

        if point is not None:
            assert len(point) == 3
            point = numpy.array(point, copy=True, dtype=numpy.float32)
            if not numpy.all(numpy.equal(self._point, point)):
                self._point = point
                planechanged = True

        if normal is not None:
            assert len(normal) == 3
            normal = numpy.array(normal, copy=True, dtype=numpy.float32)

            norm = numpy.linalg.norm(normal)
            if norm != 0.:
                normal /= norm

            if not numpy.all(numpy.equal(self._normal, normal)):
                self._normal = normal
                planechanged = True

        if planechanged:
            _logger.debug('Plane updated:\n\tpoint: %s\n\tnormal: %s',
                          str(self._point), str(self._normal))
            self.notify()

    @property
    def point(self):
        """A point on the plane."""
        return self._point.copy()

    @point.setter
    def point(self, point):
        self.setPlane(point=point)

    @property
    def normal(self):
        """The (normalized) normal of the plane."""
        return self._normal.copy()

    @normal.setter
    def normal(self, normal):
        self.setPlane(normal=normal)

    @property
    def parameters(self):
        """Plane equation parameters: a*x + b*y + c*z + d = 0."""
        return numpy.append(self._normal,
                            - numpy.dot(self._point, self._normal))

    @parameters.setter
    def parameters(self, parameters):
        assert len(parameters) == 4
        parameters = numpy.array(parameters, dtype=numpy.float32)

        # Normalize normal
        norm = numpy.linalg.norm(parameters[:3])
        if norm != 0:
            parameters /= norm

        normal = parameters[:3]
        point = - parameters[3] * normal
        self.setPlane(point, normal)

    @property
    def isPlane(self):
        """True if a plane is defined (i.e., ||normal|| != 0)."""
        return numpy.any(self.normal != 0.)

    def move(self, step):
        """Move the plane of step along the normal."""
        self.point += step * self.normal

    def segmentIntersection(self, s0, s1):
        """Compute the plane intersection with segment [s0, s1].

        :param s0: First end of the segment
        :type s0: 1D numpy.ndarray-like of length 3
        :param s1: Second end of the segment
        :type s1: 1D numpy.ndarray-like of length 3
        :return: The intersection points. The number of points goes
                 from 0 (no intersection) to 2 (segment in the plane)
        :rtype: list of 1D numpy.ndarray
        """
        if not self.isPlane:
            return []
        else:
            return segmentPlaneIntersect(s0, s1, self.normal, self.point)
