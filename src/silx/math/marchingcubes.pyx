# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""This module provides marching cubes implementation.

It provides a :class:`MarchingCubes` class allowing to build an isosurface
from data provided as a 3D data set or slice by slice.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "16/08/2017"


import numpy
cimport numpy as cnumpy
cimport cython

cimport silx.math.mc as mc


# From numpy_common.pxi to avoid warnings while compiling C code
# See this thread:
# https://mail.python.org/pipermail//cython-devel/2012-March/002137.html
cdef extern from *:
    bint FALSE "0"
    void import_array()
    void import_umath()

if FALSE:
    import_array()
    import_umath()


cdef class MarchingCubes:
    """Compute isosurface using marching cubes algorithm.

    It builds a surface from a 3D scalar dataset as a 3D contour at a
    given value.
    The resulting surface is not topologically correct.

    See: http://paulbourke.net/geometry/polygonise/

    Lorensen, W. E. and Cline, H. E. Marching cubes: A high resolution 3D
    surface construction algorithm. Computer Graphics, 21, 4 (July 1987).
    ACM, 163-169.

    Generated vertex and normal coordinates are in the same order
    as input array, i.e., (dim 0, dim 1, dim 2).

    Expected indices in memory of a (2, 2, 2) dataset:

           dim 0 (depth)
             |
             |
           4 +------+ 5
            /|     /|
           / |    / |
        6 +------+ 7|
          |  |   |  |
          |0 +---|--+ 1 --- dim 2 (width)
          | /    | /
          |/     |/
        2 +------+ 3
         /
        /
      dim 1 (height)

    Example with a 3D data set:

    >>> vertices, normals, indices = MarchingCubes(data, isolevel=1.)

    Example of code for processing a list of images:

    >>> mc = MarchingCubes(isolevel=1.)  # Create object with iso-level=1
    >>> previous_image = images[0]
    >>> for image in images[1:]:
    ...     mc.process_image(previous_image, image)  # Process one slice
    ...     previous_image = image

    >>> vertices = mc.get_vertices()  # Array of vertex positions
    >>> normals = mc.get_normals()  # Array of normals
    >>> triangle_indices = mc.get_indices()  # Array of indices of vertices

    :param data: 3D dataset of float32 or None
    :type data: numpy.ndarray of float32 of dimension 3
    :param float isolevel: The value for which to generate the isosurface
    :param bool invert_normals:
        True (default) for normals oriented in direction of gradient descent
    :param sampling: Sampling along each dimension (depth, height, width)
    """
    cdef mc.MarchingCubes[float, float] * c_mc  # Pointer to the C++ instance

    def __cinit__(self, data=None, isolevel=None,
                  invert_normals=True, sampling=(1, 1, 1)):
        self.c_mc = new mc.MarchingCubes[float, float](isolevel)
        self.c_mc.invert_normals = bool(invert_normals)
        self.c_mc.sampling[0] = sampling[0]
        self.c_mc.sampling[1] = sampling[1]
        self.c_mc.sampling[2] = sampling[2]

        if data is not None:
            self.process(data)

    def __dealloc__(self):
        del self.c_mc

    def __getitem__(self, key):
        """Allows one to unpack object as a single liner:

        vertices, normals, indices = MarchingCubes(...)
        """
        if key == 0:
            return self.get_vertices()
        elif key == 1:
            return self.get_normals()
        elif key == 2:
            return self.get_indices()
        else:
            raise IndexError("Index out of range")

    def process(self, data):
        """Compute an isosurface from a 3D scalar field.

        This builds vertices, normals and indices arrays.
        Vertices and normals coordinates are in the same order as input array,
        i.e., (dim 0, dim 1, dim 2).

        :param numpy.ndarray data: 3D scalar field
        """
        # Make sure data is a 3D contiguous array of native endian float32
        data = numpy.ascontiguousarray(data, dtype='=f4')
        assert data.ndim == 3
        cdef float[:] c_data = numpy.ravel(data)
        cdef unsigned int depth, height, width

        depth = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]

        self.c_mc.process(&c_data[0], depth, height, width)

    def process_slice(self, slice0, slice1):
        """Process a new slice to build the isosurface.

        :param numpy.ndarray slice0: Slice previously provided as slice1.
        :param numpy.ndarray slice1: Slice to process.
        """
        # Make sure slices are 2D contiguous arrays of native endian float32
        slice0 = numpy.ascontiguousarray(slice0, dtype='=f4')
        assert slice0.ndim == 2
        slice1 = numpy.ascontiguousarray(slice1, dtype='=f4')
        assert slice1.ndim == 2

        assert slice0.shape[0] == slice1.shape[0]
        assert slice0.shape[1] == slice1.shape[1]

        cdef float[:] c_slice0 = numpy.ravel(slice0)
        cdef float[:] c_slice1 = numpy.ravel(slice1)

        if self.c_mc.depth == 0:
            # Starts a new isosurface, bootstrap with slice size
            self.c_mc.set_slice_size(slice1.shape[0], slice1.shape[1])

        assert slice1.shape[0] == self.c_mc.height
        assert slice1.shape[1] == self.c_mc.width

        self.c_mc.process_slice(&c_slice0[0], &c_slice1[0])

    def finish_process(self):
        """Clear internal cache after processing slice by slice."""
        self.c_mc.finish_process()

    def reset(self):
        """Reset internal resources including computed isosurface info."""
        self.c_mc.reset()

    @cython.embedsignature(False)
    @property
    def shape(self):
        """The shape of the processed scalar field (depth, height, width)."""
        return self.c_mc.depth, self.c_mc.height, self.c_mc.width

    @cython.embedsignature(False)
    @property
    def sampling(self):
        """The sampling over each dimension (depth, height, width).

        Default: 1, 1, 1
        """
        return (self.c_mc.sampling[0],
                self.c_mc.sampling[1],
                self.c_mc.sampling[2])

    @cython.embedsignature(False)
    @property
    def isolevel(self):
        """The iso-level at which to generate the isosurface"""
        return self.c_mc.isolevel

    @cython.embedsignature(False)
    @property
    def invert_normals(self):
        """True to use gradient descent as normals."""
        return self.c_mc.invert_normals

    def get_vertices(self):
        """Vertices currently computed (ndarray of dim NbVertices x 3)

        Order is dim0, dim1, dim2 (i.e., z, y, x if dim0 is depth).
        """
        return numpy.array(self.c_mc.vertices).reshape(-1, 3)

    def get_normals(self):
        """Normals currently computed (ndarray of dim NbVertices x 3)

        Order is dim0, dim1, dim2 (i.e., z, y, x if dim0 is depth).
        """
        return numpy.array(self.c_mc.normals).reshape(-1, 3)

    def get_indices(self):
        """Triangle indices currently computed (ndarray of dim NbTriangles x 3)
        """
        return numpy.array(self.c_mc.indices,
                           dtype=numpy.uint32).reshape(-1, 3)
