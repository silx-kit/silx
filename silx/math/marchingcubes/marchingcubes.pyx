# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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
__date__ = "05/09/2016"


import numpy
cimport numpy as cnumpy

cimport mc


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

    Lorensen, W. E. and Cline, H. E. Marching cubes: A high resolution 3D
    surface construction algorithm. Computer Graphics, 21, 4 (July 1987).
    ACM, 163-169.

    Example with a 3D data set:

    >>> v, n, i = MarchingCubes(data, isolevel=1.)

    Example of code for processing a list of images:

    >>> mc = MarchingCubes(isolevel=1.)  # Create object with iso-level=1
    >>> previous_image = images[0]
    >>> for image in images[1:]:
    ...     mc.process_image(previous_image, image)  # Process one slice
    ...     previous_image = image

    >>> vertices = mc.vertices  # Array of vertex positions
    >>> normals = mc.normals  # Array of vertices normal
    >>> triangle_indices = mc.indices  # Array of indices of vertices

    :param data: 3D dataset of float32 or None
    :param isolevel: The value for which to generate the isosurface
    :param bool invert_normals:
        True (default) for normals oriented in direction of gradient descent 
    :param sampling: Sampling along each dimension (depth, height, width)
    """
    cdef mc.MarchingCubes[float] * c_mc  # Pointer to the C++ instance

    def __cinit__(self, data=None, isolevel=None,
                  invert_normals=True, sampling=(1, 1, 1)):
        assert isolevel is not None

        cdef float c_isolevel = isolevel

        self.c_mc = new mc.MarchingCubes[float](c_isolevel)
        self.c_mc.invert_normals = invert_normals
        self.c_mc.sampling[0] = sampling[0]
        self.c_mc.sampling[1] = sampling[1]
        self.c_mc.sampling[2] = sampling[2]

        if data is not None:
            self.process(data)

    def __dealloc__(self):
        del self.c_mc

    def __getitem__(self, key):
        """Allows to unpack object as a single liner:

        vertices, normals, indices = MarchingCubes(...)
        """
        if key == 0:
            return self.vertices
        elif key == 1:
            return self.normals
        elif key == 2:
            return self.indices
        else:
            raise IndexError("Index out of range")

    def process(self, cnumpy.ndarray[cnumpy.float32_t, ndim=3, mode='c'] data):
        """process(data)

        Compute an isosurface from a 3D scalar field.

        :param numpy.ndarray data: 3D scalar field
        :return: Arrays of vertices, normals and triangle indices
        :rtype: tuple of ndarray
        """
        cdef float[:] c_data = numpy.ravel(data)
        cdef unsigned int depth, height, width

        depth = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]

        self.c_mc.process(&c_data[0], depth, height, width)

    def process_slice(self,
            cnumpy.ndarray[cnumpy.float32_t, ndim=2, mode='c'] slice0,
            cnumpy.ndarray[cnumpy.float32_t, ndim=2, mode='c'] slice1):
        """process_slice(slice0, slice1)
        
        Process a new slice to build the isosurface.

        :param numpy.ndarray slice0: Slice previously provided as slice1.
        :param numpy.ndarray slice1: Slice to process.
        """
        assert slice0.shape == slice1.shape

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

    @property
    def shape(self):
        """The shape of the processed scalar field (depth, height, width)."""
        return self.c_mc.depth, self.c_mc.height, self.c_mc.width

    @property
    def sampling(self):
        """The sampling over each dimension (depth, height, width).

        Default: 1, 1, 1
        """
        return (self.c_mc.sampling[0],
                self.c_mc.sampling[1],
                self.c_mc.sampling[2])

    @property
    def isolevel(self):
        """The iso-level at which to generate the isosurface"""
        return self.c_mc.isolevel

    @property
    def invert_normals(self):
        """True to use gradient descent as normals."""
        return self.c_mc.invert_normals

    @property
    def vertices(self):
        """Vertices currently computed (ndarray of dim NbVertices x 3)

        Order is dim0, dim1, dim2 (i.e., z, y, x if dim0 is depth).
        """
        return numpy.array(self.c_mc.vertices).reshape(-1, 3)

    @property
    def normals(self):
        """Normals currently computed (ndarray of dim NbVertices x 3)

        Order is dim0, dim1, dim2 (i.e., z, y, x if dim0 is depth).
        """
        return numpy.array(self.c_mc.normals).reshape(-1, 3)
    
    @property
    def indices(self):
        """Triangle indices currently computed (ndarray of dim NbTriangles x 3)
        """
        return numpy.array(self.c_mc.indices,
                           dtype=numpy.uint32).reshape(-1, 3)
