#!/usr/bin/python

cimport cython
import numpy
cimport numpy
from cython.parallel import prange

cdef class Polygon(object):
    cdef float[:,:] vertices
    cdef int nvert
    def __init__(self, vertices):
        """
        @param vertices: Nx2 array of floats
        """
        self.vertices = numpy.ascontiguousarray(vertices, dtype=numpy.float32)
        self.nvert = vertices.shape[0]

    def isInside(self, px, py, border_value=True):
        return self.c_isInside(px, py, border_value)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint c_isInside(self, float px, float py, bint border_value=True) nogil:
        """
        Pure C_Cython class implementation
        """
        cdef int counter, i
        cdef float polypoint1x, polypoint1y, polypoint2x, polypoint2y, xinters
        counter = 0

        polypoint1x = self.vertices[self.nvert-1, 0]
        polypoint1y = self.vertices[self.nvert-1, 1]
        for i in range(self.nvert):
            if (polypoint1x == px) and (polypoint1y == py):
                return border_value
            polypoint2x = self.vertices[i, 0]
            polypoint2y = self.vertices[i, 1]
            if (py > min(polypoint1y, polypoint2y)):
                if (py <= max(polypoint1y, polypoint2y)):
                    if (px <= max(polypoint1x, polypoint2x)):
                        if (polypoint1y != polypoint2y):
                            xinters = (py - polypoint1y) * (polypoint2x - polypoint1x) / (polypoint2y - polypoint1y) + polypoint1x
                            if (polypoint1x == polypoint2x) or (px <= xinters):
                                counter += 1
            polypoint1x, polypoint1y = polypoint2x, polypoint2y
        if counter % 2 == 0:
            return False
        else:
            return True

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def make_mask(self, int dx, int dy):
        cdef numpy.ndarray[dtype=numpy.uint8_t,ndim=2] mask = numpy.empty((dx,dy),dtype=numpy.uint8)
        cdef int i, j
        for i in prange(dy, nogil=True):
            for j in range(dx):
                mask[i,j] = self.c_isInside(j,i)
        return mask


def make_vertices_np(nr, max_val=1024):
    """
    Generates a set of vertices as nr-tuple of 2-tuple if integers
    """
    return numpy.random.randint(0, max_val, nr * 2).reshape((nr, 2)).astype(numpy.float32)
