"""
Simple N-D interpolation

.. versionadded:: 0.9

"""
#
# Copyright (C)  Pauli Virtanen, 2010.
#
# Distributed under the same BSD license as Scipy.
#

#
# Note: this file should be run through the Mako template engine before
#       feeding it to Cython.
#
#       Run ``generate_qhull.py`` to regenerate the ``qhull.c`` file
#

from __future__ import absolute_import

cimport cython

from libc.float cimport DBL_EPSILON
from libc.math cimport fabs, sqrt

import numpy as np

import silx.third_party._local.scipy.qhull as qhull
cimport silx.third_party._local.scipy.qhull as qhull

import warnings

#------------------------------------------------------------------------------
# Numpy etc.
#------------------------------------------------------------------------------

cdef extern from "numpy/ndarrayobject.h":
    cdef enum:
        NPY_MAXDIMS

ctypedef fused double_or_complex:
    double
    double complex


#------------------------------------------------------------------------------
# Interpolator base class
#------------------------------------------------------------------------------

class NDInterpolatorBase(object):
    """
    Common routines for interpolators.

    .. versionadded:: 0.9

    """

    def __init__(self, points, values, fill_value=np.nan, ndim=None,
                 rescale=False, need_contiguous=True, need_values=True):
        """
        Check shape of points and values arrays, and reshape values to
        (npoints, nvalues).  Ensure the `points` and values arrays are
        C-contiguous, and of correct type.
        """

        if isinstance(points, qhull.Delaunay):
            # Precomputed triangulation was passed in
            if rescale:
                raise ValueError("Rescaling is not supported when passing "
                                 "a Delaunay triangulation as ``points``.")
            self.tri = points
            points = points.points
        else:
            self.tri = None

        points = _ndim_coords_from_arrays(points)
        values = np.asarray(values)

        _check_init_shape(points, values, ndim=ndim)

        if need_contiguous:
            points = np.ascontiguousarray(points, dtype=np.double)

        if need_values:
            self.values_shape = values.shape[1:]
            if values.ndim == 1:
                self.values = values[:,None]
            elif values.ndim == 2:
                self.values = values
            else:
                self.values = values.reshape(values.shape[0],
                                             np.prod(values.shape[1:]))

            # Complex or real?
            self.is_complex = np.issubdtype(self.values.dtype, np.complexfloating)
            if self.is_complex:
                if need_contiguous:
                    self.values = np.ascontiguousarray(self.values, dtype=np.complex)
                self.fill_value = complex(fill_value)
            else:
                if need_contiguous:
                    self.values = np.ascontiguousarray(self.values, dtype=np.double)
                self.fill_value = float(fill_value)

        if not rescale:
            self.scale = None
            self.points = points
        else:
            # scale to unit cube centered at 0
            self.offset = np.mean(points, axis=0)
            self.points = points - self.offset
            self.scale = self.points.ptp(axis=0)
            self.scale[~(self.scale > 0)] = 1.0  # avoid division by 0
            self.points /= self.scale

    def _check_call_shape(self, xi):
        xi = np.asanyarray(xi)
        if xi.shape[-1] != self.points.shape[1]:
            raise ValueError("number of dimensions in xi does not match x")
        return xi

    def _scale_x(self, xi):
        if self.scale is None:
            return xi
        else:
            return (xi - self.offset) / self.scale

    def __call__(self, *args):
        """
        interpolator(xi)

        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        """
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        shape = xi.shape
        xi = xi.reshape(-1, shape[-1])
        xi = np.ascontiguousarray(xi, dtype=np.double)

        xi = self._scale_x(xi)
        if self.is_complex:
            r = self._evaluate_complex(xi)
        else:
            r = self._evaluate_double(xi)

        return np.asarray(r).reshape(shape[:-1] + self.values_shape)


cpdef _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.

    """
    cdef ssize_t j, n

    if isinstance(points, tuple) and len(points) == 1:
        # handle argument tuple
        points = points[0]
    if isinstance(points, tuple):
        p = np.broadcast_arrays(*points)
        n = len(p)
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        points = np.empty(p[0].shape + (len(points),), dtype=float)
        for j, item in enumerate(p):
            points[...,j] = item
    else:
        points = np.asanyarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points


cdef _check_init_shape(points, values, ndim=None):
    """
    Check shape of points and values arrays

    """
    if values.shape[0] != points.shape[0]:
        raise ValueError("different number of values and points")
    if points.ndim != 2:
        raise ValueError("invalid shape for input data points")
    if points.shape[1] < 2:
        raise ValueError("input data must be at least 2-D")
    if ndim is not None and points.shape[1] != ndim:
        raise ValueError("this mode of interpolation available only for "
                         "%d-D data" % ndim)


#------------------------------------------------------------------------------
# Linear interpolation in N-D
#------------------------------------------------------------------------------

class LinearNDInterpolator(NDInterpolatorBase):
    """
    LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)

    Piecewise linear interpolant in N dimensions.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims); or Delaunay
        Data point coordinates, or a precomputed Delaunay triangulation.
    values : ndarray of float or complex, shape (npoints, ...)
        Data values.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then
        the default is ``nan``.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

    Notes
    -----
    The interpolant is constructed by triangulating the input data
    with Qhull [1]_, and on each triangle performing linear
    barycentric interpolation.

    References
    ----------
    .. [1] http://www.qhull.org/

    """

    def __init__(self, points, values, fill_value=np.nan, rescale=False):
        NDInterpolatorBase.__init__(self, points, values, fill_value=fill_value,
                rescale=rescale)
        if self.tri is None:
            self.tri = qhull.Delaunay(self.points)

    def _evaluate_double(self, xi):
        return self._do_evaluate(xi, 1.0)

    def _evaluate_complex(self, xi):
        return self._do_evaluate(xi, 1.0j)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _do_evaluate(self, double[:,::1] xi, double_or_complex dummy):
        cdef double_or_complex[:,::1] values = self.values
        cdef double_or_complex[:,::1] out
        cdef double[:,::1] points = self.points
        cdef int[:,::1] simplices = self.tri.simplices
        cdef double c[NPY_MAXDIMS]
        cdef double_or_complex fill_value
        cdef int i, j, k, m, ndim, isimplex, inside, start, nvalues
        cdef qhull.DelaunayInfo_t info
        cdef double eps, eps_broad

        ndim = xi.shape[1]
        start = 0
        fill_value = self.fill_value

        qhull._get_delaunay_info(&info, self.tri, 1, 0, 0)

        out = np.zeros((xi.shape[0], self.values.shape[1]),
                       dtype=self.values.dtype)
        nvalues = out.shape[1]

        eps = 100 * DBL_EPSILON
        eps_broad = sqrt(DBL_EPSILON)

        with nogil:
            for i in xrange(xi.shape[0]):

                # 1) Find the simplex

                isimplex = qhull._find_simplex(&info, c,
                                               &xi[0,0] + i*ndim,
                                               &start, eps, eps_broad)

                # 2) Linear barycentric interpolation

                if isimplex == -1:
                    # don't extrapolate
                    for k in xrange(nvalues):
                        out[i,k] = fill_value
                    continue

                for k in xrange(nvalues):
                    out[i,k] = 0

                for j in xrange(ndim+1):
                    for k in xrange(nvalues):
                        m = simplices[isimplex,j]
                        out[i,k] = out[i,k] + c[j] * values[m,k]

        return out
