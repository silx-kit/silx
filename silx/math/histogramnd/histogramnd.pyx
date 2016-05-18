# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/02/2016"

cimport numpy
cimport cython
import numpy as np

cimport histogramnd_c


# =====================
#  double sample, double cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_double_double(double[:] sample,
                                           double[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] bins_rng,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           double[:] cumul,
                                           int option_flags,
                                           double weight_min,
                                           double weight_max) nogil:

    return histogramnd_c.histogramnd_double_double_double(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &bins_rng[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_float_double(double[:] sample,
                                          float[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] bins_rng,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          double[:] cumul,
                                          int option_flags,
                                          float weight_min,
                                          float weight_max) nogil:

    return histogramnd_c.histogramnd_double_float_double(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &bins_rng[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_int32_t_double(double[:] sample,
                                            numpy.int32_t[:] weights,
                                            int n_dims,
                                            int n_elem,
                                            double[:] bins_rng,
                                            int[:] n_bins,
                                            numpy.uint32_t[:] histo,
                                            double[:] cumul,
                                            int option_flags,
                                            numpy.int32_t weight_min,
                                            numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_double_int32_t_double(&sample[0],
                                                           &weights[0],
                                                           n_dims,
                                                           n_elem,
                                                           &bins_rng[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)


# =====================
#  float sample, double cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_double_double(float[:] sample,
                                          double[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          float[:] bins_rng,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          double[:] cumul,
                                          int option_flags,
                                          double weight_min,
                                          double weight_max) nogil:

    return histogramnd_c.histogramnd_float_double_double(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &bins_rng[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_float_double(float[:] sample,
                                         float[:] weights,
                                         int n_dims,
                                         int n_elem,
                                         float[:] bins_rng,
                                         int[:] n_bins,
                                         numpy.uint32_t[:] histo,
                                         double[:] cumul,
                                         int option_flags,
                                         float weight_min,
                                         float weight_max) nogil:

    return histogramnd_c.histogramnd_float_float_double(&sample[0],
                                                        &weights[0],
                                                        n_dims,
                                                        n_elem,
                                                        &bins_rng[0],
                                                        &n_bins[0],
                                                        &histo[0],
                                                        &cumul[0],
                                                        option_flags,
                                                        weight_min,
                                                        weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_int32_t_double(float[:] sample,
                                           numpy.int32_t[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           float[:] bins_rng,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           double[:] cumul,
                                           int option_flags,
                                           numpy.int32_t weight_min,
                                           numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_float_int32_t_double(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &bins_rng[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


# =====================
#  numpy.int32_t sample, double cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_double_double(numpy.int32_t[:] sample,
                                            double[:] weights,
                                            int n_dims,
                                            int n_elem,
                                            numpy.int32_t[:] bins_rng,
                                            int[:] n_bins,
                                            numpy.uint32_t[:] histo,
                                            double[:] cumul,
                                            int option_flags,
                                            double weight_min,
                                            double weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_double_double(&sample[0],
                                                           &weights[0],
                                                           n_dims,
                                                           n_elem,
                                                           &bins_rng[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_float_double(numpy.int32_t[:] sample,
                                           float[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           numpy.int32_t[:] bins_rng,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           double[:] cumul,
                                           int option_flags,
                                           float weight_min,
                                           float weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_float_double(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &bins_rng[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_int32_t_double(numpy.int32_t[:] sample,
                                             numpy.int32_t[:] weights,
                                             int n_dims,
                                             int n_elem,
                                             numpy.int32_t[:] bins_rng,
                                             int[:] n_bins,
                                             numpy.uint32_t[:] histo,
                                             double[:] cumul,
                                             int option_flags,
                                             numpy.int32_t weight_min,
                                             numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_int32_t_double(&sample[0],
                                                            &weights[0],
                                                            n_dims,
                                                            n_elem,
                                                            &bins_rng[0],
                                                            &n_bins[0],
                                                            &histo[0],
                                                            &cumul[0],
                                                            option_flags,
                                                            weight_min,
                                                            weight_max)


# =====================
#  double sample, float cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_double_float(double[:] sample,
                                          double[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] bins_rng,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          float[:] cumul,
                                          int option_flags,
                                          double weight_min,
                                          double weight_max) nogil:

    return histogramnd_c.histogramnd_double_double_float(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &bins_rng[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_float_float(double[:] sample,
                                         float[:] weights,
                                         int n_dims,
                                         int n_elem,
                                         double[:] bins_rng,
                                         int[:] n_bins,
                                         numpy.uint32_t[:] histo,
                                         float[:] cumul,
                                         int option_flags,
                                         float weight_min,
                                         float weight_max) nogil:

    return histogramnd_c.histogramnd_double_float_float(&sample[0],
                                                        &weights[0],
                                                        n_dims,
                                                        n_elem,
                                                        &bins_rng[0],
                                                        &n_bins[0],
                                                        &histo[0],
                                                        &cumul[0],
                                                        option_flags,
                                                        weight_min,
                                                        weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_int32_t_float(double[:] sample,
                                           numpy.int32_t[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] bins_rng,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           float[:] cumul,
                                           int option_flags,
                                           numpy.int32_t weight_min,
                                           numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_double_int32_t_float(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &bins_rng[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


# =====================
#  float sample, float cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_double_float(float[:] sample,
                                         double[:] weights,
                                         int n_dims,
                                         int n_elem,
                                         float[:] bins_rng,
                                         int[:] n_bins,
                                         numpy.uint32_t[:] histo,
                                         float[:] cumul,
                                         int option_flags,
                                         double weight_min,
                                         double weight_max) nogil:

    return histogramnd_c.histogramnd_float_double_float(&sample[0],
                                                        &weights[0],
                                                        n_dims,
                                                        n_elem,
                                                        &bins_rng[0],
                                                        &n_bins[0],
                                                        &histo[0],
                                                        &cumul[0],
                                                        option_flags,
                                                        weight_min,
                                                        weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_float_float(float[:] sample,
                                        float[:] weights,
                                        int n_dims,
                                        int n_elem,
                                        float[:] bins_rng,
                                        int[:] n_bins,
                                        numpy.uint32_t[:] histo,
                                        float[:] cumul,
                                        int option_flags,
                                        float weight_min,
                                        float weight_max) nogil:

    return histogramnd_c.histogramnd_float_float_float(&sample[0],
                                                       &weights[0],
                                                       n_dims,
                                                       n_elem,
                                                       &bins_rng[0],
                                                       &n_bins[0],
                                                       &histo[0],
                                                       &cumul[0],
                                                       option_flags,
                                                       weight_min,
                                                       weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_int32_t_float(float[:] sample,
                                          numpy.int32_t[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          float[:] bins_rng,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          float[:] cumul,
                                          int option_flags,
                                          numpy.int32_t weight_min,
                                          numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_float_int32_t_float(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &bins_rng[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


# =====================
#  numpy.int32_t sample, float cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_double_float(numpy.int32_t[:] sample,
                                           double[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           numpy.int32_t[:] bins_rng,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           float[:] cumul,
                                           int option_flags,
                                           double weight_min,
                                           double weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_double_float(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &bins_rng[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_float_float(numpy.int32_t[:] sample,
                                          float[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          numpy.int32_t[:] bins_rng,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          float[:] cumul,
                                          int option_flags,
                                          float weight_min,
                                          float weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_float_float(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &bins_rng[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_int32_t_float(numpy.int32_t[:] sample,
                                            numpy.int32_t[:] weights,
                                            int n_dims,
                                            int n_elem,
                                            numpy.int32_t[:] bins_rng,
                                            int[:] n_bins,
                                            numpy.uint32_t[:] histo,
                                            float[:] cumul,
                                            int option_flags,
                                            numpy.int32_t weight_min,
                                            numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_int32_t_float(&sample[0],
                                                           &weights[0],
                                                           n_dims,
                                                           n_elem,
                                                           &bins_rng[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)
