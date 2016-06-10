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


"""
This module provides a function and a class to compute multidimensional
histograms.

Function
========
- :func:`histogramnd`

Class
=====

- :class:`HistogramndLut` : optimized to compute several histograms from data sharing the same coordinates.

Examples
========

Single histogram
----------------

Given some 3D data:

>>> import numpy as np
>>> shape = (10**7, 3)
>>> sample = np.random.random(shape) * 500
>>> weights = np.random.random((shape[0],))

Computing the histogram with histogramnd :

>>> from silx.math import histogramnd
>>> n_bins = 35
>>> ranges = [[40., 150.], [-130., 250.], [0., 505]]
>>> histo, w_histo, edges = histogramnd(sample, n_bins=n_bins, bins_rng=ranges, weights=weights)

Accumulating histograms
-----------------------
In some situations we need to compute the weighted histogram of several
sets of data (weights) that have the same coordinates (sample).

Again, some data (2 sets of weights) :

>>> import numpy as np
>>> shape = (10**7, 3)
>>> sample = np.random.random(shape) * 500
>>> weights_1 = np.random.random((shape[0],))
>>> weights_2 = np.random.random((shape[0],))

And getting the result with HistogramLut :

>>> from silx.math import HistogramndLut

>>> n_bins = 35
>>> ranges = [[40., 150.], [-130., 250.], [0., 505]]

>>> histo_lut = HistogramndLut(sample, ranges, n_bins)
                           
First call, with weight_1 :

>>> histo_lut.accumulate(weights_1)

Second call, with weight_2 :

>>> histo_lut.accumulate(weights_2)

Retrieving the results (this is a copy of what's actually stored in
this instance) :

>>> histo = histo_lut.histo
>>> w_histo = histo_lut.weighted_histo

Note that the following code gives the same result, but the
HistogramndLut instance does not store the accumulated weighted histogram.

First call with weights_1

>>> histo, w_histo = histo_lut.apply_lut(weights_1)

Second call with weights_2

>>> histo, w_histo = histo_lut.apply_lut(weights_2, histo=histo, weighted_histo=w_histo)

....
"""  # noqa

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "15/05/2016"

import numpy as np
from .chistogramnd import chistogramnd as histogramnd  # noqa
from .chistogramnd_lut import histogramnd_get_lut as _histo_get_lut
from .chistogramnd_lut import histogramnd_from_lut as _histo_from_lut


class HistogramndLut(object):
    """
    ``HistogramndLut(sample, bins_rng, n_bins, last_bin_closed=True)``
    The HistogramndLut class allows you to bin data onto a regular grid.
    The use of HistogramndLut is interesting when several sets of data that
    share the same coordinates (*sample*) have to be mapped onto the same grid.

    :param sample:
        The coordinates of the data to be histogrammed.
        Its shape must be either (N,) if it contains one dimensional
        coordinates, or an (N, D) array where the rows are the
        coordinates of points in a D dimensional space.
        The following dtypes are supported : :class:`numpy.float64`,
        :class:`numpy.float32`, :class:`numpy.int32`.
    :type sample: :class:`numpy.array`

    :param bins_rng:
        A (N, 2) array containing the lower and upper
        bin edges along each dimension.
    :type bins_rng: array_like

    :param n_bins:
        The number of bins :
            * a scalar (same number of bins for all dimensions)
            * a D elements array (number of bins for each dimensions)
    :type n_bins: scalar or array_like

    :param dtype: data type of the weighted histogram. If None, the data type
        will be the same as the first weights array provided (on first call of
        the instance).
    :type dtype: `numpy.dtype`

    :param last_bin_closed:
        By default the last bin is half
        open (i.e.: [x,y) ; x included, y
        excluded), like all the other bins.
        Set this parameter to true if you want
        the LAST bin to be closed.
    :type last_bin_closed: *optional*, :class:`python.boolean`
    """
    def __init__(self,
                 sample,
                 bins_rng,
                 n_bins,
                 last_bin_closed=False,
                 dtype=None):
        lut, histo, edges = _histo_get_lut(sample,
                                           bins_rng,
                                           n_bins,
                                           last_bin_closed=last_bin_closed)

        self.__n_bins = np.array(histo.shape)
        self.__bins_rng = bins_rng
        self.__lut = lut
        self.__histo = None
        self.__weighted_histo = None
        self.__edges = edges
        self.__dtype = dtype
        self.__shape = histo.shape
        self.__last_bin_closed = last_bin_closed
        self.clear()

    def clear(self):
        """
        Resets the instance (zeroes the histograms).
        """
        self.__weighted_histo = None
        self.__histo = None

    @property
    def lut(self):
        """
        Copy of the Lut
        """
        return self.__lut.copy()

    @property
    def histo(self):
        """
        Histogram (actualy a *copy* of the one stored in this instance),
        or None if `~accumulate` has not been called yet (or clear was just
        called).
        """
        histo = self.histo_ref
        if histo is not None:
            return histo.copy()
        return histo

    @property
    def histo_ref(self):
        """
        Same as `~histo`, but returns a reference (i.e : not a copy) to the
        histogram stored by this HistogramndLut instance *(use with caution)*.
        """
        if self.__histo is not None:
            return self.__histo
        else:
            return None

    @property
    def weighted_histo(self):
        """
        Weighted histogram (actualy a *copy* of the one stored in this
        instance), or None if `~accumulate` has not been called yet (or clear
        was just called).
        """
        w_histo = self.weighted_histo_ref
        if w_histo is not None:
            return w_histo.copy()
        return w_histo

    @property
    def weighted_histo_ref(self):
        """
        Same as `~weighted_histo`, but returns a reference (i.e : not a copy)
        to the weighted histogram stored by this HistogramndLut instance
        *(use with caution)*.
        """
        if self.__weighted_histo is not None:
            return self.__weighted_histo
        else:
            return None

    @property
    def bins_rng(self):
        """
        Bins ranges.
        """
        return self.__bins_rng.copy()

    @property
    def n_bins(self):
        """
        Number of bins in each direction.
        """
        return self.__n_bins.copy()

    @property
    def bins_edges(self):
        """
        Bins edges of the histograms, one array for each dimensions.
        """
        return tuple([edges[:] for edges in self.__edges])

    @property
    def last_bin_closed(self):
        """
        Returns True if the rightmost bin in each dimension is close (i.e :
        values equal to the rightmost bin edge is included in the bin).
        """
        return self.__last_bin_closed

    def accumulate(self,
                   weights,
                   weight_min=None,
                   weight_max=None):
        """
        Computes the multidimensional histogram of some data and adds it to
        the current histogram stored by this instance. The results can be
        retrieved with the :attr:`~.histo` and :attr:`~.weighted_histo`
        properties.

        :param weights:
            A numpy array of values associated with each sample. The number of
            elements in the array must be the same as the number of samples
            provided at instantiation time.
        :type bins_rng: array_like

        :param weight_min:
            Use this parameter to filter out all samples whose
            weights are lower than this value.

            .. note:: This value will be cast to the same type
                as *weights*.
        :type weight_min: *optional*, scalar

        :param weight_max:
            Use this parameter to filter out all samples whose
            weights are higher than this value.

            .. note:: This value will be cast to the same type
                as *weights*.

        :type weight_max: *optional*, scalar
        """
        if self.__dtype is None:
            self.__dtype = weights.dtype

        histo, w_histo = _histo_from_lut(weights,
                                         self.__lut,
                                         histo=self.__histo,
                                         weighted_histo=self.__weighted_histo,
                                         shape=self.__shape,
                                         dtype=self.__dtype,
                                         weight_min=weight_min,
                                         weight_max=weight_max)

        if self.__histo is None:
            self.__histo = histo

        if self.__weighted_histo is None:
            self.__weighted_histo = w_histo

    def apply_lut(self,
                  weights,
                  histo=None,
                  weighted_histo=None,
                  weight_min=None,
                  weight_max=None):
        """
        Computes the multidimensional histogram of some data and returns the
        result (it is NOT added to the current histogram stored by this
        instance).

        :param weights:
            A numpy array of values associated with each sample. The number of
            elements in the array must be the same as the number of samples
            provided at instantiation time.
        :type bins_rng: array_like

        :param histo:
            Use this parameter if you want to pass your
            own histogram array instead of the one
            created by this function. New values
            will be added to this array. The returned array
            will then be this one.
        :type histo: *optional*, :class:`numpy.array`

        :param weighted_histo:
            Use this parameter if you want to pass your
            own weighted histogram array instead of
            the created by this function. New
            values will be added to this array. The returned array
            will then be this one (same reference).
        :type weighted_histo: *optional*, :class:`numpy.array`

        :param weight_min:
            Use this parameter to filter out all samples whose
            weights are lower than this value.

            .. note:: This value will be cast to the same type
                as *weights*.
        :type weight_min: *optional*, scalar

        :param weight_max:
            Use this parameter to filter out all samples whose
            weights are higher than this value.

            .. note:: This value will be cast to the same type
                as *weights*.
        :type weight_max: *optional*, scalar
        """
        histo, w_histo = _histo_from_lut(weights,
                                         self.__lut,
                                         histo=histo,
                                         weighted_histo=weighted_histo,
                                         shape=self.__shape,
                                         dtype=self.__dtype,
                                         weight_min=weight_min,
                                         weight_max=weight_max)
        self.__dtype = w_histo.dtype
        return histo, w_histo

if __name__ == '__main__':
    pass
