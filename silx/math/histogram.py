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


Classes
=======

- :class:`Histogramnd` : multi dimensional histogram.
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

Computing the histogram with Histogramnd :

>>> from silx.math import Histogramnd
>>> n_bins = 35
>>> ranges = [[40., 150.], [-130., 250.], [0., 505]]
>>> histo, w_histo, edges = Histogramnd(sample, n_bins=n_bins, histo_range=ranges, weights=weights)

Histogramnd can accumulate sets of data that don't have the same
coordinates :

>>> from silx.math import Histogramnd
>>> histo_obj = Histogramnd(sample, n_bins=n_bins, histo_range=ranges, weights=weights)
>>> sample_2 = np.random.random(shape) * 200
>>> weights_2 = np.random.random((shape[0],))
>>> histo_obj.accumulate(sample_2, weights=weights_2)

And then access the results:

>>> histo = histo_obj.histo
>>> weighted_histo = histo_obj.weighted_histo

or even:

>>> histo, w_histo, edges = histo_obj

Accumulating histograms (LUT)
-----------------------------
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

Bin edges
---------
When computing an histogram the caller is asked to provide the histogram
range along each coordinates (parameter *histo_range*). This parameter must
be given a [N, 2] array where N is the number of dimensions of the histogram.

In other words, the caller must provide, for each dimension,
the left edge of the first (*leftmost*) bin, and the right edge of the
last (*rightmost*) bin.

E.g. : for a 1D sample, for a histo_range equal to [0, 10] and n_bins=4, the
bins ranges will be :

* [0, 2.5[, [2.5, 5[, [5, 7.5[, [7.5, 10 **[** if last_bin_closed = **False**
* [0, 2.5[, [2.5, 5[, [5, 7.5[, [7.5, 10 **]** if last_bin_closed = **True**

....
"""

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "02/10/2017"

import numpy as np
from .chistogramnd import chistogramnd as _chistogramnd  # noqa
from .chistogramnd_lut import histogramnd_get_lut as _histo_get_lut
from .chistogramnd_lut import histogramnd_from_lut as _histo_from_lut


class Histogramnd(object):
    """
    Computes the multidimensional histogram of some data.
    """

    def __init__(self,
                 sample,
                 histo_range,
                 n_bins,
                 weights=None,
                 weight_min=None,
                 weight_max=None,
                 last_bin_closed=False,
                 wh_dtype=None):
        """
        :param sample:
            The data to be histogrammed.
            Its shape must be either
            (N,) if it contains one dimensional coordinates,
            or an (N,D) array where the rows are the
            coordinates of points in a D dimensional space.
            The following dtypes are supported : :class:`numpy.float64`,
            :class:`numpy.float32`, :class:`numpy.int32`.

            .. warning:: if sample is not a C_CONTIGUOUS ndarray (e.g : a non
                contiguous slice) then histogramnd will have to do make an internal
                copy.
        :type sample: :class:`numpy.array`

        :param histo_range:
            A (N, 2) array containing the histogram range along each dimension,
            where N is the sample's number of dimensions.
        :type histo_range: array_like

        :param n_bins:
            The number of bins :
                * a scalar (same number of bins for all dimensions)
                * a D elements array (number of bins for each dimensions)
        :type n_bins: scalar or array_like

        :param weights:
            A N elements numpy array of values associated with
            each sample.
            The values of the *weighted_histo* array
            returned by the function are equal to the sum of
            the weights associated with the samples falling
            into each bin.
            The following dtypes are supported : :class:`numpy.float64`,
            :class:`numpy.float32`, :class:`numpy.int32`.

            .. note:: If None, the weighted histogram returned will be None.
        :type weights: *optional*, :class:`numpy.array`

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

        :param last_bin_closed:
            By default the last bin is half
            open (i.e.: [x,y) ; x included, y
            excluded), like all the other bins.
            Set this parameter to true if you want
            the LAST bin to be closed.
        :type last_bin_closed: *optional*, :class:`python.boolean`

        :param wh_dtype: type of the weighted histogram array.
            If not provided, the weighted histogram array will contain values
            of type numpy.double. Allowed values are : `numpy.double` and
            `numpy.float32`
        :type wh_dtype: *optional*, numpy data type
        """

        self.__histo_range = histo_range
        self.__n_bins = n_bins
        self.__last_bin_closed = last_bin_closed
        self.__wh_dtype = wh_dtype

        if sample is None:
            self.__data = [None, None, None]
        else:
            self.__data = _chistogramnd(sample,
                                        self.__histo_range,
                                        self.__n_bins,
                                        weights=weights,
                                        weight_min=weight_min,
                                        weight_max=weight_max,
                                        last_bin_closed=self.__last_bin_closed,
                                        wh_dtype=self.__wh_dtype)

    def __getitem__(self, key):
        """
        If necessary, results can be unpacked from an instance of Histogramnd :
        *histogram*, *weighted histogram*, *bins edge*.

        Example :

        .. code-block:: python

            histo, w_histo, edges = Histogramnd(sample, histo_range, n_bins, weights)

        """
        return self.__data[key]

    def accumulate(self,
                   sample,
                   weights=None,
                   weight_min=None,
                   weight_max=None):
        """
        Computes the multidimensional histogram of some data and accumulates it
        into the histogram held by this instance of Histogramnd.

        :param sample:
            The data to be histogrammed.
            Its shape must be either
            (N,) if it contains one dimensional coordinates,
            or an (N,D) array where the rows are the
            coordinates of points in a D dimensional space.
            The following dtypes are supported : :class:`numpy.float64`,
            :class:`numpy.float32`, :class:`numpy.int32`.

            .. warning:: if sample is not a C_CONTIGUOUS ndarray (e.g : a non
                contiguous slice) then histogramnd will have to do make an internal
                copy.
        :type sample: :class:`numpy.array`

        :param weights:
            A N elements numpy array of values associated with
            each sample.
            The values of the *weighted_histo* array
            returned by the function are equal to the sum of
            the weights associated with the samples falling
            into each bin.
            The following dtypes are supported : :class:`numpy.float64`,
            :class:`numpy.float32`, :class:`numpy.int32`.

            .. note:: If None, the weighted histogram returned will be None.
        :type weights: *optional*, :class:`numpy.array`

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
        result = _chistogramnd(sample,
                               self.__histo_range,
                               self.__n_bins,
                               weights=weights,
                               weight_min=weight_min,
                               weight_max=weight_max,
                               last_bin_closed=self.__last_bin_closed,
                               histo=self.__data[0],
                               weighted_histo=self.__data[1],
                               wh_dtype=self.__wh_dtype)
        if self.__data[0] is None:
            self.__data = result
        elif self.__data[1] is None and result[1] is not None:
            self.__data = result

    histo = property(lambda self: self[0])
    """ Histogram array, or None if this instance was initialized without
        <sample> and accumulate has not been called yet.

        .. note:: this is a **reference** to the array store in this
             Histogramnd instance, use with caution.
    """
    weighted_histo = property(lambda self: self[1])
    """ Weighted Histogram, or None if this instance was initialized without
        <sample>, or no weights have been passed to __init__ nor accumulate.

        .. note:: this is a **reference** to the array store in this
            Histogramnd instance, use with caution.
    """
    edges = property(lambda self: self[2])
    """ Bins edges, or None if this instance was initialized without
        <sample> and accumulate has not been called yet.
    """


class HistogramndLut(object):
    """
    The HistogramndLut class allows you to bin data onto a regular grid.
    The use of HistogramndLut is interesting when several sets of data that
    share the same coordinates (*sample*) have to be mapped onto the same grid.
    """

    def __init__(self,
                 sample,
                 histo_range,
                 n_bins,
                 last_bin_closed=False,
                 dtype=None):
        """
        :param sample:
            The coordinates of the data to be histogrammed.
            Its shape must be either (N,) if it contains one dimensional
            coordinates, or an (N, D) array where the rows are the
            coordinates of points in a D dimensional space.
            The following dtypes are supported : :class:`numpy.float64`,
            :class:`numpy.float32`, :class:`numpy.int32`.
        :type sample: :class:`numpy.array`

        :param histo_range:
            A (N, 2) array containing the histogram range along each dimension,
            where N is the sample's number of dimensions.
        :type histo_range: array_like

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
        lut, histo, edges = _histo_get_lut(sample,
                                           histo_range,
                                           n_bins,
                                           last_bin_closed=last_bin_closed)

        self.__n_bins = np.array(histo.shape)
        self.__histo_range = histo_range
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

    def histo(self, copy=True):
        """
        Histogram (a copy of it), or None if `~accumulate` has not been called yet
        (or clear was just called).
        If *copy* is set to False then the actual reference to the array is
        returned *(use with caution)*.
        """
        if copy and self.__histo is not None:
            return self.__histo.copy()
        return self.__histo

    def weighted_histo(self, copy=True):
        """
        Weighted histogram (a copy of it), or None if `~accumulate` has not been called yet
        (or clear was just called). If *copy* is set to False then the actual
        reference to the array is returned *(use with caution)*.
        """
        if copy and self.__weighted_histo is not None:
            return self.__weighted_histo.copy()
        return self.__weighted_histo

    @property
    def histo_range(self):
        """
        Bins ranges.
        """
        return self.__histo_range.copy()

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
        :type histo_range: array_like

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
        :type histo_range: array_like

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
