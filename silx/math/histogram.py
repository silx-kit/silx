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
__date__ = "15/05/2016"

"""
TOP DOC
"""

from .chistogramnd import chistogramnd as _chistogramnd
from .chistogramnd_lut import histogramnd_get_lut as _histo_get_lut
from .chistogramnd_lut import histogramnd_from_lut as _histo_from_lut


def histogramnd(sample,
                bins_rng,
                n_bins,
                weights=None,
                weight_min=None,
                weight_max=None,
                last_bin_closed=False,
                histo=None,
                cumul=None):
    """
    histogramnd(sample, bins_rng, n_bins, weights=None, weight_min=None, weight_max=None, last_bin_closed=False, histo=None, cumul=None)

    Computes the multidimensional histogram of some data.

    :param sample:
        The data to be histogrammed.
        Its shape must be either
        (N,) if it contains one dimensional coordinates,
        or an (N,D) array where the rows are the
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

    :param weights:
        A N elements numpy array of values associated with
        each sample.
        The values of the *cumul* array
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

    :param histo:
        Use this parameter if you want to pass your
        own histogram array instead of the one
        created by this function. New values
        will be added to this array. The returned array
        will then be this one (same reference).

        .. warning:: If the histo array was created by a previous
            call to histogramnd then the user is
            responsible for providing the same parameters
            (*n_bins*, *bins_rng*, ...).
    :type histo: *optional*, :class:`numpy.array`

    :param cumul:
        Use this parameter if you want to pass your
        own weighted histogram array instead of
        the created by this function. New
        values will be added to this array. The returned array
        will then be this one (same reference).

        .. warning:: If the cumul array was created by a previous
            call to histogramnd then the user is
            responsible for providing the same parameters
            (*n_bins*, *bins_rng*, ...).
    :type cumul: *optional*, :class:`numpy.array`

    :return: Histogram (bin counts, always returned) and weighted histogram of
        the sample (or *None* if weights is *None*).
    :rtype: *tuple* (:class:`numpy.array`, :class:`numpy.array`) or
        (:class:`numpy.array`, None)
    """
    return _chistogramnd(sample,
                         bins_rng,
                         n_bins,
                         weights=weights,
                         weight_min=weight_min,
                         weight_max=weight_max,
                         last_bin_closed=last_bin_closed,
                         histo=histo,
                         cumul=cumul)


class HistogramndLut(object):
    """
    ``HistogramndLut(sample, bins_rng, n_bins, last_bin_closed=True)``
    The HistogramndLut class allows you to bin data onto a regular grid.
    The use of HistogramndLut is interesting when several sets of data that
    share the same coordinates (*sample*) have to be mapped onto the same grid.

    .. seealso::
        `silx.math.histogramnd.histogramnd`

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
                 last_bin_closed=True,
                 dtype=None):
        """
        TTTTT
        """
        lut, histo = _histo_get_lut(sample,
                                    bins_rng,
                                    n_bins,
                                    last_bin_closed=last_bin_closed)
        self.__n_bins = n_bins
        self.__bins_rng = bins_rng
        self.__lut = lut
        self.__histo = histo
        self.__dtype = dtype
        self.reset()
        
    def reset(self):
        self.__weighted_histo = None
        self.__n_histo = 0

    def __get_weighted_histo(self):
        if self.__n_histo > 0:
            return self.__weighted_histo.copy()
        else:
            return None

    def __get_histo(self):
        return self.__histo.copy()
#        if self.__n_histo > 0:
#            return self.__histo.copy() * self.__n_histo
#        else:
#            return None

    def __get_dtype(self):
        return self.__dtype

    def __get_bins_rng(self):
        return self.__bins_rng

    def __get_n_bins(self):
        return self.__n_bins

    def __get_n_histo(self):
        return self.__n_histo

    def accumulate(self,
                   weights,
                   weight_min=None,
                   weight_max=None):

        if self.__dtype is None:
            self.__dtype = weights.dtype

        w_histo = _histo_from_lut(weights,
                                  self.__lut,
                                  shape=self.__histo.shape,
                                  weighted_histo=self.__weighted_histo,
                                  dtype=self.__dtype,
                                  weight_min=weight_min,
                                  weight_max=weight_max)
        self.__weighted_histo = w_histo
        self.__n_histo += 1

    def apply(self,
              weights,
              weighted_histo=None,
              weight_min=None,
              weight_max=None):

        if weighted_histo is None:
            if self.__dtype is None:
                dtype = weights.dtype
            else:
                dtype = self.__dtype

        w_histo = _histog_from_lut(weights,
                                   self.__lut,
                                   shape=self.__histo.shape,
                                   weighted_histo=weighted_histo,
                                   dtype=dtype,
                                   weight_min=weight_min,
                                   weight_max=weight_max)
        return self.__histo, w_histo

    histo = property(__get_histo)
    """ Test doc property """
    weighted_histo = property(__get_weighted_histo)
    dtype = property(__get_dtype)
    bins_rng = property(__get_bins_rng)
    n_bins = property(__get_n_bins)
    n_histo = property(__get_n_histo)

if __name__ == '__main__':
    pass
