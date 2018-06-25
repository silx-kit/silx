#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

'''
Unit tests become more and more difficult as we progress in the global SIFT algorithm
For a better code visibility, the setups required by kernels will be put here
'''
from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2018"

import numpy
try:
    import scipy.ndimage
    import scipy.misc
except ImportError:
    scipy = None

from .test_image_functions import my_gradient, normalize_image, shrink, my_local_maxmin, \
    my_interp_keypoint, my_descriptor, my_orientation
from .test_algebra import my_compact
from math import ceil


def my_blur(img, sigma):
    if scipy is None:
        raise ImportError("scipy required")
    ksize = int(ceil(8 * sigma + 1))
    if (ksize % 2 == 0):
        ksize += 1
    x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
    gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
    gaussian /= gaussian.sum(dtype=numpy.float32)
    tmp1 = scipy.ndimage.filters.convolve1d(img, gaussian, axis=-1, mode="reflect")
    return scipy.ndimage.filters.convolve1d(tmp1, gaussian, axis=0, mode="reflect")


def local_maxmin_setup():
    if scipy is None:
        raise ImportError("scipy required")

    border_dist = numpy.int32(5)  # SIFT
    peakthresh = numpy.float32(255.0 * 0.04 / 3.0)  # SIFT uses 255.0 * 0.04 / 3.0
    EdgeThresh = numpy.float32(0.06)  # SIFT
    EdgeThresh0 = numpy.float32(0.08)  # SIFT
    octsize = numpy.int32(2)  # initially 1, then twiced at each new octave
    scale = numpy.int32(1)  # 1,2 or 3
    nb_keypoints = 1000  # constant size !
    doubleimsize = 0  # par.DoubleImSize = 0 by default

    if hasattr(scipy.misc, "ascent"):
        l2 = scipy.misc.ascent().astype(numpy.float32)
    else:
        l2 = scipy.misc.lena().astype(numpy.float32)

    l2 = numpy.ascontiguousarray(l2[0:507, 0:209])
    # l2 = scipy.misc.imread("../aerial.tiff").astype(numpy.float32)
    l = normalize_image(l2)  # do not forget to normalize the image if you want to compare with sift.cpp
    for octave_cnt in range(1, int(numpy.log2(octsize)) + 1 + 1):

        width = numpy.int32(l.shape[1])
        height = numpy.int32(l.shape[0])

        # Blurs and DoGs pre-allocating
        g = (numpy.zeros(6 * height * width).astype(numpy.float32)).reshape(6, height, width)  # vector of 6 blurs
        DOGS = numpy.zeros((5, height, width), dtype=numpy.float32)  # vector of 5 DoGs
        g[0, :, :] = numpy.copy(l)
        '''
        sift.cpp pre-process
        '''
        if (octave_cnt == 1):
            initsigma = 1.6
            if (doubleimsize):
                cursigma = 1.0
            else:
                cursigma = 0.5
            # Convolving initial image to achieve std = initsigma = 1.6
            if (initsigma > cursigma):
                sigma = numpy.sqrt(initsigma ** 2 - cursigma ** 2)
                g[0, :, :] = my_blur(l, sigma)
        else:
            g[0, :, :] = numpy.copy(l)
        '''
        Blurs and DoGs
        '''
        sigmaratio = 2 ** (1 / 3.0)  # sift.cpp
        # sift.cpp : for a given "i", we have : increase = initsigma*(sigmaratio)^(i-1)*sqrt(sigmaratio**2 -1)
        for i in range(1, 6):
            sigma = initsigma * (sigmaratio) ** (i - 1.0) * numpy.sqrt(sigmaratio ** 2 - 1.0)  # sift.cpp "increase"
            g[i] = my_blur(g[i - 1], sigma)  # blur[i]

        for s in range(1, 6):
            DOGS[s - 1] = -(g[s] - g[s - 1])  # DoG[s-1]

        if (octsize > 1):  # if a higher octave is required, we have to sample Blur[3]
            l = shrink(g[3], 2, 2)
    return border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, scale, nb_keypoints, width, height, DOGS, g


def interpolation_setup():
    '''
    Provides the values required by "test_interpolation"
    Previous step: local extrema detection - we got a vector of keypoints to be interpolated
    '''

    border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, s, nb_keypoints, width, height, DOGS, g = local_maxmin_setup()

    nb_keypoints = numpy.int32(nb_keypoints)

    # Assumes that local_maxmin is working so that we can use Python's "my_local_maxmin" instead of the kernel
    keypoints_prev, actual_nb_keypoints = my_local_maxmin(DOGS, peakthresh, border_dist, octsize,
                                                          EdgeThresh0, EdgeThresh, nb_keypoints, s, width, height)

    return border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, actual_nb_keypoints, width, height, DOGS, s, keypoints_prev, g[s]


def orientation_setup():
    '''
    Provides the values required by "test_orientation"
    Previous step: interpolation - we got a vector of valid keypoints
    '''
    border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, actual_nb_keypoints, width, height, DOGS, s, keypoints_prev, blur = interpolation_setup()

    # actual_nb_keypoints = numpy.int32(len((keypoints_prev[:,0])[keypoints_prev[:,1] != -1]))
    ref = numpy.copy(keypoints_prev)
    # There are actually less than "actual_nb_keypoints" keypoints ("holes" in the vector), but we can use it as a boundary
    for i, k in enumerate(ref[:actual_nb_keypoints, :]):
        ref[i] = my_interp_keypoint(DOGS, s, k[1], k[2], 5, peakthresh, width, height)

    grad, ori = my_gradient(blur)  # gradient is applied on blur[s]

    return ref, nb_keypoints, actual_nb_keypoints, grad, ori, octsize


def descriptor_setup():
    '''
    Provides the values required by "test_descriptor"
    Previous step: orientation - we got a vector of keypoints with an orientation, and several additional keypoints
    '''
    keypoints, nb_keypoints, actual_nb_keypoints, grad, ori, octsize = orientation_setup()
    orisigma = numpy.float32(1.5)  # SIFT
    keypoints_start = numpy.int32(0)
    keypoints_end = actual_nb_keypoints  # numpy.int32(actual_nb_keypoints)
    ref, updated_nb_keypoints = my_orientation(keypoints, nb_keypoints, keypoints_start, keypoints_end, grad, ori, octsize, orisigma)

    return ref, nb_keypoints, updated_nb_keypoints, grad, ori, octsize


def matching_setup():
    '''
    Provides the values required by "test_matching"
    Previous step: descriptors - we got a vector of 128-values descriptors
    '''
    keypoints, nb_keypoints, actual_nb_keypoints, grad, ori, octsize = descriptor_setup()
    keypoints, actual_nb_keypoints = my_compact(numpy.copy(keypoints), nb_keypoints)
    keypoints_start, keypoints_end = 0, actual_nb_keypoints
    desc = my_descriptor(keypoints, grad, ori, octsize, keypoints_start, keypoints_end)
    # keypoints with their descriptors
    # FIXME: structure including keypoint (float32) and descriptors (uint8)
    kp1 = desc
    kp2 = numpy.ascontiguousarray(desc[::-1])
    return kp1, kp2, nb_keypoints, actual_nb_keypoints
