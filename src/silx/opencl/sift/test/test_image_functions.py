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

"""
Python implementation of a few functions
"""
from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2018"

import numpy


def normalize_image(img):
    maxi = numpy.float32(img.max())
    mini = numpy.float32(img.min())
    return numpy.ascontiguousarray(numpy.float32(255) * (img - mini) / (maxi - mini), dtype=numpy.float32)


def shrink(img, xs, ys):
    return img[0::ys, 0::xs]


def my_gradient(mat):
    """
    numpy implementation of gradient :
    "The gradient is computed using central differences in the interior and first differences at the boundaries. The returned gradient hence has the same shape as the input array."
    NOTE:
        -with numpy.gradient, the amplitude is twice smaller than in SIFT.cpp, therefore we multiply the amplitude by two
    """
    g = numpy.gradient(mat)
    return 2.0 * numpy.sqrt(g[0] ** 2 + g[1] ** 2), numpy.arctan2(g[0], g[1])  # sift.cpp puts a "-" here


def my_local_maxmin(DOGS, thresh, border_dist, octsize, EdgeThresh0, EdgeThresh, nb_keypoints, s, dog_width, dog_height):
    """
    a python implementation of 3x3 maximum (positive values) or minimum (negative or null values) detection
    an extremum candidate "val" has to be greater than 0.8*thresh
    The three DoG have the same size.
    """
    output = -numpy.ones((nb_keypoints, 4), dtype=numpy.float32)  # for invalid keypoints

    dog_prev = DOGS[s - 1]
    dog = DOGS[s]
    dog_next = DOGS[s + 1]
    counter = 0

    for j in range(border_dist, dog_width - border_dist):
        for i in range(border_dist, dog_height - border_dist):
            val = dog[i, j]
            if (numpy.abs(val) > 0.8 * thresh):  # keypoints refinement: eliminating low-contrast points
                if (is_maxmin(dog_prev, dog, dog_next, val, i, j, octsize, EdgeThresh0, EdgeThresh) != 0):
                    output[counter, 0] = val
                    output[counter, 1] = i
                    output[counter, 2] = j
                    output[counter, 3] = numpy.float32(s)
                    counter += 1
    return output, counter


def is_maxmin(dog_prev, dog, dog_next, val, i0, j0, octsize, EdgeThresh0, EdgeThresh):
    """
    return 1 iff mat[i0,j0] is a local (3x3) maximum
    return -1 iff mat[i0,j0] is a local (3x3) minimum
    return 0 by default (neither maximum nor minimum, or value on an edge)
     * Assumes that we are not on the edges, i.e border_dist >= 2 above
    """
    ismax = 0
    ismin = 0
    res = 0
    if (val > 0.0):
        ismax = 1
    else:
        ismin = 1
    for j in range(j0 - 1, j0 + 1 + 1):
        for i in range(i0 - 1, i0 + 1 + 1):
            if (ismax == 1):
                if (dog_prev[i, j] > val or dog[i, j] > val or dog_next[i, j] > val):
                    ismax = 0
            if (ismin == 1):
                if (dog_prev[i, j] < val or dog[i, j] < val or dog_next[i, j] < val):
                    ismin = 0

    if (ismax == 1):
        res = 1
    if (ismin == 1):
        res = -1

    # keypoint refinement: eliminating points at edges
    H00 = dog[i0 - 1, j0] - 2.0 * dog[i0, j0] + dog[i0 + 1, j0]
    H11 = dog[i0, j0 - 1] - 2.0 * dog[i0, j0] + dog[i0, j0 + 1]
    H01 = ((dog[i0 + 1, j0 + 1] - dog[i0 + 1, j0 - 1])
           - (dog[i0 - 1, j0 + 1] - dog[i0 - 1, j0 - 1])) / 4.0

    det = H00 * H11 - H01 * H01
    trace = H00 + H11

    if (octsize <= 1):
        thr = EdgeThresh0
    else:
        thr = EdgeThresh
    if (det < thr * trace * trace):
        res = 0

    return res


def my_interp_keypoint(DOGS, s, r, c, movesRemain, peakthresh, width, height):
    ''''
     A Python implementation of SIFT "InterpKeyPoints"
     (s,r,c) : coords of the processed keypoint in the scale space
     WARNING: replace "1.6" by "InitSigma" if InitSigma has not its default value
     The recursive calls were replaced by a loop.
    '''
    if (r == -1):
        return (-1, -1, -1, -1)
    dog_prev = DOGS[s - 1]
    dog = DOGS[s]
    dog_next = DOGS[s + 1]
    newr = r
    newc = c
    loop = 1
    movesRemain = 5
    while (loop == 1):

        r0, c0 = newr, newc
        x, peakval = fit_quadratic(dog_prev, dog, dog_next, newr, newc)

        if (x[1] > 0.6 and newr < height - 3):
            newr += 1
        elif (x[1] < -0.6 and newr > 3):
            newr -= 1
        if (x[2] > 0.6 and newc < width - 3):
            newc += 1
        elif (x[2] < -0.6 and newc > 3):
            newc -= 1

        # loop test
        if (movesRemain > 0) and (newr != r or newc != c):
            movesRemain -= 1
        else:
            loop = 0

    if (abs(x[0]) < 1.5 and abs(x[1]) < 1.5 and abs(x[2]) < 1.5 and abs(peakval) > peakthresh):
        ki = numpy.zeros(4, dtype=numpy.float32)
        ki[0] = peakval
        ki[1] = r0 + x[1]
        ki[2] = c0 + x[2]
        ki[3] = 1.6 * 2.0 ** ((float(s) + x[0]) / 3.0)  # 3.0 is "par.Scales"
    else:
        ki = (-1, -1, -1, -1)

    return ki  # our interpolated keypoint


def fit_quadratic(dog_prev, dog, dog_next, r, c):
    '''
    quadratic interpolation around the keypoint (s,r,c)
    '''
    r = int(round(r))
    c = int(round(c))
    # gradient
    g = numpy.zeros(3, dtype=numpy.float32)
    g[0] = (dog_next[r, c] - dog_prev[r, c]) / 2.0
    g[1] = (dog[r + 1, c] - dog[r - 1, c]) / 2.0
    g[2] = (dog[r, c + 1] - dog[r, c - 1]) / 2.0
    # hessian
    H = numpy.zeros((3, 3)).astype(numpy.float32)
    H[0][0] = dog_prev[r, c] - 2.0 * dog[r, c] + dog_next[r, c]
    H[1][1] = dog[r - 1, c] - 2.0 * dog[r, c] + dog[r + 1, c]
    H[2][2] = dog[r, c - 1] - 2.0 * dog[r, c] + dog[r, c + 1]
    H[0][1] = H[1][0] = ((dog_next[r + 1, c] - dog_next[r - 1, c])
                         - (dog_prev[r + 1, c] - dog_prev[r - 1, c])) / 4.0

    H[0][2] = H[2][0] = ((dog_next[r, c + 1] - dog_next[r, c - 1])
                         - (dog_prev[r, c + 1] - dog_prev[r, c - 1])) / 4.0

    H[1][2] = H[2][1] = ((dog[r + 1, c + 1] - dog[r + 1, c - 1])
                         - (dog[r - 1, c + 1] - dog[r - 1, c - 1])) / 4.0

    x = -numpy.dot(numpy.linalg.inv(H), g)  # extremum position
    peakval = dog[r, c] + 0.5 * (x[0] * g[0] + x[1] * g[1] + x[2] * g[2])

    return x, peakval


def my_orientation(keypoints, nb_keypoints, keypoints_start, keypoints_end, grad,
                   ori, octsize, orisigma):
    '''
    Python implementation of orientation assignment
    '''

    counter = keypoints_end  # actual_nb_keypoints
    hist = numpy.zeros(36, dtype=numpy.float32)
    rows, cols = grad.shape
    for index, k in enumerate(keypoints[keypoints_start:keypoints_end]):
        if (k[1] != -1.0):
            hist = hist * 0  # do not forget this memset at each loop...
            row = numpy.int32(k[1] + 0.5)
            col = numpy.int32(k[2] + 0.5)
            sigma = orisigma * k[3]
            radius = numpy.int32(sigma * 3.0)
            rmin = max(0, row - radius)
            cmin = max(0, col - radius)
            rmax = min(row + radius, rows - 2)
            cmax = min(col + radius, cols - 2)
            radius2 = numpy.float32(radius * radius)
            sigma2 = 2.0 * sigma * sigma
            # print rmin, rmax, cmin, cmax

            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    gval = grad[r, c]
                    distsq = (r - k[1]) * (r - k[1]) + (c - k[2]) * (c - k[2])

                    if (gval > 0.0) and (distsq < radius2 + 0.5):
                        weight = numpy.exp(-distsq / sigma2)
                        angle = ori[r, c]
                        mybin = numpy.int32((36 * (angle + numpy.pi + 0.001) / (2.0 * numpy.pi)))
                        if (mybin >= 0 and mybin <= 36):
                            mybin = min(mybin, 36 - 1)
                            hist[mybin] += weight * gval

            for i in range(0, 6):
                hist = smooth_histogram(hist)

            maxval = hist.max()
            argmax = hist.argmax()

            if argmax == 0:
                prev = 35
            else:
                prev = argmax - 1
            if argmax == 35:
                next = 0
            else:
                next = argmax + 1
            if (maxval < 0.0):
                hist[prev] = -hist[prev]
                maxval = -maxval
                hist[next] = -hist[next]

            interp = 0.5 * (hist[prev] - hist[next]) / (hist[prev] - 2.0 * maxval + hist[next])
            angle = 2.0 * numpy.pi * (argmax + 0.5 + interp) / 36 - numpy.pi
            k[0] = k[2] * octsize
            k[1] = k[1] * octsize
            k[2] = k[3] * octsize
            k[3] = angle
            keypoints[index] = k

            k2 = numpy.zeros(4, dtype=numpy.float32)
            k2[0] = k[0]
            k2[1] = k[1]
            k2[2] = k[2]
            k2[3] = 0.0
            for i in range(0, 36):
                if i == 0:
                    prev = 35
                else:
                    prev = i - 1
                if i == 35:
                    next = 0
                else:
                    next = i + 1
                if (hist[i] > hist[prev] and hist[i] > hist[next] and hist[i] >= 0.8 * maxval and i != argmax):
                    if (hist[i] < 0.0):
                        hist[prev] = -hist[prev]
                        hist[i] = -hist[i]
                        hist[next] = -hist[next]
                    if (hist[i] >= hist[prev] and hist[i] >= hist[next]):
                        interp = 0.5 * (hist[prev] - hist[next]) / (hist[prev] - 2.0 * hist[i] + hist[next])

                    angle = 2.0 * numpy.pi * (i + 0.5 + interp) / 36 - numpy.pi
                    if (angle >= -numpy.pi and angle <= numpy.pi):
                        k2[3] = angle
                        if (counter < nb_keypoints):
                            keypoints[counter] = k2
                            counter += 1

            # end of additional keypoints creation
        # end of "if valid keypoint"
    # end of loop
    return keypoints, counter


def smooth_histogram(hist):
    prev = hist[35]
    for i in range(0, 36):
        temp = hist[i]
        if (i + 1 == 36):
            idx = 0
        else:
            idx = i + 1
        hist[i] = (prev + hist[i] + hist[idx]) / 3.0
        prev = temp
    return hist


def my_descriptor(keypoints, grad, orim, octsize, keypoints_start, keypoints_end):
    '''
    Python implementation of keypoint descriptor computation
    '''
    # a descriptor is a 128-vector (4,4,8) ; we need keypoints_end-keypoints_start+1  descriptors
    descriptors = numpy.zeros((keypoints_end - keypoints_start, 4, 4, 8), dtype=numpy.float32)
    for index, k in enumerate(keypoints[keypoints_start:keypoints_end]):
        if (k[1] != -1.0):
            irow, icol = int(k[1] / octsize + 0.5), int(k[0] / octsize + 0.5)
            sine, cosine = numpy.sin(k[3]), numpy.cos(k[3])
            spacing = k[2] / octsize * 3
            iradius = int((1.414 * spacing * (5) / 2.0) + 0.5)
            for i in range(-iradius, iradius + 1):
                for j in range(-iradius, iradius + 1):
                    (rx, cx) = (numpy.dot(numpy.array([[cosine, -sine], [sine, cosine]]), numpy.array([i, j]))
                                - numpy.array([k[1] / octsize - irow, k[0] / octsize - icol])) / spacing + 1.5

                    if (rx > -1.0 and rx < 4.0 and cx > -1.0 and cx < 4.0
                       and (irow + i) >= 0 and (irow + i) < grad.shape[0]
                       and (icol + j) >= 0 and (icol + j) < grad.shape[1]):

                        mag = grad[int(irow + i), int(icol + j)] * numpy.exp(-((rx - 1.5) ** 2 + (cx - 1.5) ** 2) / 8.0)
                        ori = orim[int(irow + i), int(icol + j)] - k[3]

                        while (ori > 2.0 * numpy.pi):
                            ori -= 2.0 * numpy.pi
                        while (ori < 0.0):
                            ori += 2.0 * numpy.pi

                        oval = 8 * ori / (2.0 * numpy.pi)

                        ri = int(rx if (rx >= 0.0) else rx - 1.0)
                        ci = int(cx if (cx >= 0.0) else cx - 1.0)
                        oi = int(oval if (oval >= 0.0) else oval - 1.0)
                        rfrac, cfrac, ofrac = rx - ri, cx - ci, oval - oi

                        if (ri >= -1 and ri < 4 and oi >= 0 and oi <= 8
                           and rfrac >= 0.0 and rfrac <= 1.0):
                            for r in range(0, 2):
                                rindex = ri + r
                                if (rindex >= 0 and rindex < 4):
                                    rweight = mag * (1.0 - rfrac if (r == 0) else rfrac)
                                    for c in range(0, 2):
                                        cindex = ci + c
                                        if (cindex >= 0 and cindex < 4):
                                            cweight = rweight * (1.0 - cfrac if (c == 0) else cfrac)
                                            for orr in range(0, 2):
                                                oindex = oi + orr
                                                if (oindex >= 8):
                                                    oindex = 0
                                                descriptors[index][rindex][cindex][oindex] += cweight * (1.0 - ofrac if (orr == 0) else ofrac)
                                        # end "valid cindex"
                                # end "valid rindex"
                        # end "sample in boundaries"
                # end "j loop"
            # end "i loop"
        # end "valid keypoint"
    # end loop in keypoints

    # unwrap and normalize the 128-vector
    descriptors = descriptors.reshape(keypoints_end - keypoints_start, 128)

    for i in range(0, keypoints_end - keypoints_start):
        descriptors[i] = normalize(descriptors[i])

    # threshold to 0.2 like in sift.cpp
    changed = 0
    for i in range(0, keypoints_end - keypoints_start):
        idx = descriptors[i] > 0.2
        if (idx.shape[0] != 0):
            (descriptors[i])[idx] = 0.2
            changed = 1

    # if values were actually threshold, we have to normalize again
    if (changed == 1):
        for i in range(0, keypoints_end - keypoints_start):
            descriptors[i] = normalize(descriptors[i])

    # cast to "unsigned char"
    descriptors = 512 * descriptors
    for i in range(0, keypoints_end - keypoints_start):
        (descriptors[i])[255 <= descriptors[i]] = 255

    descriptors = descriptors.astype(numpy.uint8)

    return descriptors


def normalize(vec):
    return (vec / numpy.linalg.norm(vec) if numpy.linalg.norm(vec) != 0 else 0)


def my_matching(keypoints1, keypoints2, start, end, ratio_th=0.5329):
    '''
    Python implementation of SIFT keypoints matching
    '''
    counter = 0
    matchings = numpy.zeros((end - start, 2), dtype=numpy.uint32)
    for i, desc1 in enumerate(keypoints1):  # FIXME: keypoints1.desc.... idem below
        ratio, match = check_for_match(desc1, keypoints2)
        if (ratio < ratio_th and i <= match):
            matchings[counter] = i, match
            counter += 1
    return matchings, counter


def check_for_match(desc1, keypoints2):
    '''
    check if the descriptor "desc1" has matches in the list "keypoints2"
    '''
    current_min = 0
    dist1 = dist2 = 1000000000000.0
    for j, desc2 in enumerate(keypoints2):
        dst = l1_distance(desc1, desc2)
        if (dst < dist1):
            dist2 = dist1
            dist1 = dst
            current_min = j
        elif (dst < dist2):
            dist2 = dst

    return dist1 / dist2, current_min


def l1_distance(desc1, desc2):
    '''
    L1 distance between two vectors
    '''
    return abs(desc1 - desc2).sum()


def keypoints_compare(ref, res):
    '''
    When using atomic instructions in kernels, the resulting keypoints are not in the same order as Python implementation
    '''
    res_c = res[(res[:, 0].argsort(axis=0)), 0]
    ref_c = ref[(ref[:, 0].argsort(axis=0)), 0]
    res_r = res[(res[:, 1].argsort(axis=0)), 1]
    ref_r = ref[(ref[:, 1].argsort(axis=0)), 1]
    res_s = res[(res[:, 2].argsort(axis=0)), 2]
    ref_s = ref[(ref[:, 2].argsort(axis=0)), 2]
    res_angle = res[(res[:, 3].argsort(axis=0)), 3]
    ref_angle = ref[(ref[:, 3].argsort(axis=0)), 3]

    return abs(res_c - ref_c).max(), abs(res_r - ref_r).max(), abs(res_s - ref_s).max(), abs(res_angle - ref_angle).max()


def descriptors_compare(ref, res):
    # count null descriptors in "ref" (take care of the order in the arguments : (ref, res) and not (res, ref))
    nulldesc = 0
    for descriptor2 in res:
        if abs(descriptor2).sum() == 0:
            nulldesc += 1
    # count descriptors that are (exactly) equal
    match = 0
    delta = 0
    for descriptor in ref:
        for descriptor2 in res:
            delta = abs(descriptor - descriptor2).sum()
            if delta == 0:
                match += 1
    return match, nulldesc


def check_for_doubles(res):
    # check for descriptors that appear more than one time in the list
    doubles = 0
    cnt = numpy.zeros(res.shape[0])
    for idx, desc in enumerate(res):
        for idx2, desc2 in enumerate(res):
            if abs(desc - desc2).sum() == 0 and idx != idx2 and (cnt[idx] == 0 or cnt[idx2] == 0):
                cnt[idx] = 1
                cnt[idx2] = 1
                doubles += 1
    return doubles


def my_compact(keypoints, nbkeypoints):
    '''
    Reference compacting
    '''
    output = -numpy.ones_like(keypoints)
    idx = numpy.where(keypoints[:, 1] != -1)[0]
    length = idx.size
    output[:length, 0] = keypoints[idx, 0]
    output[:length, 1] = keypoints[idx, 1]
    output[:length, 2] = keypoints[idx, 2]
    output[:length, 3] = keypoints[idx, 3]
    return output, length


def norm_L1(dset1, dset2):
    """Checks the similarity of two vectors of vectors:

    S = max_along_dim0_for_i_in_dset1(min along_dim0_for_j_in_dset2(sum_for_k_in_dim1(abs(dset1[i,k]-dset2[j,k]))))
    :return: Similarity S
    """
    if len(dset2) > len(dset1):
        dset2, dset1 = dset1, dset2
    d1 = dset1[numpy.newaxis, ...]
    d2 = dset2[:, numpy.newaxis, ...]
    # numpy.save("file", d1 - d2)
    # d = abs(d2 - d1).sum(axis=-1)
    # print(d.shape)
    # print(d)
    d = abs(d2 - d1).min(axis=-1)
    return d.min(axis=-1).max()


'''


  function KahanSum(input)
    var sum = 0.0
    var c = 0.0
    for i = 1 to input.length do
        y = input[i] - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    return sum



'''
