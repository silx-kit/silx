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
histogramnd benchmarks, vs numpy.histogramdd (bin counts and weights).
"""

import numpy as np

import time

from silx.math import histogramnd


def print_times(t0s, t1s, t2s, t3s):
    c_times = t1s - t0s
    np_times = t2s - t1s
    np_w_times = t3s - t2s

    time_txt = 'min : {0: <7.3f}; max : {1: <7.3f}; avg : {2: <7.3f}'

    print('\tTimes :')
    print('\tC     : ' + time_txt.format(c_times.min(),
                                         c_times.max(),
                                         c_times.mean()))
    print('\tNP    : ' + time_txt.format(np_times.min(),
                                         np_times.max(),
                                         np_times.mean()))
    print('\tNP(W) : ' + time_txt.format(np_w_times.min(),
                                         np_w_times.max(),
                                         np_w_times.mean()))


def commpare_results(txt,
                     times,
                     result_c,
                     result_np,
                     result_np_w,
                     sample,
                     weights,
                     raise_ex=False):

    if result_np:
        hits_cmp = np.array_equal(result_c[0], result_np[0])
    else:
        hits_cmp = None

    if result_np_w and result_c[1] is not None:
        weights_cmp = np.array_equal(result_c[1], result_np_w[0])
    else:
        weights_cmp = None

    if((hits_cmp is not None and not hits_cmp) or
       (weights_cmp is not None and not weights_cmp)):
        err_txt = (txt + ' : results arent the same : '
                   'hits : {0}, '
                   'weights : {1}.'
                   ''.format('OK' if hits_cmp else 'NOK',
                             'OK' if weights_cmp else 'NOK'))
        print('\t' + err_txt)
        if raise_ex:
            raise ValueError(err_txt)
        return False

    result_txt = ' : results OK. c : {0: <7.3f};'.format(times[0])
    if result_np or result_np_w:
        result_txt += (' np : {0: <7.3f}; '
                       'np (weights) {1: <7.3f}.'
                       ''.format(times[1], times[2]))
    print('\t' + txt + result_txt)
    return True


def benchmark(n_loops,
              sample_shape,
              sample_rng,
              weights_rng,
              histo_range,
              n_bins,
              weight_min,
              weight_max,
              last_bin_closed,
              dtype=np.double,
              do_weights=True,
              do_numpy=True):

    int_min = 0
    int_max = 100000

    sample = np.random.randint(int_min,
                               high=int_max,
                               size=sample_shape).astype(np.double)
    sample = (sample_rng[0] +
              (sample - int_min) *
              (sample_rng[1] - sample_rng[0]) /
              (int_max - int_min))
    sample = sample.astype(dtype)

    if do_weights:
        weights = np.random.randint(int_min,
                                    high=int_max,
                                    size=(ssetup.pyample_shape[0],))
        weights = weights.astype(np.double)
        weights = (weights_rng[0] +
                   (weights - int_min) *
                   (weights_rng[1] - weights_rng[0]) /
                   (int_max - int_min))
    else:
        weights = None

    t0s = []
    t1s = []
    t2s = []
    t3s = []

    for i in range(n_loops):
        t0s.append(time.time())
        result_c = histogramnd(sample,
                               histo_range,
                               n_bins,
                               weights=weights,
                               weight_min=weight_min,
                               weight_max=weight_max,
                               last_bin_closed=last_bin_closed)
        t1s.append(time.time())
        if do_numpy:
            result_np = np.histogramdd(sample,
                                       bins=n_bins,
                                       range=histo_range)
            t2s.append(time.time())
            result_np_w = np.histogramdd(sample,
                                         bins=n_bins,
                                         range=histo_range,
                                         weights=weights)
            t3s.append(time.time())
        else:
            result_np = None
            result_np_w = None
            t2s.append(0)
            t3s.append(0)

        commpare_results('Run {0}'.format(i),
                         [t1s[-1] - t0s[-1], t2s[-1] - t1s[-1], t3s[-1] - t2s[-1]],
                         result_c,
                         result_np,
                         result_np_w,
                         sample,
                         weights)

    print_times(np.array(t0s), np.array(t1s), np.array(t2s), np.array(t3s))


def run_benchmark(dtype=np.double,
                  do_weights=True,
                  do_numpy=True):
    n_loops = 5

    weights_rng = [0., 100.]
    sample_rng = [0., 100.]

    weight_min = None
    weight_max = None
    last_bin_closed = True

    # ====================================================
    # ====================================================
    # 1D
    # ====================================================
    # ====================================================

    print('==========================')
    print(' 1D [{0}]'.format(dtype))
    print('==========================')
    sample_shape = (10**7,)
    histo_range = [[0., 100.]]
    n_bins = 30

    benchmark(n_loops,
              sample_shape,
              sample_rng,
              weights_rng,
              histo_range,
              n_bins,
              weight_min,
              weight_max,
              last_bin_closed,
              dtype=dtype,
              do_weights=True,
              do_numpy=do_numpy)

    # ====================================================
    # ====================================================
    # 2D
    # ====================================================
    # ====================================================

    print('==========================')
    print(' 2D [{0}]'.format(dtype))
    print('==========================')
    sample_shape = (10**7, 2)
    histo_range = [[0., 100.], [0., 100.]]
    n_bins = 30

    benchmark(n_loops,
              sample_shape,
              sample_rng,
              weights_rng,
              histo_range,
              n_bins,
              weight_min,
              weight_max,
              last_bin_closed,
              dtype=dtype,
              do_weights=True,
              do_numpy=do_numpy)

    # ====================================================
    # ====================================================
    # 3D
    # ====================================================
    # ====================================================

    print('==========================')
    print(' 3D [{0}]'.format(dtype))
    print('==========================')
    sample_shape = (10**7, 3)
    histo_range = np.array([[0., 100.], [0., 100.], [0., 100.]])
    n_bins = 30

    benchmark(n_loops,
              sample_shape,
              sample_rng,
              weights_rng,
              histo_range,
              n_bins,
              weight_min,
              weight_max,
              last_bin_closed,
              dtype=dtype,
              do_weights=True,
              do_numpy=do_numpy)

if __name__ == '__main__':
    types = (np.double, np.int32, np.float32,)

    for t in types:
        run_benchmark(t,
                      do_weights=True,
                      do_numpy=True)
