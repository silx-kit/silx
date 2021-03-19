/*
 *   Project: Silx statics calculation
 *
 *
 *
 *   Copyright (C) 2012-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 13/12/2018
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * \file
 *
 * \brief OpenCL kernels for min, max, mean and std calculation
 *
 * Constant to be provided at build time:
 *
 */

#include "for_eclipse.h"

/* \brief read a value at given position and initialize the float8 for the reduction
 *
 * The float8 returned contains:
 * s0: minimum value
 * s1: maximum value
 * s2: count number of valid pixels
 * s3: count (error associated to)
 * s4: sum of valid pixels
 * s5: sum (error associated to)
 * s6: variance*count
 * s7: variance*count (error associated to)
 *
 */
static inline float8 map_statistics(global float* data, int position)
{
    float value = data[position];
    float8 result;

    if (isfinite(value))
    {
        result = (float8)(value, value, 1.0f, 0.0f, value, 0.0f, 0.0f, 0.0f);
        //                min     max   cnt   cnt_err  sum   sum_err M  M_err
    }
    else
    {
        result = (float8)(FLT_MAX, -FLT_MAX, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    return result;
}

/* \brief reduction function associated to the statistics.
 *
 * this is described in:
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 *
 * The float8 used here contain contains:
 * s0: minimum value
 * s1: maximum value
 * s2: count number of valid pixels
 * s3: count (error associated to)
 * s4: sum of valid pixels
 * s5: sum (error associated to)
 * s6: M=variance*(count-1)
 * s7: M=variance*(count-1) (error associated to)
 *
 */

static inline float8 reduce_statistics(float8 a, float8 b)
{
    float2 sum_a, sum_b, M_a, M_b, count_a, count_b;

    //test on count
    if (a.s2 == 0.0f)
    {
        return b;
    }
    else
    {
        count_a = (float2)(a.s2, a.s3);
        sum_a = (float2)(a.s4, a.s5);
        M_a = (float2)(a.s6, a.s7);
    }
    //test on count
    if (b.s2 == 0.0f)
    {
        return a;
    }
    else
    {
        count_b = (float2)(b.s2, b.s3);
        sum_b = (float2)(b.s4, b.s5);
        M_b = (float2)(b.s6, b.s7);
    }
    // count = count_a + count_b
    float2 count = compensated_sum(count_a, count_b);
    // sum = sum_a + sum_b
    float2 sum = compensated_sum(sum_a, sum_b);

    //delta = avg_b - avg_a
    //delta^2 = avg_b^2 + avg_a^2 - 2*avg_b*avg_a
    //coount_a*count_b*delta^2 = count_a/count_b * sum_b^2 + count_b/count_a*sum_a^2 - 2*sum_a*sum_b

    //float2 sum2_a = compensated_mul(sum_a, sum_a);
    //float2 sum2_b = compensated_mul(sum_b, sum_b);
    //float2 ca_over_cb = compensated_mul(count_a, compensated_inv(count_b));
    //float2 cb_over_ca = compensated_mul(count_b, compensated_inv(count_a));

    //float2 delta2cbca = compensated_sum(compensated_sum(
    //                    compensated_mul(ca_over_cb, sum2_b),
    //                    compensated_mul(cb_over_ca, sum2_a)),
    //                    -2.0f * compensated_mul(sum_a, sum_b));
//////////////
//    float2 delta = compensated_sum(
//                  compensated_mul(sum_b, compensated_inv(count_b)),
//              -1*(compensated_mul(sum_a, compensated_inv(count_a))));
    float2 delta = compensated_sum(compensated_div(sum_b, count_b),
                                   -1*compensated_div(sum_a, count_a));

    float2 delta2cbca = compensated_mul(compensated_mul(delta, delta),
                                        compensated_mul(count_a, count_b));
    float2 M2 = compensated_sum(compensated_sum(M_a, M_b),
                                compensated_mul(delta2cbca, compensated_inv(count)));
    //M2 = M_a + M_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    float8 result = (float8)(min(a.s0, b.s0), max(a.s1, b.s1),
                             count.s0,        count.s1,
                             sum.s0,          sum.s1,
                             M2.s0,           M2.s1);
    return result;
}

/* \brief reduction function associated to the statistics without compensated arithmetics.
 *
 * this is described in:
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 *
 * The float8 used here contain contains:
 * s0: minimum value
 * s1: maximum value
 * s2: count number of valid pixels
 * s3: count (error associated to)
 * s4: sum of valid pixels
 * s5: sum (error associated to)
 * s6: M=variance*(count-1)
 * s7: M=variance*(count-1) (error associated to)
 *
 */

static inline float8 reduce_statistics_simple(float8 a, float8 b)
{
    float sum_a, sum_b, M_a, M_b, count_a, count_b;

    //test on count
    if (a.s2 == 0.0f)
    {
        return b;
    }
    else
    {
        count_a = a.s2;
        sum_a = a.s4;
        M_a = a.s6;
    }
    //test on count
    if (b.s2 == 0.0f)
    {
        return a;
    }
    else
    {
        count_b = b.s2;
        sum_b = b.s4;
        M_b = b.s6;
    }
    float count = count_a + count_b;
    float sum = sum_a + sum_b;
    float delta = sum_a/count_a - sum_b/count_b;
    float delta2cbca = count_b * count_a * delta * delta;
    float M2 = M_a + M_b + delta2cbca/count;
    //M2 = M_a + M_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    float8 result = (float8)(min(a.s0, b.s0), max(a.s1, b.s1),
                             count,        0.0f,
                             sum,          0.0f,
                             M2,           0.0f);
    return result;
}


