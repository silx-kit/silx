/*
 *   Project: Silx statics calculation
 *
 *
 *
 *   Copyright (C) 2012-2021 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 17/05/2021
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
 * This module provides two functions to perform the `map` and the `reduce` 
 * to be used with pyopencl reduction to calculate in a single pass the minimum, 
 * maximum, sum, count, mean and standart deviation for an array.
 * 
 * So beside the reduction mechanisme from pyopencl, this algorithm implementes equations from
 * https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
 *
 * let A and B be 2 disjoint partition of all elements
 * 
 * Omega_A = sum_{i \in A}(omaga_i) The sum of all weights
 * V_A is the weighted sum of the signal over the partition
 * VV_A is the weighted sum of deviation squarred
 * 
 * With this the mean is V / Omega and the variance equals VV / omega.
 * 
 * Redction operator performs:
 * Omega_{AB} = Omega_A + Omega_B
 * V_{AB} = V_A + V_B
 * VV{AB} = VV_A + VV_B + (Omega_A*V_B-Omega_B*V_A)² / (Omega_A * Omega_B * Omega_{AB})
 *
 * To avoid any numerical degradation, the doubleword library is used to perform all floating point operations. 
 *
 */
#include "for_eclipse.h"

/* \brief read a value at given position and initialize the float8 for the reduction
 *
 * The float8 returned contains:
 * s0: minimum value
 * s1: maximum value
 * s2: Omega_h count number of valid pixels
 * s3: Omega_l error associated to the count
 * s4: V_h sum of signal
 * s5: V_l error associated to the sum of signal
 * s6: VVh variance*count
 * s7: VVl error associated to variance*count
 *
 */
static inline float8 map_statistics(global float* data, int position)
{
    float value = data[position];
    float8 result;

    if (isfinite(value))
    {
        result = (float8)(value, value, 1.0f, 0.0f, value, 0.0f, 0.0f, 0.0f);
        //                min     max   cnt_h cnt_l sum_h  sum_l M2_h  M2_l
    }
    else
    {
        result = (float8)(FLT_MAX, -FLT_MAX, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        //                min        max     cnt_h cnt_l sum_h sum_l M2_h  M2_l
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
    float2 count, sum;
    float2 M, delta, delta2, omega3;
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
    count = dw_plus_dw(count_a, count_b);
    // sum = sum_a + sum_b
    sum = dw_plus_dw(sum_a, sum_b);
    
    // M2 = M_a + M_b + (Omega_A*V_B-Omega_B*V_A)² / (Omega_A * Omega_B * Omega_{AB})
    M =  dw_plus_dw(M_a, M_b);
    delta = dw_plus_dw(dw_times_dw(count_b, sum_a),
                      -dw_times_dw(count_a, sum_b));
    delta2 = dw_times_dw(delta, delta);
    omega3 = dw_times_dw(count, dw_times_dw(count_a, count_b));            
    M = dw_plus_dw(M, dw_div_dw(delta2, omega3));
                                    
    float8 result = (float8)(min(a.s0, b.s0), max(a.s1, b.s1),
                             count.s0,        count.s1,
                             sum.s0,          sum.s1,
                             M.s0,            M.s1);
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
 * s2: count number of valid pixels (major)
 * s3: count number of valid pixels (minor)
 * s4: sum of valid pixels (major)
 * s5: sum of valid pixels (minor)
 * s6: variance*count (major)
 * s7: variance*count (minor)
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
    float delta = sum_a*count_b - sum_b*count_a;
    float delta2 = delta * delta;
    float M2 = M_a + M_b + delta2/(count*count_a*count_b);
    float8 result = (float8)(min(a.s0, b.s0), max(a.s1, b.s1),
                             count,        0.0f,
                             sum,          0.0f,
                             M2,           0.0f);
    return result;
}


#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


// unpack a double in two floats such as the  sum of the two is the double number
static inline float2 unpack_double(double inp){
    float major = (float) inp;
    float minor = (float) (inp - major);
    return (float2)(major, minor);  
}

// pack two floats into a double
static inline double pack_double(float major, float minor){
    return (double)major + (double)minor;
}

/* \brief reduction function associated to the statistics using double precision arithmetics
 *
 * this is described in:
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
 *
 * The float8 used here contain contains:
 * s0: minimum value
 * s1: maximum value
 * s2: count number of valid pixels (major)
 * s3: count number of valid pixels (minor)
 * s4: sum of valid pixels (major)
 * s5: sum of valid pixels (minor)
 * s6: variance*count (major)
 * s7: variance*count (minor)
 *
 */

static inline float8 reduce_statistics_double(float8 a, float8 b)
{
    double sum_a, sum_b, M_a, M_b, count_a, count_b;

    //test on count
    if (a.s2 == 0.0)
    {
        return b;
    }
    else
    {
        count_a = pack_double(a.s2, a.s3);
        sum_a = pack_double(a.s4,a.s5);
        M_a = pack_double(a.s6, a.s7);
    }
    //test on count
    if (b.s2 == 0.0)
    {
        return a;
    }
    else
    {
        count_b = pack_double(b.s2, b.s3);
        sum_b = pack_double(b.s4, b.s5);
        M_b = pack_double(b.s6, b.s7);
    }
    double count = count_a + count_b;
    double sum = sum_a + sum_b;
    double delta = sum_a*count_b - sum_b*count_a;
    double delta2 = delta * delta;
    double M2 = M_a + M_b + delta2/(count*count_a*count_b);
    float8 result = (float8)((float2)(min(a.s0, b.s0), max(a.s1, b.s1)),
                                unpack_double(count),
                                unpack_double( sum),
                                unpack_double( M2));
    return result;
}

#endif