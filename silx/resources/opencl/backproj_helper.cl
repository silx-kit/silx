/*
 *   Project: silx: backprojection helper functions
 *
 *   Copyright (C) 2016-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: P. Paleo
 *                      J. Kieffer (kieffer@esrf.fr)
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */


kernel void mult(   global float2* d_sino,
                    global float2* d_filter,
                    int num_bins,
                    int num_projs)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    if (gid0 < num_bins && gid1 < num_projs)
    {
        // d_sino[gid1*num_bins+gid0] *= d_filter[gid0];
        d_sino[gid1*num_bins+gid0].x *= d_filter[gid0].x;
        d_sino[gid1*num_bins+gid0].y *= d_filter[gid0].x;
      }
}

// copy only the real part of the valid data to the real array
kernel void cpy2d_c2r(
                      global float* d_sino,
                      global float2* d_sino_complex,
                      int num_bins,
                      int num_projs,
                      int fft_size)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    if (gid0 < num_bins && gid1 < num_projs) {
        d_sino[gid1*num_bins+gid0] = d_sino_complex[gid1*fft_size+gid0].x;
    }
}







