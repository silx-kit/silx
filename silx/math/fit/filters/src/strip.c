#/*##########################################################################
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
#############################################################################*/
/*
    This file provides a background strip function, to isolate low frequency
    background signal from a spectrum (and later substact it from the signal
    to be left only with the peaks to be fitted).

    It is adapted from PyMca source file "SpecFitFuns.c". The main difference
    with the original code is that this code does not handle the python
    wrapping, which is done elsewhere using cython.

    Authors: V.A. Sole, P. Knobel
    License: MIT
    Last modified: 17/06/2016
*/

#include <string.h>

#include <stdio.h>

/*  strip(double* input, double c, long niter, double* output)

    The strip background is probably PyMca's  most popular background model.

    In its simplest implementation it is just as an iterative procedure depending
    on two parameters. These parameters are the strip background width w, and the
    strip background number of iterations. At each iteration, if the contents of
    channel i, y(i), is above the average of the contents of the channels at w
    channels of distance, y(i-w) and y(i+w),  y(i) is replaced by the average.
    At the end of the process we are left with something that resembles a spectrum
    in which the peaks have been "stripped".

    Parameters:

        - input: Input data array
        - c: scaling factor applied to the average of y(i-w) and y(i+w) before
          comparing to y(i)
        - niter: number of iterations
        - deltai: operator width (in number of channels)
        - anchors: Array of anchors, indices of points that will not be
          modified during the stripping procedure.
        - output: output array

*/
int strip(double* input, long len_input,
          double c, long niter, int deltai,
          long* anchors, long len_anchors,
          double* output)
{
    long iter_index, array_index, anchor_index, anchor;
    int anchor_nearby_flag;
    double  t_mean;

    memcpy(output, input, len_input * sizeof(double));

    if (deltai <=0) deltai = 1;

    if (len_input < (2*deltai+1)) return(-1);

    if (len_anchors > 0) {
        for (iter_index = 0; iter_index < niter; iter_index++) {
            for (array_index = deltai; array_index < len_input - deltai; array_index++) {
                /* if index is within +- deltai of an anchor, don't do anything */
                anchor_nearby_flag = 0;
                for (anchor_index=0; anchor_index<len_anchors; anchor_index++)
                {
                    anchor = anchors[anchor_index];
                    if (array_index > (anchor - deltai) && array_index < (anchor + deltai))
                    {
                        anchor_nearby_flag = 1;
                        break;
                    }
                }
                /* skip this array_index index */
                if (anchor_nearby_flag) {
                    continue;
                }

                t_mean = 0.5 * (input[array_index-deltai] + input[array_index+deltai]);
                if (input[array_index] > (t_mean * c))
                    output[array_index] = t_mean;
            }
            memcpy(input, output, len_input * sizeof(double));
        }
    }
    else {
        for (iter_index = 0; iter_index < niter; iter_index++) {
            for (array_index=deltai; array_index < len_input - deltai; array_index++) {
                t_mean = 0.5 * (input[array_index-deltai] + input[array_index+deltai]);

                if (input[array_index] > (t_mean * c))
                    output[array_index] = t_mean;
            }
            memcpy(input, output, len_input * sizeof(double));
        }
    }
    return(0);
}
