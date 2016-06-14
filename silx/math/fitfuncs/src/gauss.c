#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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

*/
#include <math.h>
#include <stdlib.h>
#include "fitfunctions.h"

/* Defined in .h file
typedef struct {
    double  height;
    double  centroid;
    double  fwhm;
} gaussian_params; */

double* gauss(double* x, int len_x, double* pgauss, int len_pgauss)
{
    double* ret;
    int i, j;
    double dhelp, log2;
    double fwhm, centroid, height;

    /* Create the output array */
    ret = malloc(len_x * sizeof(double));
    for (j=0; j<len_x;  j++) {
        ret[j] = 0;
    }

    log2 = 0.69314718055994529;

    for (i=0; i<len_pgauss/3; i++) {
        for (j=0; j<len_x;  j++) {
            fwhm = pgauss[3*i+2];
            centroid = pgauss[3*i+1];
            height = pgauss[3*i];

            dhelp = fwhm / (2.0 * sqrt(2.0 * log2));
            dhelp = (x[j] - centroid)/dhelp;
            if (dhelp <= 20) {
                ret[j] += height * exp (-0.5 * dhelp * dhelp);
            }
        }
    }

    return ret;
}
