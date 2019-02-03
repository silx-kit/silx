/*##########################################################################
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

#ifndef FITFILTERS_H
#define FITFILTERS_H

/* Background functions */
void snip1d(double *data, int size, int width);
//void snip1d_multiple(double *data, int n_channels, int snip_width, int n_spectra);
void snip2d(double *data, int nrows, int ncolumns, int width);
void snip3d(double *data, int nx, int ny, int nz, int width);

int strip(double* input, long len_input, double c, long niter, int deltai,
          long* anchors, long len_anchors, double* output);

/* Smoothing functions */

int SavitskyGolay(double* input, long len_input, int npoints, double* output);

void smooth1d(double *data, int size);
void smooth2d(double *data, int size0, int size1);
void smooth3d(double *data, int size0, int size1, int size2);


#endif /* #define FITFILTERS_H */
