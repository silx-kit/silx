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
   Implementation of the algorithm SNIP in 2D described in
   Miroslav Morhac et al. Nucl. Instruments and Methods in Physics Research A401 (1997) 113-132.
*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void lls(double *data, int size);
void lls_inv(double *data, int size);

void snip2d(double *data, int nrows, int ncolumns, int width)
{
	int i, j;
	int p;
	int size;
	double *w;
	double P1, P2, P3, P4;
	double S1, S2, S3, S4;
	double dhelp;
	int	iminuspxncolumns; /* (i-p) * ncolumns */
	int	ixncolumns; /*  i * ncolumns */
	int	ipluspxncolumns; /* (i+p) * ncolumns */

	size = nrows * ncolumns;
	w = (double *) malloc(size * sizeof(double));

	for (p=width; p > 0; p--)
	{
		for (i=p; i<(nrows-p); i++)
		{
			iminuspxncolumns = (i-p) * ncolumns;
			ixncolumns = i * ncolumns;
			ipluspxncolumns = (i+p) * ncolumns;
			for (j=p; j<(ncolumns-p); j++)
			{
				P4 = data[ iminuspxncolumns + (j-p)]; /* P4 = data[i-p][j-p] */
				S4 = data[ iminuspxncolumns + j];     /* S4 = data[i-p][j]   */
				P2 = data[ iminuspxncolumns + (j+p)]; /* P2 = data[i-p][j+p] */
				S3 = data[ ixncolumns + (j-p)];       /* S3 = data[i][j-p]   */
				S2 = data[ ixncolumns + (j+p)];       /* S2 = data[i][j+p]   */
				P3 = data[ ipluspxncolumns + (j-p)];  /* P3 = data[i+p][j-p] */
				S1 = data[ ipluspxncolumns + j];      /* S1 = data[i+p][j]   */
				P1 = data[ ipluspxncolumns + (j+p)];  /* P1 = data[i+p][j+p] */
				dhelp = 0.5*(P1+P3);
				S1 = MAX(S1, dhelp) - dhelp;
				dhelp = 0.5*(P1+P2);
				S2 = MAX(S2, dhelp) - dhelp;
				dhelp = 0.5*(P3+P4);
				S3 = MAX(S3, dhelp) - dhelp;
				dhelp = 0.5*(P2+P4);
				S4 = MAX(S4, dhelp) - dhelp;
				w[ixncolumns + j] = MIN(data[ixncolumns + j], 0.5 * (S1+S2+S3+S4) + 0.25 * (P1+P2+P3+P4));
			}
		}
		for (i=p; i<(nrows-p); i++)
		{
			ixncolumns = i * ncolumns;
			for (j=p; j<(ncolumns-p); j++)
			{
				data[ixncolumns + j] = w[ixncolumns + j];
			}
		}
	}
	free(w);
}
