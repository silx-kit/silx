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
   Implementation of the algorithm SNIP in 3D described in
   Miroslav Morhac et al. Nucl. Instruments and Methods in Physics Research A401 (1997) 113-132.
*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void lls(double *data, int size);
void lls_inv(double *data, int size);

void snip3d(double *data, int nx, int ny, int nz, int width)
{
	int i, j, k;
	int p;
	int size;
	double *w;
	double P1, P2, P3, P4, P5, P6, P7, P8;
	double R1, R2, R3, R4, R5, R6;
	double S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12;
	double dhelp;
	long ioffset;
	long iplus;
	long imin;
	long joffset;
	long jplus;
	long jmin;

	size = nx * ny * nz;
	w = (double *) malloc(size * sizeof(double));

	for (p=width; p > 0; p--)
	{
		for (i=p; i<(nx-p); i++)
		{
			ioffset = i * ny * nz;
			iplus = (i + p) * ny * nz;
			imin =  (i - p) * ny * nz;
			for (j=p; j<(ny-p); j++)
			{
				joffset = j * nz;
				jplus = (j + p) * nz;
				jmin =  (j - p) * nz;
				for (k=p; k<(nz-p); k++)
				{
					P1 = data[iplus + jplus + k-p];  /* P1 = data[i+p][j+p][k-p] */
					P2 = data[imin  + jplus + k-p];  /* P2 = data[i-p][j+p][k-p] */
					P3 = data[iplus + jmin  + k-p];  /* P3 = data[i+p][j-p][k-p] */
					P4 = data[imin  + jmin  + k-p];  /* P4 = data[i-p][j-p][k-p] */
					P5 = data[iplus + jplus + k+p];  /* P5 = data[i+p][j+p][k+p] */
					P6 = data[imin  + jplus + k+p];  /* P6 = data[i-p][j+p][k+p] */
					P7 = data[imin  + jmin  + k+p];  /* P7 = data[i-p][j-p][k+p] */
					P8 = data[iplus + jmin  + k+p];  /* P8 = data[i+p][j-p][k+p] */

					S1 = data[iplus   + joffset + k-p]; /* S1  = data[i+p][j][k-p] */
					S2 = data[ioffset + jmin    + k-p]; /* S2  = data[i][j+p][k-p] */
					S3 = data[imin    + joffset + k-p]; /* S3  = data[i-p][j][k-p] */
					S4 = data[ioffset + jmin    + k-p]; /* S4  = data[i][j-p][k-p] */
					S5 = data[imin    + joffset + k+p]; /* S5  = data[i-p][j][k+p] */
					S6 = data[ioffset + jplus   + k+p]; /* S6  = data[i][j+p][k+p] */
					S7 = data[imin    + joffset + k+p]; /* S7  = data[i-p][j][k+p] */
					S8 = data[ioffset + jmin    + k+p]; /* S8  = data[i][j-p][k+p] */
					S9 = data[imin    + jplus   + k];   /* S9  = data[i-p][j+p][k] */
					S10 = data[imin   + jmin    + k];   /* S10 = data[i-p][j-p][k] */
					S11 = data[iplus  + jmin    + k];   /* S11 = data[i+p][j-p][k] */
					S12 = data[iplus  + jplus   + k];   /* S12 = data[i+p][j+p][k] */

					R1 = data[ioffset + joffset + k-p]; /* R1 = data[i][j][k-p] */
					R2 = data[ioffset + joffset + k+p]; /* R2 = data[i][j][k+p] */
					R3 = data[imin    + joffset + k];   /* R3 = data[i-p][j][k] */
					R4 = data[iplus   + joffset + k];   /* R4 = data[i+p][j][k] */
					R5 = data[ioffset + jplus   + k];   /* R5 = data[i][j+p][k] */
					R6 = data[ioffset + jmin    + k];   /* R6 = data[i][j-p][k] */

					dhelp = 0.5*(P1+P3);
					S1 = MAX(S1, dhelp) - dhelp;

					dhelp = 0.5*(P1+P2);
					S2 = MAX(S2, dhelp) - dhelp;

					dhelp = 0.5*(P2+P4);
					S3 = MAX(S3, dhelp) - dhelp;

					dhelp = 0.5*(P3+P4);
					S4 = MAX(S4, dhelp) - dhelp;

					dhelp = 0.5*(P5+P8); /* Different from paper (P5+P7) but according to drawing */
					S5 = MAX(S5, dhelp) - dhelp;

					dhelp = 0.5*(P5+P6);
					S6 = MAX(S6, dhelp) - dhelp;

					dhelp = 0.5*(P6+P7); /* Different from paper (P6+P8) but according to drawing */
					S7 = MAX(S7, dhelp) - dhelp;

					dhelp = 0.5*(P7+P8);
					S8 = MAX(S8, dhelp) - dhelp;

					dhelp = 0.5*(P2+P6);
					S9 = MAX(S9, dhelp) - dhelp;

					dhelp = 0.5*(P4+P7); /* Different from paper (P4+P8) but according to drawing */
					S10 = MAX(S10, dhelp) - dhelp;

					dhelp = 0.5*(P3+P8); /* Different from paper (P1+P5) but according to drawing */
					S11 = MAX(S11, dhelp) - dhelp;

					dhelp = 0.5*(P1+P5); /* Different from paper (P3+P7) but according to drawing */
					S12 = MAX(S12, dhelp) - dhelp;

					/* The published formulae correspond to have:
					   P7 and P8 interchanged, and S11 and S12 interchanged
					   with respect to the published drawing */

					dhelp = 0.5 * (S1+S2+S3+S4)   + 0.25 * (P1+P2+P3+P4);
					R1 = MAX(R1, dhelp) - dhelp;

					dhelp = 0.5 * (S5+S6+S7+S8)   + 0.25 * (P5+P6+P7+P8);
					R2 = MAX(R2, dhelp) - dhelp;

					dhelp = 0.5 * (S3+S7+S9+S10)  + 0.25 * (P2+P4+P6+P7); /* Again same P7 and P8 change */
					R3 = MAX(R3, dhelp) - dhelp;

					dhelp = 0.5 * (S1+S5+S11+S12) + 0.25 * (P1+P3+P5+P8); /* Again same P7 and P8 change */
					R4 = MAX(R4, dhelp) - dhelp;

					dhelp = 0.5 * (S2+S6+S9+S12)  + 0.25 * (P1+P2+P5+P6); /* Again same S11 and S12 change */
					R5 = MAX(R5, dhelp) - dhelp;

					dhelp = 0.5 * (S4+S8+S10+S11) + 0.25 * (P3+P4+P7+P8); /* Again same S11 and S12 change */
					R6 = MAX(R6, dhelp) - dhelp;

					dhelp = 0.5   * (R1 + R2 + R3 + R4 + R5 + R6)  +\
						    0.25  * (S1 + S2 + S3 + S4 + S5 + S6) +\
							0.25  * (S7 + S8 + S9 + S10 + S11 + S12) +\
							0.125 * (P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8);
					w[ioffset + joffset + k] = MIN(data[ioffset + joffset + k], dhelp);
				}
			}
		}
		for (i=p; i<(nx-p); i++)
		{
			ioffset = i * ny * nz;
			for (j=p; j<(ny-p); j++)
			{
				joffset = j * nz;
				for (k=p; k<(nz-p); j++)
				{
					data[ioffset + joffset + k] = w[ioffset + joffset + k];
				}
			}
		}
	}
	free(w);
}
