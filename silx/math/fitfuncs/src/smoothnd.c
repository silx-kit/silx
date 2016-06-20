#/*##########################################################################
#
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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define MAX_SAVITSKY_GOLAY_WIDTH 101
#define MIN_SAVITSKY_GOLAY_WIDTH 3

void smooth1d(double *data, int size);
void smooth2d(double *data, int size0, int size1);
void smooth3d(double *data, int size0, int size1, int size2);
int SavitskyGolay(double* input, long len_input, int npoints, double* output);

void smooth1d(double *data, int size)
{
	long i;
	double oldy;
	double newy;

	if (size < 3)
	{
		return;
	}
	oldy = data[0];
	for (i=0; i<(size-1); i++)
	{
		newy = 0.25 * (oldy + 2 * data[i] + data[i+1]);
		oldy = data[i];
		data[i] = newy;
	}
	data[size-1] = 0.25 * oldy + 0.75 * data[size-1];
	return;
}

void smooth2d(double *data, int size0, int size1)
{
	long i, j;
	double *p;

	/* smooth the first dimension */
	for (i=0; i < size0; i++)
	{
		smooth1d(&data[i*size1], size1);
	}

	/* smooth the 2nd dimension */
	p = (double *) malloc(size0 * sizeof(double));
	for (i=0; i < size1; i++)
	{
		for (j=0; j<size0; j++)
		{
			p[j] = data[j*size1+i];
		}
		smooth1d(p, size0);
	}
	free(p);
}

void smooth3d(double *data, int size0, int size1, int size2)
{
	long i, j, k, ihelp, jhelp;
	double *p;
	int size;


	size = size1*size2;

	/* smooth the first dimension */
	for (i=0; i < size0; i++)
	{
		smooth2d(&data[i*size], size1, size2);
	}

	/* smooth the 2nd dimension */
	size = size0 * size2;
	p = (double *) malloc(size * sizeof(double));

	for (i=0; i < size1; i++)
	{
		ihelp = i * size2;
		for (j=0; j<size0; j++)
		{
			jhelp = j * size1 * size2 + ihelp;
			for(k=0; k<size2; k++)
			{
				p[j*size2+k] = data[jhelp+k];
			}
		}
		smooth2d(p, size0, size2);
	}
	free(p);

	/* smooth the 3rd dimension */
	size = size0 * size1;
	p = (double *) malloc(size * sizeof(double));

	for (i=0; i < size2; i++)
	{
		for (j=0; j<size0; j++)
		{
			jhelp = j * size1 * size2 + i;
			for(k=0; k<size1; k++)
			{
				p[j*size1+k] = data[jhelp+k*size2];
			}
		}
		smooth2d(p, size0, size1);
	}
	free(p);
}


int SavitskyGolay(double* input, long len_input, int npoints, double* output)
{

    //double dpoints = 5.;
    double coeff[MAX_SAVITSKY_GOLAY_WIDTH];
    int i, j, m;
    double  dhelp, den;
    double  *data;

    memcpy(output, input, len_input * sizeof(double));

    if (!(npoints % 2)) npoints +=1;

    if((npoints < MIN_SAVITSKY_GOLAY_WIDTH) || (len_input < npoints))
    {
        /* do not smooth data */
        return 1;
    }

    /* calculate the coefficients */
    m     = (int) (npoints/2);
    den = (double) ((2*m-1) * (2*m+1) * (2*m + 3));
    for (i=0; i<= m; i++){
        coeff[m+i] = (double) (3 * (3*m*m + 3*m - 1 - 5*i*i ));
        coeff[m-i] = coeff[m+i];
    }

    /* simple smoothing at the beginning */
    for (j=0; j<=(int)(npoints/3); j++)
    {
        smooth1d(output, m);
    }

    /* simple smoothing at the end */
    for (j=0; j<=(int)(npoints/3); j++)
    {
        smooth1d((output+len_input-m-1), m);
    }

    /*one does not need the whole spectrum buffer, but code is clearer */
    data = (double *) malloc(len_input * sizeof(double));
    memcpy(data, output, len_input * sizeof(double));

    /* the actual SG smoothing in the middle */
    for (i=m; i<(len_input-m); i++){
        dhelp = 0;
        for (j=-m;j<=m;j++) {
            dhelp += coeff[m+j] * (*(data+i+j));
        }
        if(dhelp > 0.0){
            *(output+i) = dhelp / den;
        }
    }
    free(data);
    return (0);
}

