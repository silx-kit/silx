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

/* Wrapped functions */
void smooth1d(double *data, int size);
void smooth2d(double *data, int size0, int size1);
void smooth3d(double *data, int size0, int size1, int size2);
int SavitskyGolay(double* input, long len_input, int npoints, double* output);

/* Internal functions */
long index2d(long row_idx, long col_idx, long ncols);
long index3d(long x_idx, long y_idx, long z_idx, long ny, long nz);
void smooth1d_rows(double *data, long nrows, long ncols);
void smooth1d_cols(double *data, long nrows, long ncols);
void smooth1d_x(double *data, long nx, long ny, long nz);
void smooth1d_y(double *data, long nx, long ny, long nz);
void smooth1d_z(double *data, long nx, long ny, long nz);
void smooth2d_yzslice(double *data, long nx, long ny, long nz);
void smooth2d_xzslice(double *data, long nx, long ny, long nz);
void smooth2d_xyslice(double *data, long nx, long ny, long nz);


/* Simple smoothing of a 1D array */
void smooth1d(double *data, int size)
{
	long i;
	double prev_sample;
	double next_sample;

	if (size < 3)
	{
		return;
	}
	prev_sample = data[0];
	for (i=0; i<(size-1); i++)
	{
		next_sample = 0.25 * (prev_sample + 2 * data[i] + data[i+1]);
		prev_sample = data[i];
		data[i] = next_sample;
	}
	data[size-1] = 0.25 * prev_sample + 0.75 * data[size-1];
	return;
}

/* Smoothing of a 2D array*/
void smooth2d(double *data, int nrows, int ncols)
{
	/* smooth the first dimension (rows) */
    smooth1d_rows(data, nrows, ncols);

	/* smooth the 2nd dimension */
    smooth1d_cols(data, nrows, ncols);
}

/* Smoothing of a 3D array */
void smooth3d(double *data, int nx, int ny, int nz)
{
    smooth2d_xyslice(data, nx, ny, nz);
    smooth2d_xzslice(data, nx, ny, nz);
    smooth2d_yzslice(data, nx, ny, nz);
}

/* 1D Savitsky-Golay smoothing */
int SavitskyGolay(double* input, long len_input, int npoints, double* output)
{

    //double dpoints = 5.;
    double coeff[MAX_SAVITSKY_GOLAY_WIDTH];
    int i, j, m;
    double  dhelp, den;
    double  *data;

    memcpy(output, input, len_input * sizeof(double));

    if (!(npoints % 2)) npoints +=1;

    if((npoints < MIN_SAVITSKY_GOLAY_WIDTH) || (len_input < npoints) || \
       (npoints > MAX_SAVITSKY_GOLAY_WIDTH))
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

/*********************/
/* Utility functions */
/*********************/

long index2d(long row_idx, long col_idx, long ncols)
{
    return (row_idx*ncols+col_idx);
}

/* Apply smooth 1d on all rows in a 2D array*/
void smooth1d_rows(double *data, long nrows, long ncols)
{
	long row_idx;

	for (row_idx=0; row_idx < nrows; row_idx++)
	{
		smooth1d(&data[row_idx * ncols], ncols);
	}
}

/* Apply smooth 1d on all columns in a 2D array*/
void smooth1d_cols(double *data, long nrows, long ncols)
{
    long row_idx, col_idx;
    long this_idx2d, next_idx2d;
	double prev_sample;
	double next_sample;

	for (col_idx=0; col_idx < ncols; col_idx++)
	{
		prev_sample = data[index2d(0, col_idx, ncols)];
        for (row_idx=0; row_idx<(nrows-1); row_idx++)
        {
            this_idx2d = index2d(row_idx, col_idx, ncols);
            next_idx2d = index2d(row_idx+1, col_idx, ncols);

            next_sample = 0.25 * (prev_sample + \
                                  2 * data[this_idx2d] + \
                                  data[next_idx2d]);
            prev_sample = data[this_idx2d];
            data[this_idx2d] = next_sample;
        }

        this_idx2d = index2d(nrows-1, col_idx, ncols);
        data[this_idx2d] = 0.25 * prev_sample + 0.75 * data[this_idx2d];
	}
}

long index3d(long x_idx, long y_idx, long z_idx, long ny, long nz)
{
    return ((x_idx*ny + y_idx) * nz + z_idx);
}

/* Apply smooth 1d along first dimension in a 3D array*/
void smooth1d_x(double *data, long nx, long ny, long nz)
{
	long x_idx, y_idx, z_idx;
    long this_idx3d, next_idx3d;
	double prev_sample;
	double next_sample;

	for (y_idx=0; y_idx < ny; y_idx++)
	{
	    for (z_idx=0; z_idx < nz; z_idx++)
	    {
            prev_sample = data[index3d(0, y_idx, z_idx, ny, nz)];
            for (x_idx=0; x_idx<(nx-1); x_idx++)
            {
                this_idx3d = index3d(x_idx, y_idx, z_idx, ny, nz);
                next_idx3d = index3d(x_idx+1, y_idx, z_idx, ny, nz);

                next_sample = 0.25 * (prev_sample + \
                                      2 * data[this_idx3d] + \
                                      data[next_idx3d]);
                prev_sample = data[this_idx3d];
                data[this_idx3d] = next_sample;
            }

            this_idx3d = index3d(nx-1, y_idx, z_idx, ny, nz);
            data[this_idx3d] = 0.25 * prev_sample + 0.75 * data[this_idx3d];
        }
	}
}

/* Apply smooth 1d along second dimension in a 3D array*/
void smooth1d_y(double *data, long nx, long ny, long nz)
{
	long x_idx, y_idx, z_idx;
    long this_idx3d, next_idx3d;
	double prev_sample;
	double next_sample;

	for (x_idx=0; x_idx < nx; x_idx++)
	{
	    for (z_idx=0; z_idx < nz; z_idx++)
	    {
            prev_sample = data[index3d(x_idx, 0, z_idx, ny, nz)];
            for (y_idx=0; y_idx<(ny-1); y_idx++)
            {
                this_idx3d = index3d(x_idx, y_idx, z_idx, ny, nz);
                next_idx3d = index3d(x_idx, y_idx+1, z_idx, ny, nz);

                next_sample = 0.25 * (prev_sample + \
                                      2 * data[this_idx3d] + \
                                      data[next_idx3d]);
                prev_sample = data[this_idx3d];
                data[this_idx3d] = next_sample;
            }

            this_idx3d = index3d(x_idx, ny-1, z_idx, ny, nz);
            data[this_idx3d] = 0.25 * prev_sample + 0.75 * data[this_idx3d];
        }
	}
}

/* Apply smooth 1d along third dimension in a 3D array*/
void smooth1d_z(double *data, long nx, long ny, long nz)
{
	long x_idx, y_idx;
    long idx3d_first_sample;

	for (x_idx=0; x_idx < nx; x_idx++)
	{
	    for (y_idx=0; y_idx < ny; y_idx++)
	    {
	        idx3d_first_sample = index3d(x_idx, y_idx, 0, ny, nz);
	        /*We can use regular 1D smoothing function because z samples
	          are contiguous in memory*/
	        smooth1d(&data[idx3d_first_sample], nz);
        }
	}
}

/* 2D smoothing of a YZ slice in a 3D volume*/
void smooth2d_yzslice(double *data, long nx, long ny, long nz)
{
    long x_idx;
    long slice_size = ny * nz;

    /* a YZ slice is a "normal" 2D array of memory-contiguous data*/
	for (x_idx=0; x_idx < nx; x_idx++)
	{
		smooth2d(&data[x_idx*slice_size], ny, nz);
	}
}

/* 2D smoothing of a XZ slice in a 3D volume*/
void smooth2d_xzslice(double *data, long nx, long ny, long nz)
{

    /* smooth along the first dimension */
    smooth1d_x(data, nx, ny, nz);

    /* smooth along the third dimension */
    smooth1d_z(data, nx, ny, nz);
}

/* 2D smoothing of a XY slice in a 3D volume*/
void smooth2d_xyslice(double *data, long nx, long ny, long nz)
{
    /* smooth along the first dimension */
    smooth1d_x(data, nx, ny, nz);

    /* smooth along the second dimension */
    smooth1d_y(data, nx, ny, nz);
}

