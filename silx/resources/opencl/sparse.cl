/*
 *   Copyright (C) 2016-2019 European Synchrotron Radiation Facility
 *                           Grenoble, France
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

#ifndef IMAGE_WIDTH
  #error "Please define IMAGE_WIDTH parameter"
#endif

#ifndef DTYPE
  #error "Please define DTYPE parameter"
#endif

#ifndef IDX_DTYPE
  #error "Please define IDX_DTYPE parameter"
#endif



/**
 * Densify a matric from CSR format to "dense" 2D format.
 * The input CSR data consists in 3 arrays: (data, ind, iptr).
 * The output array is a 2D array of dimensions IMAGE_WIDTH * image_height.
 *
 * data: 1D array containing the nonzero data items.
 * ind: 1D array containing the column indices of the nonzero data items.
 * iptr: 1D array containing indirection indices, such that range
 *    [iptr[i], iptr[i+1]-1] of "data" and "ind" contain the relevant data
 *    of output row "i".
 * output: 2D array containing the densified data.
 * image_height: height (number of rows) of the output data.
**/

kernel void densify_csr(
    const global DTYPE* data,
    const global IDX_DTYPE* ind,
    const global IDX_DTYPE* iptr,
    global DTYPE* output,
    int image_height
)
{
    uint tid = get_local_id(0);
    uint row_idx = get_global_id(1);
    if ((tid >= IMAGE_WIDTH) || (row_idx >= image_height)) return;

    local DTYPE line[IMAGE_WIDTH];

    // Memset
    //~ #pragma unroll
    for (int k = 0; tid+k < IMAGE_WIDTH; k += get_local_size(0)) {
        if (tid+k >= IMAGE_WIDTH) break;
        line[tid+k] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    uint start = iptr[row_idx], end = iptr[row_idx+1];
    //~ #pragma unroll
    for (int k = start; k < end; k += get_local_size(0)) {
        // Current work group handles one line of the final array
        // on the current line, write data[start+tid] at column index ind[start+tid]
        if (k+tid < end)
            line[ind[k+tid]] = data[k+tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write the current line (shared mem) into the output array (global mem)
    //~ #pragma unroll
    for (int k = 0; tid+k < IMAGE_WIDTH; k += get_local_size(0)) {
        output[row_idx*IMAGE_WIDTH + tid+k] = line[tid+k];
        if (k+tid >= IMAGE_WIDTH) return;
    }
}
