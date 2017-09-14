/*
 *   Copyright (C) 2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
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
 *
 */

/**
 *
 * Compute the spatial gradient of an image.
 *
 * slice: input image
 * slice_grad: output gradient
 * sizeX: number of columns of the image
 * sizeY: number of rows of the image
 *
 **/
__kernel void kern_gradient2D(
    __global float* slice,
    __global float2* slice_grad,
    int sizeX,
    int sizeY)
{

    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    float val_x = 0, val_y = 0;

    if (gidx < sizeX && gidy < sizeY) {
        if (gidx == sizeX-1) val_y = 0;
        else val_y = slice[(gidy)*sizeX+gidx+1] - slice[(gidy)*sizeX+gidx];
        if (gidy == sizeY-1) val_x = 0;
        else val_x = slice[(gidy+1)*sizeX+gidx] - slice[(gidy)*sizeX+gidx];

        slice_grad[(gidy)*sizeX+gidx].x = val_x;
        slice_grad[(gidy)*sizeX+gidx].y = val_y;
    }
}

/**
 *
 * Compute the spatial divergence of an image gradient.
 *
 * slice_grad: input gradient-like image
 * slice: output image
 * sizeX: number of columns of the input
 * sizeY: number of rows of the input
 *
 **/
__kernel void kern_divergence2D(
    __global float2* slice_grad,
    __global float* slice,
    int sizeX,
    int sizeY)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    float val_x = 0, val_y = 0;

    if (gidx < sizeX && gidy < sizeY) {
        if (gidx == 0) val_y = slice_grad[(gidy)*sizeX+gidx].y;
        else val_y = slice_grad[(gidy)*sizeX+gidx].y - slice_grad[(gidy)*sizeX+gidx-1].y;
        if (gidy == 0) val_x = slice_grad[(gidy)*sizeX+gidx].x;
        else val_x = slice_grad[(gidy)*sizeX+gidx].x - slice_grad[(gidy-1)*sizeX+gidx].x;
        slice[(gidy)*sizeX+gidx] = val_x + val_y;
    }
}




