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
kernel void kern_gradient2D(
    global float* slice,
    global float2* slice_grad,
    int sizeX,
    int sizeY)
{

    int gidx = (int) get_global_id(0);
    int gidy = (int) get_global_id(1);

    if ((gidx < sizeX) && (gidy < sizeY))
    {
    	// Note the direction inconstancy ! (JK 07/2018)

        float val_y = (gidx == (sizeX-1))? 0: slice[gidy*sizeX+gidx+1] - slice[gidy*sizeX+gidx];
        float val_x = (gidy == (sizeY-1))? 0: slice[(gidy+1)*sizeX+gidx] - slice[(gidy)*sizeX+gidx];

        slice_grad[gidy*sizeX+gidx].x = val_x;
        slice_grad[gidy*sizeX+gidx].y = val_y;
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
kernel void kern_divergence2D(
    global float2* slice_grad,
    global float* slice,
    int sizeX,
    int sizeY)
{
    int gidx = (int) get_global_id(0);
    int gidy = (int) get_global_id(1);

    if (gidx < sizeX && gidy < sizeY)
    {
        float val_x, val_y;
        val_y = (gidx == 0)?
        		slice_grad[(gidy)*sizeX+gidx].y :
        		slice_grad[(gidy)*sizeX+gidx].y - slice_grad[(gidy)*sizeX+gidx-1].y;
        val_x = (gidy == 0)?
        		slice_grad[(gidy)*sizeX+gidx].x:
                slice_grad[(gidy)*sizeX+gidx].x - slice_grad[(gidy-1)*sizeX+gidx].x;
        slice[gidy*sizeX+gidx] = val_x + val_y;
    }
}
