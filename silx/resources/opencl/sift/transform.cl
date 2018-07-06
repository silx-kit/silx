/*
 *   Project: SIFT: An algorithm for image alignement
 *            kernel for maximum and minimum calculation
 *
 *
 *   Copyright (C) 2013-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *
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
 *
 *
 */

/*
 *
 *        Computes the transformation to correct the image given a set of parameters
 *                [[a b c]
 *                [d e f]]
 *
 *                = [matrix, offset]
 *
 * :param image: Pointer to global memory with the input image
 * :param output: Pointer to global memory with the outpu image
 * :param matrix: "float4" struct for the transformation matrix
 * :param offset: "float2" struct for the offset vector
 * :param image_width Image width
 * :param image_height Image height
 * :param output_width Output width, can differ from image width
 * :param output_height Ouput height, can differ from image height
 * :param fill: Default value to fill the image with
 * :param mode: Interpolation mode. 0 = no interpolation, 1 = bilinear interpolation
 *
 */

kernel void transform(
        global float* image,
        global float* output,
        global float4* matrix,
        global float2* offset,
        int image_width,
        int image_height,
        int output_width,
        int output_height,
        float fill,
        int mode)
{
        int gid0 = (int) get_global_id(0);
        int gid1 = (int) get_global_id(1);
        float4 mat = *matrix;
        float2 off  = *offset;

        if (!(gid0 < output_width && gid1 < output_height))
                return;

        int x = gid0;
        int y = gid1;

        float tx = dot(mat.s23,(float2) (y,x)); //be careful to the order that differs from Python...Here Fortran convention is used
        float ty = dot(mat.s01,(float2) (y,x));

        tx += off.s1;
        ty += off.s0;

        int tx_next = ((int) tx) + 1;
        int tx_prev = (int) tx;
        int ty_next = ((int) ty) + 1;
        int ty_prev = (int) ty;

        float interp = fill;

        if (0.0f <= tx && tx < image_width && 0.0f <= ty && ty < image_height) 
        {


                if (mode == 1) 
                { //bilinear interpolation
                        float image_p = image[ty_prev*image_width+tx_prev],
                                image_x = image[ty_prev*image_width+tx_next],
                                image_y = image[ty_next*image_width+tx_prev],
                                image_n = image[ty_next*image_width+tx_next];

                        if (tx_next >= image_width) 
                        {
                                image_x = fill;
                                image_n = fill;
                        }
                        if (ty_next >= image_height) 
                        {
                                image_y = fill;
                                image_n = fill;
                        }

                        //bilinear interpolation
                        float interp1 = ((float) (tx_next - tx)) * image_p
                                                  + ((float) (tx - tx_prev)) * image_x,

                                interp2 = ((float) (tx_next - tx)) * image_y
                                                + ((float) (tx - tx_prev)) * image_n;

                        interp = ((float) (ty_next - ty)) * interp1
                                   + ((float) (ty - ty_prev)) * interp2;

                }

                else 
                { //no interpolation
                        interp = image[((int) ty)*image_width+((int) tx)];
                }
        }

        //to be coherent with scipy.ndimage.interpolation.affine_transform
        float u = -1.0; //-0.5; //-0.95
        float v = -1.0; //-0.5;
        if (tx >= image_width+u) 
        {
                interp = fill;
        }
        if (ty >= image_height+v) 
        {
                interp = fill;
        }


        output[gid1*output_width+gid0] = interp;

}

/*
 * Same as previously except that dim0 [0..4[ is the color (R,G,B), 4 is never used
 *                                dim1 [0..width[
 *                                dim2 [0..height[
 *
 */
kernel void transform_RGB(
                        global unsigned char* image,
                        global unsigned char* output,
                        global float4* matrix,
                        global float2* offset,
                        int image_width,
                        int image_height,
                        int output_width,
                        int output_height,
                        float fill,
                        int mode)
{
        int color = get_global_id(0);
        int gid0 = get_global_id(1);
        int gid1 = get_global_id(2);
        float4 mat = *matrix;
        float2 off  = *offset;

        if (!(gid0 < output_width && gid1 < output_height && color<3))
                return;

        int x = gid0,
                y = gid1;

        float tx = dot(mat.s23,(float2) (y,x)), //be careful to the order that differs from Python...Here Fortran convention is used
                ty = dot(mat.s01,(float2) (y,x));

        tx += off.s1;
        ty += off.s0;

        int tx_next = ((int) tx) + 1;
        int tx_prev = (int) tx;
        int ty_next = ((int) ty) + 1;
        int ty_prev = (int) ty;

        float interp = fill;

        if (0.0f <= tx && tx < image_width && 0.0f <= ty && ty < image_height) 
        {
                if (mode == 1) 
                { //bilinear interpolation
                        float image_p = image[3*(ty_prev*image_width+tx_prev) + color],
                                image_x = image[3*(ty_prev*image_width+tx_next) + color],
                                image_y = image[3*(ty_next*image_width+tx_prev) + color],
                                image_n = image[3*(ty_next*image_width+tx_next) + color];

                        if (tx_next >= image_width) 
                        {
                                image_x = fill;
                                image_n = fill;
                        }
                        if (ty_next >= image_height) 
                        {
                                image_y = fill;
                                image_n = fill;
                        }

                        //bilinear interpolation
                        float interp1 = ((float) (tx_next - tx)) * image_p
                                                  + ((float) (tx - tx_prev)) * image_x,

                                interp2 = ((float) (tx_next - tx)) * image_y
                                                + ((float) (tx - tx_prev)) * image_n;

                        interp = ((float) (ty_next - ty)) * interp1
                                   + ((float) (ty - ty_prev)) * interp2;

                }

                else 
                { //no interpolation
                        interp = image[ 3 * (((int) ty)*image_width+((int) tx)) + color];
                }
        }


        //to be coherent with scipy.ndimage.interpolation.affine_transform
        float u = -0.5; //-0.95
        float v = -0.5;
        if (tx >= image_width+u) 
        {
                interp = fill;
        }
        if (ty >= image_height+v) 
        {
                interp = fill;
        }

        output[3*(gid1*output_width+gid0) + color] = interp;

}
