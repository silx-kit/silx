/*
 *   Project: SILX: Data analysis library
 *            kernel for maximum and minimum calculation
 *
 *
 *   Copyright (C) 2013-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
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

#ifndef NB_COLOR
    #define NB_COLOR 1
#endif


/**
 * \brief Linear scale of signal in an image (maybe multi-color)
 *
 * :param image: contains the input image
 * :param output: contains the output image after scaling
 * :param max_min_input: 2-floats containing the maximum and the minimum value of image
 * :param minimum: float scalar with the minimum of the output
 * :param maximum: float scalar with the maximum of the output
 * :param width: integer, number of columns the matrices
 * :param height: integer, number of lines of the matrices
 *
 *
 */


kernel void normalize_image(global float* image,
                            global float* output,
                            int width,
                            int height,
                            global float* max_min_input,
                            float minimum,
                            float maximum
                            )
{
    //Global memory guard for padding
    if((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int idx = NB_COLOR* (get_global_id(0) + width * get_global_id(1));
        float mini_in, maxi_in, scale;
        maxi_in = max_min_input[0];
        mini_in = max_min_input[1];
        if (maxi_in == mini_in)
        {
            scale = NAN;
        }
        else
        {
            scale = (maximum - minimum) / (maxi_in - mini_in);
        }

        for (int c=0; c<NB_COLOR; c++)
        {
            output[idx+c] = fma(scale, image[idx+c]-mini_in, minimum);
        } // end color loop
    }//end if in IMAGE
}//end kernel
