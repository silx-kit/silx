/*
 *   Project: SILX: Alogorithms for image processing
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
 */

#ifndef NB_COLOR
    #define NB_COLOR 1
#endif


/**
 * \brief Cast values of an array of uint8 into a float output array.
 *
 * :param array_input:   Pointer to global memory with the input data as unsigned8 array
 * :param array_float:   Pointer to global memory with the output data as float array
 * :param width:         Width of the image
 * :param height:        Height of the image
 */
kernel void u8_to_float( global unsigned char  *array_input,
                         global float *array_float,
                         const int width,
                         const int height)
{
    //Global memory guard for padding
    if ((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int i = NB_COLOR * (get_global_id(0) + width * get_global_id(1));
        for (int c=0; c<NB_COLOR; c++)
        {
            array_float[i + c] = (float) array_input[i + c];
        } //end loop over colors
    } //end test in image
} //end kernel

/**
 * \brief Cast values of an array of uint8 into a float output array.
 *
 * :param array_input:   Pointer to global memory with the input data as signed8 array
 * :param array_float:   Pointer to global memory with the output data as float array
 * :param width:         Width of the image
 * :param height:        Height of the image
 */
kernel void s8_to_float( global char  *array_input,
                         global float *array_float,
                         const int width,
                         const int height)
{
    //Global memory guard for padding
    if ((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int i = NB_COLOR * (get_global_id(0) + width * get_global_id(1));
        for (int c=0; c<NB_COLOR; c++)
        {
            array_float[i + c] = (float) array_input[i + c];
        } //end loop over colors
    } //end test in image
} //end kernel


/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * :param array_input:    Pointer to global memory with the input data as unsigned16 array
 * :param array_float:    Pointer to global memory with the output data as float array
 * :param width:          Width of the image
 * :param height:         Height of the image
 */
kernel void u16_to_float(global unsigned short  *array_input,
                         global float *array_float,
                         const int width,
                         const int height)
{
    //Global memory guard for padding
    if ((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int i = NB_COLOR * (get_global_id(0) + width * get_global_id(1));
        for (int c=0; c<NB_COLOR; c++)
        {
            array_float[i + c] = (float) array_input[i + c];
        }           //end loop over colors
    } //end test in image
} //end kernel

/**
 * \brief cast values of an array of int16 into a float output array.
 *
 * :param array_input:    Pointer to global memory with the input data as signed16 array
 * :param array_float:    Pointer to global memory with the output data as float array
 * :param width:          Width of the image
 * :param height:         Height of the image
 */
kernel void s16_to_float(global short *array_input,
                         global float *array_float,
                         const int width,
                         const int height)
{
    //Global memory guard for padding
    if ((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int i = NB_COLOR * (get_global_id(0) + width * get_global_id(1));
        for (int c=0; c<NB_COLOR; c++)
        {
            array_float[i + c] = (float) array_input[i + c];
        } //end loop over colors
    } //end test in image
}//end kernel

/**
 * \brief cast values of an array of uint32 into a float output array.
 *
 * :param array_input:    Pointer to global memory with the input data as unsigned32 array
 * :param array_float:    Pointer to global memory with the output data as float array
 * :param width:          Width of the image
 * :param height:         Height of the image
 */
kernel void u32_to_float(global unsigned int  *array_input,
                         global float *array_float,
                         const int width,
                         const int height)
{
    //Global memory guard for padding
    if ((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int i = NB_COLOR * (get_global_id(0) + width * get_global_id(1));
        for (int c=0; c<NB_COLOR; c++)
        {
            array_float[i + c] = (float) array_input[i + c];
        } //end loop over colors
    } //end test in image
}//end kernel


/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * :param array_input:    Pointer to global memory with the data in int
 * :param array_float:    Pointer to global memory with the data in float
 * :param width:          Width of the image
 * :param height:         Height of the image
 */
kernel void s32_to_float(global int  *array_input,
                         global float  *array_float,
                         const int width,
                         const int height)
{
    //Global memory guard for padding
    if ((get_global_id(0) < width) && (get_global_id(1)<height))
    {
        int i = NB_COLOR * (get_global_id(0) + width * get_global_id(1));
        for (int c=0; c<NB_COLOR; c++)
        {
            array_float[i + c] = (float) array_input[i + c];
        } //end loop over colors
    } //end test in image
}//end kernel

