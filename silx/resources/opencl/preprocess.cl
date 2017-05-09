/*
 *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
 *            Preprocessing program
 *
 *
 *   Copyright (C) 2012-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 19/01/2017
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
 */

/**
 * \file
 *
 * \brief OpenCL kernels for image array casting, array mem-setting and normalizing
 *
 * Constant to be provided at build time:
 *   NIMAGE: size of the image
 *
 */

#include "for_eclipse.h"

/**
 * \brief cast values of an array of int8 into a float output array.
 *
 * - array_s8: Pointer to global memory with the input data as signed8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
s8_to_float(__global char  *array_s8,
            __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_s8[i];
}


/**
 * \brief cast values of an array of uint8 into a float output array.
 *
 * - array_u8: Pointer to global memory with the input data as unsigned8 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u8_to_float(__global unsigned char  *array_u8,
            __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_u8[i];
}


/**
 * \brief cast values of an array of int16 into a float output array.
 *
 * - array_s16: Pointer to global memory with the input data as signed16 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
s16_to_float(__global short *array_s16,
             __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_s16[i];
}


/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * - array_u16: Pointer to global memory with the input data as unsigned16 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u16_to_float(__global unsigned short  *array_u16,
             __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_u16[i];
}

/**
 * \brief cast values of an array of uint32 into a float output array.
 *
 * - array_u32: Pointer to global memory with the input data as unsigned32 array
 * - array_float:  Pointer to global memory with the output data as float array
 */
__kernel void
u32_to_float(__global unsigned int  *array_u32,
             __global float *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)array_u32[i];
}

/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * - array_int:  Pointer to global memory with the data as unsigned32 array
 * - array_float:  Pointer to global memory with the data float array
 */
__kernel void
s32_to_float(__global int  *array_int,
             __global float  *array_float
)
{
  int i = get_global_id(0);
  //Global memory guard for padding
  if (i < NIMAGE)
    array_float[i] = (float)(array_int[i]);
}


/**
 * Functions to be called from an actual kernel.
 * \brief Performs the normalization of input image by dark subtraction,
 *        variance is propagated to second member of the float3
 *        flatfield, solid angle, polarization and absorption are stored in
 *        third member of the float3 returned.
 *
 * Invalid/Dummy pixels will all have the third-member set to 0, i.e. no weight
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall polarization correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_absorption   Bool/int: shall absorption correction be applied ?
 * - absorption      Float pointer to global memory storing the effective absoption of each pixel.
 * - do_mask         perform mask correction ?
 * - mask            Bool/char pointer to mask array
 * - do_dummy        Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy           Float: value for bad pixels
 * - delta_dummy     Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
**/

static float3 _preproc3(const __global float  *image,
                        const __global float  *variance,
                        const          char   do_dark,
                        const __global float  *dark,
                        const          char   do_dark_variance,
                        const __global float  *dark_variance,
                        const          char   do_flat,
                        const __global float  *flat,
                        const          char   do_solidangle,
                        const __global float  *solidangle,
                        const          char   do_polarization,
                        const __global float  *polarization,
                        const          char   do_absorption,
                        const __global float  *absorption,
                        const          char   do_mask,
                        const __global char   *mask,
                        const          char   do_dummy,
                        const          float  dummy,
                        const          float  delta_dummy,
                        const          float  normalization_factor)
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        if ((!do_mask) || (!mask[i]))
        {
            result.s0 = image[i];
            if (variance != 0)
                result.s1 = variance[i];
            result.s2 = normalization_factor;
            if ( (!do_dummy)
                  ||((delta_dummy != 0.0f) && (fabs(result.s0-dummy) > delta_dummy))
                  ||((delta_dummy == 0.0f) && (result.s0 != dummy)))
            {
                if (do_dark)
                    result.s0 -= dark[i];
                if (do_dark_variance)
                    result.s1 += dark_variance[i];
                if (do_flat)
                {
                    float one_flat = flat[i];
                    if ( (!do_dummy)
                         ||((delta_dummy != 0.0f) && (fabs(one_flat-dummy) > delta_dummy))
                         ||((delta_dummy == 0.0f) && (one_flat != dummy)))
                        result.s2 *= one_flat;
                    else
                        result.s2 = 0.0f;
                }
                if (do_solidangle)
                    result.s2 *= solidangle[i];
                if (do_polarization)
                    result.s2 *= polarization[i];
                if (do_absorption)
                    result.s2 *= absorption[i];
                if (isnan(result.s0) || isnan(result.s1) || isnan(result.s2) || (result.s2 == 0.0f))
                    result = (float3)(0.0, 0.0, 0.0);
            }
            else
            {
                result = (float3)(0.0, 0.0, 0.0);
            }//end if do_dummy
        } // end if mask
    };//end if NIMAGE
    return result;
};//end function


/**
 * \brief Performs the normalization of input image by dark subtraction,
 *        flatfield, solid angle, polarization and absorption division.
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall polarization correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_absorption   Bool/int: shall absorption correction be applied ?
 * - absorption      Float pointer to global memory storing the effective absoption of each pixel.
 * - do_mask         perform mask correction ?
 * - mask            Bool/char pointer to mask array
 * - do_dummy        Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy           Float: value for bad pixels
 * - delta_dummy     Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
**/

__kernel void
corrections(const __global float  *image,
            const          char   do_dark,
            const __global float  *dark,
            const          char   do_flat,
            const __global float  *flat,
            const          char   do_solidangle,
            const __global float  *solidangle,
            const          char   do_polarization,
            const __global float  *polarization,
			const          char   do_absorption,
			const __global float  *absorption,
            const          char   do_mask,
            const __global char   *mask,
            const          char   do_dummy,
            const          float  dummy,
            const          float  delta_dummy,
            const          float  normalization_factor,
                  __global float  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                            0,
                            do_dark,
                            dark,
                            0,
                            0,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_mask,
                            mask,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        if (result.s2 != 0.0f)
            output[i] = result.s0 / result.s2;
        else
            output[i] = dummy;
    };//end if NIMAGE

};//end kernel


/**
 * \brief Performs Normalization of input image with float2 output (num,denom)
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (denominator)
 *  - polarization correction (denominator)
 *  - flat fiels correction (denominator)
 *
 * Corrections are made out of place.
 * Dummy pixels set both the numerator and denominator to 0
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy          Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy             Float: value for bad pixels
 * - delta_dummy       Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
 *
**/
__kernel void
corrections2(const __global float  *image,
             const          char   do_dark,
             const __global float  *dark,
             const          char   do_flat,
             const __global float  *flat,
             const          char   do_solidangle,
             const __global float  *solidangle,
             const          char   do_polarization,
             const __global float  *polarization,
             const          char   do_absorption,
             const __global float  *absorption,
             const          char   do_mask,
             const __global char   *mask,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
                   __global float2  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                            0,
                            do_dark,
                            dark,
                            0,
                            0,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_mask,
                            mask,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        output[i] = (float2)(result.s0, result.s2);
    };//end if NIMAGE
};//end kernel

/**
 * \brief Performs Normalization of input image with float3 output (signal, variance, normalization) assuming poissonian signal
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (denominator)
 *  - polarization correction (denominator)
 *  - flat fiels correction (denominator)
 *
 * Corrections are made out of place.
 * Dummy pixels set both the numerator and denominator to 0
 *
 * - image           Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy          Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy             Float: value for bad pixels
 * - delta_dummy       Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
 *
**/
__kernel void
corrections3Poisson( const __global float  *image,
                     const          char   do_dark,
                     const __global float  *dark,
                     const          char   do_flat,
                     const __global float  *flat,
                     const          char   do_solidangle,
                     const __global float  *solidangle,
                     const          char   do_polarization,
                     const __global float  *polarization,
                     const          char   do_absorption,
                     const __global float  *absorption,
                     const          char   do_mask,
                     const __global char   *mask,
                     const          char   do_dummy,
                     const          float  dummy,
                     const          float  delta_dummy,
                     const          float  normalization_factor,
                           __global float3  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3(image,
                           image,
                           do_dark,
                           dark,
                           do_dark,
                           dark,
                           do_flat,
                           flat,
                           do_solidangle,
                           solidangle,
                           do_polarization,
                           polarization,
                           do_absorption,
                           absorption,
                           do_mask,
                           mask,
                           do_dummy,
                           dummy,
                           delta_dummy,
                           normalization_factor);
        output[i] = result;
    };//end if NIMAGE
};//end kernel


/**
 * \brief Performs Normalization of input image with float3 output (signal, variance, normalization)
 *
 * Intensities of images are corrected by:
 *  - dark (read-out) noise subtraction for the data
 *  - Solid angle correction (division)
 *  - polarization correction (division)
 *  - flat fiels correction (division)
 * Corrections are made in place unless the pixel is dummy.
 * Dummy pixels are left untouched so that they remain dummy
 *
 * - image              Float pointer to global memory storing the input image.
 * - do_dark         Bool/int: shall dark-current correction be applied ?
 * - dark            Float pointer to global memory storing the dark image.
 * - do_flat         Bool/int: shall flat-field correction be applied ?
 * - flat            Float pointer to global memory storing the flat image.
 * - do_solidangle   Bool/int: shall flat-field correction be applied ?
 * - solidangle      Float pointer to global memory storing the solid angle of each pixel.
 * - do_polarization Bool/int: shall flat-field correction be applied ?
 * - polarization    Float pointer to global memory storing the polarization of each pixel.
 * - do_dummy          Bool/int: shall the dummy pixel be checked. Dummy pixel are pixels marked as bad and ignored
 * - dummy             Float: value for bad pixels
 * - delta_dummy       Float: precision for bad pixel value
 * - normalization_factor : divide the input by this value
 *
 *
**/

__kernel void
corrections3(const __global float  *image,
             const __global float  *variance,
             const          char   do_dark,
             const __global float  *dark,
             const          char   do_dark_variance,
             const __global float  *dark_variance,
             const          char   do_flat,
             const __global float  *flat,
             const          char   do_solidangle,
             const __global float  *solidangle,
             const          char   do_polarization,
             const __global float  *polarization,
             const          char   do_absorption,
             const __global float  *absorption,
             const          char   do_mask,
             const __global char   *mask,
             const          char   do_dummy,
             const          float  dummy,
             const          float  delta_dummy,
             const          float  normalization_factor,
                   __global float3  *output
            )
{
    size_t i= get_global_id(0);
    float3 result = (float3)(0.0, 0.0, 0.0);
    if (i < NIMAGE)
    {
        result = _preproc3( image,
                            variance,
                            do_dark,
                            dark,
                            do_dark_variance,
                            dark_variance,
                            do_flat,
                            flat,
                            do_solidangle,
                            solidangle,
                            do_polarization,
                            polarization,
                            do_absorption,
                            absorption,
                            do_mask,
                            mask,
                            do_dummy,
                            dummy,
                            delta_dummy,
                            normalization_factor);
        output[i] = result;
    };//end if NIMAGE
};//end kernel


