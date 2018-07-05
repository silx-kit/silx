/*
 *   Project: SIFT: An algorithm for image alignement
 *            preproces.cl: Kernels for image pre-processing, Normalization, ...
 *
 *
 *   Copyright (C) 2013-2018 European Synchrotron Radiation Facility
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
 **/



#ifndef WORKGROUP_SIZE
    #define WORKGROUP_SIZE 1024
#endif
    
#define MAX_CONST_SIZE 16384


/**
 * \brief Cast values of an array of uint8 into a float output array.
 *
 * :param array_int:     Pointer to global memory with the input data as unsigned8 array
 * :param array_float:   Pointer to global memory with the output data as float array
 * :param IMAGE_W:       Width of the image
 * :param IMAGE_H:       Height of the image
 */
kernel void
u8_to_float( global unsigned char  *array_int,
             global float *array_float,
             const int IMAGE_W,
             const int IMAGE_H
)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
        int i = gid0 + IMAGE_W * gid1;
    	array_float[i] = (float)array_int[i];
    } //end test in image
}//end kernel

/**
 * \brief cast values of an array of uint16 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the input data as unsigned16 array
 * :param array_float:  Pointer to global memory with the output data as float array
 * :param IMAGE_W:      Width of the image
 * :param IMAGE_H:      Height of the image
 */
kernel void
u16_to_float(global unsigned short  *array_int,
             global float *array_float,
             const int IMAGE_W,
             const int IMAGE_H
)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
    	int i = gid0 + IMAGE_W * gid1;
    	array_float[i] = (float)array_int[i];
    }
}//end kernel

/**
 * \brief cast values of an array of uint32 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the input data as unsigned32 array
 * :param array_float:  Pointer to global memory with the output data as float array
 * :param IMAGE_W:        Width of the image
 * :param IMAGE_H:         Height of the image
 */
kernel void
u32_to_float(global unsigned int  *array_int,
             global float *array_float,
             const int IMAGE_W,
             const int IMAGE_H)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
    	int i = gid0 + IMAGE_W * gid1;
    	array_float[i] = (float)array_int[i];
    }
}//end kernel

/**
 * \brief cast values of an array of uint64 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the input data as unsigned64 array
 * :param array_float:  Pointer to global memory with the output data as float array
 * :param IMAGE_W:        Width of the image
 * :param IMAGE_H:         Height of the image
 */
kernel void
u64_to_float(global unsigned long  *array_int,
             global float *array_float,
             const int IMAGE_W,
             const int IMAGE_H)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0<IMAGE_W) && (gid1 < IMAGE_H))
    {
        int i = gid0 + IMAGE_W * gid1;
        array_float[i] = (float)array_int[i];
    }
}//end kernel

/**
 * \brief convert values of an array of int32 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the data in int
 * :param array_float:  Pointer to global memory with the data in float
 * :param IMAGE_W:        Width of the image
 * :param IMAGE_H:         Height of the image
 */
kernel void
s32_to_float(    global int  *array_int,
                 global float  *array_float,
                 const int IMAGE_W,
                 const int IMAGE_H)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
    	int i = gid0 + IMAGE_W * gid1;
        array_float[i] = (float)(array_int[i]);
    }//end test in image
}//end kernel

/**
 * \brief convert values of an array of int64 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the data in int
 * :param array_float:  Pointer to global memory with the data in float
 * :param IMAGE_W:        Width of the image
 * :param IMAGE_H:         Height of the image
 */
kernel void
s64_to_float(    global long *array_int,
                global float  *array_float,
                 const int IMAGE_W,
                 const int IMAGE_H)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
    	int i = gid0 + IMAGE_W * gid1;
        array_float[i] = (float)(array_int[i]);
    }//end test in image
}//end kernel

/**
 * \brief convert values of an array of float64 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the data in double
 * :param array_float:  Pointer to global memory with the data in float
 * :param IMAGE_W:        Width of the image
 * :param IMAGE_H:         Height of the image
 *
 * COMMENTED OUT AS THIS RUNS ONLY ON GPU WITH FP64
 */
//kernel void
//double_to_float(global double *array_int,
//                global float  *array_float,
//                 const int IMAGE_W,
//                 const int IMAGE_H
//)
//{
//    int i = get_global_id(0) * IMAGE_W + get_global_id(1);
//    //Global memory guard for padding
//    if(i < IMAGE_W*IMAGE_H)
//        array_float[i] = (float)(array_int[i]);
//}//end kernel


/**
 * \brief convert RGB of an array of 3xuint8 into a float output array.
 *
 * :param array_int:    Pointer to global memory with the data in int
 * :param array_float:  Pointer to global memory with the data in float
 * :param IMAGE_W:        Width of the image
 * :param IMAGE_H:         Height of the image
 *
 * WARNING: still untested (formula is the same as PIL)
 */
kernel void
rgb_to_float(    global unsigned char *array_int,
                global float  *array_float,
                 const int IMAGE_W,
                 const int IMAGE_H)
{
	int gid0 = (int) get_global_id(0);
	int gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if ((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
    	int i = gid0 + IMAGE_W * gid1;
        array_float[i] = 0.299f*array_int[3*i] + 0.587f*array_int[3*i+1] + 0.114f*array_int[3*i+2];
    }  //end test in image
}//end kernel


/**
 * \brief Performs normalization of image between 0 and max_out (255) in place.
 *
 *
 * :param image        Float pointer to global memory storing the image.
 * :param min_in:     Minimum value in the input array
 * :param max_in:     Maximum value in the input array
 * :param max_out:     Maximum value in the output array (255 adviced)
 * :param IMAGE_W:    Width of the image
 * :param IMAGE_H:     Height of the image
 *
**/
kernel void
normalizes(    global       float     *image,
            constant        float * min_in __attribute__((max_constant_size(MAX_CONST_SIZE))),
            constant        float * max_in __attribute__((max_constant_size(MAX_CONST_SIZE))),
            constant        float * max_out __attribute__((max_constant_size(MAX_CONST_SIZE))),
            const             int IMAGE_W,
            const             int IMAGE_H
)
{
    //Global memory guard for padding
    int gid0 = (int) get_global_id(0);
    int gid1 = (int) get_global_id(1);

    if((gid0 < IMAGE_W) && (gid1 < IMAGE_H))
    {
        int i = gid0 + IMAGE_W * gid1;
        image[i] = max_out[0]*(image[i]-min_in[0])/(max_in[0]-min_in[0]);
    };//end if in IMAGE
};//end kernel

/**
 * \brief shrink: Subsampling of the image_in into a smaller image_out.
 *
 *
 * :param image_in        Float pointer to global memory storing the big image.
 * :param image_ou        Float pointer to global memory storing the small image.
 * :param scale_w:     Minimum value in the input array
 * :param scale_h:     Maximum value in the input array
 * :param IMAGE_W:    Width of the output image
 * :param IMAGE_H:     Height of the output image
 *
**/
kernel void
shrink(const global     float     *image_in,
            global     float     *image_out,
            const             int scale_w,
            const             int scale_h,
            const             int LARGE_W,
            const             int LARGE_H,
            const             int SMALL_W,
            const             int SMALL_H
)
{
    int gid0 = (int) get_global_id(0), 
        gid1 = (int) get_global_id(1);
    int j,i = gid0 + SMALL_W * gid1;
    //Global memory guard for padding
    if ((gid0 < SMALL_W) && (gid1 <SMALL_H))
    {
        j = gid0 * scale_w + gid1 * scale_h * LARGE_W;
        image_out[i] = image_in[j];
    };//end if in IMAGE
};//end kernel


/**
 * \brief bin: resampling of the image_in into a smaller image_out with higher dynamics.
 *
 *
 * :param image_in        Float pointer to global memory storing the big image.
 * :param image_ou        Float pointer to global memory storing the small image.
 * :param scale_width:    Binning factor in horizontal           
 * :param scale_heigth:   Binning factor in vertical
 * :param orig_width:     Original image size in horizontal
 * :param orig_heigth:    Original image size in vertical
 * :param binned_width:   Width of the output binned image
 * :param binned_heigth:  Height of the output binned image
 *
 * Nota: this is a 2D kernel. This is non working and non TESTED !!!
**/
kernel void
bin(        const    global     float     *image_in,
                     global     float     *image_out,
            const                 int     scale_width,
            const                 int     scale_heigth,
            const                 int     orig_width,
            const                 int     orig_heigth,
            const                 int     binned_width,
            const                 int     binned_heigth
)
{
    int gid0 = (int) get_global_id(0), 
        gid1 = (int) get_global_id(1);
    //Global memory guard for padding
    if((gid0 < binned_width) && (gid1 < binned_heigth) )
    {
        int j,i = gid0 + binned_width * gid1;
        float data=0.0f;
        int w, h, big_h, big_w;
        for (h=gid1 * scale_heigth; h<(gid1+1) * scale_heigth; h++){
            if (h>=orig_heigth){
                big_h = 2*orig_heigth - h - 1;
            }else{
                big_h = h;
            }
            for (w=gid0*scale_width; w<(gid0+1)*scale_width; w++)
            {
                if (w>=orig_width){
                	big_w = 2*orig_width - w - 1;
                }else{
                	big_w = w;
                }
                j = big_h * orig_width + big_w;
                data += image_in[j];
            };//end for horiz
        };//end for vertical
        image_out[i] = data/((float)(scale_width*scale_heigth));
    };//end if in IMAGE
};//end kernel


/**
 * \brief divide_cst: divide a vector by a constant.
 *
 *
 * :param data:     Float pointer to global memory storing the vector.
 * :param value:    calc data/value
 * :param size:     size of the vector
 *
**/

kernel void
divide_cst( global     float     *data,
            global     float     *value,
            const        int     SIZE)
{
    int gid = (int) get_global_id(0);
    if (gid < SIZE)
    {
        data[gid] = data[gid] / value[0];
    }
}

