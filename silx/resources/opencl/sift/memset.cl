/*
 *   Project: SIFT: An algorithm for image alignement
 *            Kernel for image pre-processing: Normalization, ...
 *
 *
 *   Copyright (C) 2013 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *   All rights reserved.
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *   Last revision: 16/07/2013
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
/*
	Keypoint (x, y, scale, angle) without its descriptor
*/
typedef float4 keypoint;


/*
	Keypoint with its descriptor
*/
typedef struct t_keypoint {
	float x, y, scale, angle;
	unsigned char desc[128];
} t_keypoint;


/**
 * \brief Fills a float-array with the given value.
 *
 * :param array:         Pointer to global memory with the data as float array
 * :param value:         Value used for filling
 * :param SIZE:          Size if the array
 */


__kernel void
memset_float( __global float *array,
				const float value,
				const int SIZE
){
	int gid = get_global_id(0);
	if (gid<SIZE){
		array[gid] = value;
	}
}

/**
 * \brief Fills a int-array with the given value.
 *
 * :param array:         Pointer to global memory with the data as float array
 * :param value:         Value used for filling
 * :param SIZE:          Size if the array
 */


__kernel void
memset_int( __global int *array,
				const int value,
				const int SIZE
){
	int gid = get_global_id(0);
	if (gid<SIZE){
		array[gid] = value;
	}
}

/**
 * \brief Fills an array of keypoints with the given value.
 *
 * :param array:         Pointer to global memory with the data as float array
 * :param value:         Value used for filling
 * :param SIZE:          Size if the array
 */


__kernel void
memset_kp( __global t_keypoint *array,
				const float fvalue,
				const unsigned char uvalue,
				const int SIZE
){
	int gid = get_global_id(0);
	if (gid<SIZE){
		t_keypoint kp;
		kp.x =  fvalue;
		kp.y =  fvalue;
		kp.scale =  fvalue;
		kp.angle =  fvalue;
		for (int i=0;i<128;i++){
			kp.desc[i] = uvalue;
		}
		array[gid] = kp;
	}
}
