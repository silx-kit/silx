/*
 *   Project: SIFT: An algorithm for image alignement
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
 */

/**
 * \brief Linear combination of two matrices
 *
 * :param u: Pointer to global memory with the input data of the first matrix
 * :param a: float scalar which multiplies the first matrix
 * :param v: Pointer to global memory with the input data of the second matrix
 * :param b: float scalar which multiplies the second matrix
 * :param w: Pointer to global memory with the output data
 * :param width: integer, number of columns the matrices
 * :param height: integer, number of lines of the matrices
 *
 * Nota: updated to have coalesced access on dim[0]
 */

kernel void combine(
                    global float *u,
                    float a,
                    global float *v,
                    float b,
                    global float *w,
                    int dog,
                    int width,
                    int height)
{

    int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);

    if (gid0 < width && gid1 < height) 
    {
        int index = gid0 + width * gid1;
        int index_dog = dog * width * height +  index;
        w[index_dog] = a * u[index] + b * v[index];
    }
}



/**
 * \brief Deletes the (-1,-1,-1,-1) in order to get a more "compact" keypoints vector
 *  This is based on atomic add
 *
 * :param keypoints: Pointer to global memory with the keypoints
 * :param output: Pointer to global memory with the output
 * :param counter: Pointer to global memory with the shared counter in the output
 * :param start_keypoint: start compaction at this index. counter should be equal to start at the begining.
 * :param end_keypoint: index of last keypoints
 *
 */

kernel void compact(
    global actual_keypoint* keypoints,
    global actual_keypoint* output,
    global int* counter,
    int start_keypoint,
    int end_keypoint)
{

    int gid0 = (int) get_global_id(0);
    if (gid0 < start_keypoint)
    {
        output[gid0] = keypoints[gid0];
    }
    else if (gid0 < end_keypoint)
    {
        actual_keypoint k = keypoints[gid0];

        if (k.row >= 0.0f)
        { //Coordinates are never negative
            int old = atomic_inc(counter);
            if (old < end_keypoint)
            {
                output[old] = k;
            }
        }
    }
}
