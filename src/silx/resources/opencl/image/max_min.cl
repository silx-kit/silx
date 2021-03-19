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



#define REDUCE(a, b) ((float2) (fmax(a.x, b.x), fmin(a.y, b.y)))

static float2 read_and_map(int idx,
                           global float* data)
{
    idx *= NB_COLOR;
    float tmp = data[idx];
    float2 res = (float2) (tmp, tmp);
    if (NB_COLOR > 1)
    {
        for (int c=1; c<NB_COLOR; c++)
        {
            tmp = data[idx + c];
            res = (float2) (fmax(res.x, tmp), fmin(res.y, tmp));
        }
    }
    return res;
}


/**
 * \brief max_min_global_stage1: Look for the maximum an the minimum of an array. stage1
 *
 * optimal workgroup size: 2^n greater than sqrt(size), limited to 512
 * optimal total item size:  (workgroup size)^2
 * if size >total item size: adjust seq_count.
 *
 * :param data:       Float pointer to global memory storing the vector of data.
 * :param out:        Float2 pointer to global memory storing the temporary results (workgroup size)
 * :param seq_count:  how many blocksize each thread should read
 * :param size:       size of the problem (number of element in the image
 * :param l_data      Shared memory: 2 float per thread in workgroup
 *
**/


kernel void max_min_reduction_stage1( global const float *data,
                                      global float2 *out,
                                      int size,
                                      local  float2 *l_data)// local storage 2 float per thread
{
    int group_size =  get_local_size(0);
    int lid = get_local_id(0);
    float2 acc;
    int big_block = group_size * get_num_groups(0);
    int i =  lid + group_size * get_group_id(0);

    if (lid<size)
        acc = read_and_map(lid, data);
    else
        acc = read_and_map(0, data);

    // Linear pre-reduction stage 0

    while (i<size){
      acc = REDUCE(acc, read_and_map(i, data));
      i += big_block;
    }

    // parallel reduction stage 1

    l_data[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int block=group_size/2; block>1; block/=2)
        {
            if ((lid < block) && ((lid + block)<group_size)){
                l_data[lid] = REDUCE(l_data[lid], l_data[lid + block]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    if (lid == 0)
    {
        if (group_size > 1)
        {
            acc = REDUCE(l_data[0], l_data[1]);
        }
        else
        {
            acc = l_data[0];
        }
        out[get_group_id(0)] = acc;
    }
}


/**
 * \brief global_max_min: Look for the maximum an the minimum of an array.
 *
 *
 *
 * :param data2:      Float2 pointer to global memory storing the vector of pre-reduced data (workgroup size).
 * :param maximum:    Float pointer to global memory storing the maximum value
 * :param minumum:    Float pointer to global memory storing the minimum value
 * :param l_data      Shared memory: 2 float per thread in workgroup
 *
**/

kernel void max_min_reduction_stage2(
        global const float2 *data2,
        global float2 *maxmin,
        local  float2 *l_data)// local storage 2 float per thread
{
    int lid = get_local_id(0);
    int group_size =  get_local_size(0);
    float2 acc = (float2)(-1.0f, -1.0f);
    if (lid<=group_size)
    {
        l_data[lid] = data2[lid];
    }
    else
    {
        l_data[lid] = acc;
    }

    // parallel reduction stage 2


    barrier(CLK_LOCAL_MEM_FENCE);
    for (int block=group_size/2; block>1; block/=2)
    {
        if ((lid < block) && ((lid + block)<group_size))
        {
            l_data[lid] = REDUCE(l_data[lid], l_data[lid + block]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (lid == 0  )
    {
        if ( group_size > 1)
        {
            acc = REDUCE(l_data[0], l_data[1]);
        }
        else
        {
            acc = l_data[0];
        }
        maxmin[0] = acc;
    }
}

/*This is the serial version of the min_max kernel.
 *
 * It has to be launched with WG=1 and only 1 WG has to be launched !
 *
 * :param data:       Float pointer to global memory storing the vector of data.
 * :param size:       size of the
 * :param maximum:    Float pointer to global memory storing the maximum value
 * :param minumum:    Float pointer to global memory storing the minimum value
 *
 *
 */
kernel void max_min_serial(
                            global const float *data,
                            unsigned int size,
                            global float2 *maxmin
                            )
{
    float2 acc = read_and_map(0, data);
    for (int i=1; i<size; i++)
    {
        acc = REDUCE(acc, read_and_map(i, data));
    }

    maxmin[0] = acc;
}
