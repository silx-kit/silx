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


#define REDUCE(a, b) ((float2)(fmax(a.x,b.x),fmin(a.y,b.y)))
#define READ_AND_MAP(i) ((float2)(data[i],data[i]))

/**
 * \brief max_min_global_stage1: Look for the maximum an the minimum of an array. stage1
 *
 * optimal workgroup size: 2^n greater than sqrt(SIZE), limited to 512
 * optimal total item size:  (workgroup size)^2
 * if SIZE >total item size: adjust seq_count.
 *
 * :param data:       Float pointer to global memory storing the vector of data.
 * :param out:          Float2 pointer to global memory storing the temporary results (workgroup size)
 * :param seq_count:  how many blocksize each thread should read
 * :param SIZE:          size of the initial array
 * :param ldata:        shared memory of size SIZE*2*4
 *
**/


kernel void max_min_global_stage1(
        global const   float  *data,
        global         float2 *out,
        unsigned       int     SIZE,
        local volatile float2 *ldata){

    unsigned int group_size =  get_local_size(0);
    unsigned int lid = get_local_id(0);
    float2 acc;
    unsigned int big_block = group_size * get_num_groups(0);
    unsigned int i =  lid + group_size * get_group_id(0);

    if (lid<SIZE)
        acc = READ_AND_MAP(lid);
    else
        acc = READ_AND_MAP(0);
    while (i<SIZE)
    {
      acc = REDUCE(acc, READ_AND_MAP(i));
      i += big_block;
    }

    ldata[lid] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int bs=group_size/2; bs>=1; bs/=2)
    {
        if ((lid<group_size) && (lid < bs) && ((lid + bs)<group_size))
        {
            ldata[lid] = REDUCE(ldata[lid], ldata[lid + bs]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[get_group_id(0)] = ldata[0];
}


/**
 * \brief global_max_min: Look for the maximum an the minimum of an array.
 *
 *
 *
 * :param data2:      Float2 pointer to global memory storing the vector of pre-reduced data (workgroup size).
 * :param maximum:    Float pointer to global memory storing the maximum value
 * :param minumum:    Float pointer to global memory storing the minimum value
 *
**/

kernel void max_min_global_stage2(
        global const   float2 *data2,
        global         float  *maximum,
        global         float  *minimum,
        local volatile float2 *ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int group_size =  get_local_size(0);
    float2 acc = (float2)(-1.0f, -1.0f);
    if (lid<=group_size)
    {
        ldata[lid] = data2[lid];
    }
    else
    {
        ldata[lid] = acc;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int bs=group_size; bs>=1; bs/=2)
    {
        if ((group_size>bs))
        {
            if ((lid<group_size) && (lid < bs) && ((lid + bs)<group_size))
            {
                ldata[lid] = REDUCE(ldata[lid], ldata[lid + bs]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0)
    {
        acc = ldata[0];
        maximum[0] = acc.x;
        minimum[0] = acc.y;
    }
}

/*This is the serial version of the min_max kernel.
 *
 * It has to be launched with WG=1 and only 1 WG has to be launched !
 *
 * :param data:       Float pointer to global memory storing the vector of data.
 * :param SIZE:          size of the
 * :param maximum:    Float pointer to global memory storing the maximum value
 * :param minumum:    Float pointer to global memory storing the minimum value
 *
 *
 */
kernel void max_min_serial(
        global const float *data,
        unsigned int SIZE,
        global float *maximum,
        global float *minimum)
{
float value, maxi, mini;
value = data[0];
mini = value;
maxi = value;
for (int i=1; i<SIZE; i++)
{
    value = data[i];
    if (value>maxi)
        maxi = value;
    if (value<mini)
        mini = value;
}

maximum[0] = maxi;
minimum[0] = mini;

}
