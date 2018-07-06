/*
 *   Project: SIFT: An algorithm for image alignement
 *            gaussian.cl: Kernel for gaussian signal generation.
 *
 *
 *   Copyright (C) 2013-8 European Synchrotron Radiation Facility
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
 **/

/**
 * \brief gaussian: Initialize a vector with a gaussian function.
 * 
 * This is a 3 part kernel with first the computation of the gaussisan then a parallel 
 * reduction summation and a normalization.
 * This kernel must be run with size == workgroup
 *
 *
 * :param data:        Float pointer to global memory storing the vector.
 * :param sigma:    width of the gaussian
 * :param size:     size of the function
 *
**/

kernel void gaussian(global     float    *data,
                     const      float     sigma,
                     const        int     SIZE,
                     local      float    *shm1,
                     local      float    *shm2)
{
    int lid = (int) get_local_id(0);
    int group_size = (int) get_local_size(0);
    if(lid < SIZE)
    {
        float x = ((float)lid - ((float)SIZE - 1.0f)/2.0f) / sigma;
        float y = exp(-x * x / 2.0f);
        shm1[lid] = y / sigma / sqrt(2.0f * M_PI_F);
        shm2[lid] = shm1[lid];
    }
    else
    {
        shm2[lid] = 0.0f;
    }
    
//    Now we sum all in shared memory
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int bs=group_size/2; bs>=1; bs/=2)
    {
        if ((lid<group_size) && (lid < bs) && ((lid + bs)<group_size))
        {
            shm2[lid] = shm2[lid] + shm2[lid + bs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //    Now we normalize the gaussian curve

    if(lid < SIZE)
    {
        data[lid] = shm1[lid] / shm2[0];
    }
}
/**
 * \brief gaussian: Initialize a vector with a gaussian function.
 *
 * Same as previous except that there is no synchronization: use the sum of the integral 
 *
 * :param data:        Float pointer to global memory storing the vector.
 * :param sigma:    width of the gaussian
 * :param size:     size of the function
 *
 * Nota:  shm1 & shm2 are unused.
**/

kernel void
gaussian_nosync(global     float   *data,
                const      float   sigma,
                const      int     SIZE)
//                local      float*    shm1,
//                local      float*    shm2)
{
    int gid = (int) get_global_id(0);
    if(gid < SIZE)
    {
        float x = ((float)gid - ((float)SIZE - 1.0f)/2.0f) / sigma;
        float y = exp(-x * x / 2.0f);
        data[gid] = y / sigma / sqrt(2.0f * M_PI_F);
    }
}
