//CL//

/*
 *   Project: SILX: Alogorithms for image processing
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

/* Single kernel histogram for float array.
 *
 * Can perform histograming in log-scale (using the arcsinh)

Parameters:
  - data: buffer with the image content in float (input)
  - data_size: input
  - mini: Lower bound of the first bin
  - maxi: upper bouns of the last bin
  - map_operation: Type of pre-processor to use. if if set to !=0, use log scale
  - hist: resulting histogram (ouptut)
  - hist_size: number of bins
  - tmp_hist: temporary storage of size  hist_size*num_groups
  - processed: temporary storage of size 1 initialially set to 0
  - local_hist: local storage of size hist_size


Grid information:
    * use the largest WG size. If it is larger than the number of bins (hist_size),
    take the power of 2 immediately above
    *Schedule as many WG as the device has compute engines. No need to schuedule more,
    it is just a waist of memory
    * The histogram should fit in shared (local) memory and tmp_hist can be large!

Assumes:
    hist and local_hist have size hist_size
    edges has size hist_size+2
    tmp_hist has size hist_size*num_groups
    processed is of size one and initially set to 0

*/


static float preprocess(float value, int code)
{
    //This function can be modified to have more scales
    if (code!=0)
    {
        return asinh(value);
    }
    else
    {
        return value;
    }
}

kernel void histogram(global float *data,
                      int data_size,
                      float mini,
                      float maxi,
                      int map_operation,
                      global int *hist,
                      global float *edges,
                      int hist_size,
                      global int *tmp_hist,
                      global int *processed,
                      local int *local_hist)
{
    // each thread
    int lid = get_local_id(0);
    int ws = get_local_size(0);
    int nb_engine = get_num_groups(0);
    int engine_id = get_group_id(0);

    // memset the local_hist array
    for (int i=0; i<hist_size; i+=ws)
    {
        int j = i + lid;
        if (j<hist_size)
        {
            local_hist[j] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Process the local data and accumulate in shared memory
    int bloc_size = (int) ceil((float)data_size/(float)nb_engine);
    int start = bloc_size * engine_id;
    int stop = min(start + bloc_size, data_size);
    float vmini = preprocess(mini, map_operation);
    float vscale = (float)hist_size/(preprocess(maxi, map_operation) -vmini);
    for (int i = start; i<stop; i+=ws)
    {
        int address = i + lid;
        if (address < stop)
        {
            float value = data[address];
            if ((!isnan(value)) && (value>=mini) && (value<=maxi))
            {
                float vvalue = (preprocess(value, map_operation)-vmini)*vscale;
                int idx = clamp((int) vvalue, 0, hist_size - 1);
                atomic_inc(&local_hist[idx]);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now copy result into the right place and reset the first value of the shared array
    for (int i=0; i<hist_size; i+=ws)
    {
        int j = i + lid;
        if (j<hist_size)
        {
            tmp_hist[hist_size * engine_id + j] = local_hist[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    local_hist[0] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    //Increment the system wide shared variable processed and share the result with the workgroup
    if (lid == 0)
    {
        local_hist[0] = atomic_inc(processed);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // If we are the engine last to work, perform the concatenation of all results

    if ((local_hist[0] + 1) == nb_engine)
    {
        for (int i=0; i<hist_size; i+=ws)
        {
            int j = i + lid;
            int lsum = 0;
            if (j<hist_size)
            {
                for (int k=0; k<nb_engine; k++)
                {
                    lsum += tmp_hist[hist_size * k + j];
                }
                hist[j] = lsum;
                edges[j] = vmini + j/vscale;
            }
        }
        // Finally reset the counter
        if (lid == 0)
        {
            processed[0] = 0;
            edges[hist_size] = vmini + hist_size/vscale;;
        }

    }
}
