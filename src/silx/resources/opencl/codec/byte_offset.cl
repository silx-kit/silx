/*
 *   Project: SILX: A data analysis tool-kit
 *
 *   Copyright (C) 2017 European Synchrotron Radiation Facility
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

/* To decompress CBF byte-offset compressed in parallel on GPU one needs to:
 * - Set all values in mask and exception counter to zero.
 * - Mark regions with exceptions and set values without exception.
 *   This generates the values (zeros for exceptions), the exception mask,
 *   counts the number of exception region and provides a start position for
 *   each exception.
 * - Treat exceptions. For this, one thread in a workgoup treats a complete
 *   masked region in a serial fashion. All regions are treated in parallel.
 *   Values written at this stage are marked in the mask with -1.
 * - Double scan: inclusive cum sum for values, exclusive cum sum to generate
 *   indices in output array. Values with mask = 1 are considered as 0.
 * - Compact and copy output by removing duplicated values in exceptions.
 */

kernel void mark_exceptions(global char* raw,
                            int size,
                            int full_size,
                            global int* mask,
                            global int* values,
                            global int* cnt,
                            global int* exc)
{
    int gid;
    gid = get_global_id(0);
    if (gid<size)
    {
        int value, position;
        value = raw[gid];
        if (value == -128)
        {
            int maxi;
            values[gid] = 0;
            position = atomic_inc(cnt);
            exc[position] = gid;
            maxi = size - 1;
            mask[gid] = 1;
            mask[min(maxi, gid+1)] = 1;
            mask[min(maxi, gid+2)] = 1;

            if (((int) raw[min(gid+1, maxi)] == 0) &&
                ((int) raw[min(gid+2, maxi)] == -128))
            {
                mask[min(maxi, gid+3)] = 1;
                mask[min(maxi, gid+4)] = 1;
                mask[min(maxi, gid+5)] = 1;
                mask[min(maxi, gid+6)] = 1;
            }
        }
        else
        { // treat simple data

            values[gid] = value;
        }
    }
    else if (gid<full_size)
    {
        mask[gid]=1;
        values[gid] = 0;
    }
}

//run with WG=1, as may as exceptions
kernel void treat_exceptions(global char* raw,  //raw compressed stream
                             int size,          //size of the raw compressed stream
                             global int* mask,  //tells if the value is masked
                             global int* exc,   //array storing the position of the start of exception zones
                             global int* values)// stores decompressed values.
{
    int gid = get_global_id(0);
    int inp_pos = exc[gid];
    if ((inp_pos<=0) || ((int)mask[inp_pos - 1] == 0))
    {
        int value, is_masked, next_value, inc;
        is_masked = (mask[inp_pos] != 0);
        while ((is_masked) && (inp_pos<size))
        {
            value = (int) raw[inp_pos];
            if (value == -128)
            { // this correspond to 16 bits exception
                uchar low_byte = raw[inp_pos+1];
                char high_byte = raw[inp_pos+2] ;
                next_value = high_byte<<8 | low_byte;
                if (next_value == -32768)
                { // this correspond to 32 bits exception
                    uchar low_byte1 = raw[inp_pos+3],
                          low_byte2 = raw[inp_pos+4],
                          low_byte3 = raw[inp_pos+5];
                    char high_byte4 = raw[inp_pos+6] ;
                    value = high_byte4<<24 | low_byte3<<16 | low_byte2<<8 | low_byte1;
                    inc = 7;
                }
                else
                {
                    value = next_value;
                    inc = 3;
                }
            }
            else
            {
                inc = 1;
            }
            values[inp_pos] = value;
            mask[inp_pos] = -1; // mark the processed data as valid in the mask
            inp_pos += inc;
            is_masked = (mask[inp_pos] != 0);
        }
    }
}

// copy the values of the elements to definitive position
kernel void copy_result_int(global int* values,
                            global int* indexes,
                            int in_size,
                            int out_size,
                            global int* output
                            )
{
    int gid = get_global_id(0);
    if (gid < in_size)
    {
        int current = max(indexes[gid], 0),
               next = (gid >= (in_size - 1)) ? in_size + 1 : indexes[gid + 1];
        //we keep always the last element
        if ((current <= out_size) && (current < next))
        {
            output[current] = values[gid];
        }
    }
}

// copy the values of the elements to definitive position
kernel void copy_result_float(global int* values,
                              global int* indexes,
                              int in_size,
                              int out_size,
                              global float* output
                              )
{
    int gid = get_global_id(0);
    if (gid<in_size)
    {
        int current = max(indexes[gid], 0),
               next = (gid >= (in_size - 1)) ? in_size + 1 : indexes[gid + 1];
        if ((current < out_size) && (current < next))
        {
            output[current] = (float) values[gid];
        }
    }
}


// combined memset for all arrays used for Byte Offset decompression
kernel void byte_offset_memset(global char* raw,
                               global int* mask,
                               global int* index,
                               global int* result,
                               int full_size,
                               int actual_size
                              )
{
    int gid = get_global_id(0);
    if (gid < full_size)
    {
        raw[gid] = 0;
        index[gid] = 0;
        result[gid] = 0;
        if (gid<actual_size)
        {
            mask[gid] = 0;
        }
        else
        {
            mask[gid] = 1;
        }

    }
}


//Simple memset kernel for char arrays
kernel void fill_char_mem(global char* ary,
                          int size,
                          char pattern,
                          int start_at)
{
    int gid = get_global_id(0);
    if ((gid >= start_at) && (gid < size))
    {
        ary[gid] = pattern;
    }
}

//Simple memset kernel for int arrays
kernel void fill_int_mem(global int* ary,
                         int size,
                         int pattern,
                         int start_at)
{
    int gid = get_global_id(0);
    if ((gid >= start_at) && (gid < size))
    {
        ary[gid] = pattern;
    }
}

