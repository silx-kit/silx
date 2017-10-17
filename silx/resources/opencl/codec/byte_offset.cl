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
 * - Mark regions with exceptions. This is a map kernel, the workgroup size does
 *   not matter. This generates the mask, counts the number of region and provides
 *   a start position for each of them.
 * - treat exceptions. For this, one thread in a workgoup treats a complete masked
 *   region in a serial fashion. All regions are treated in parallel.
 * - Normal pixels are marked having a size of 1 (TODO: do this with previous kernel + fill!)
 * - Valid pixels are copied (compact for zeros in delta)
 * - Input position is calculated from a cum_sum
 * - proper pixel values are copied
 * - Finally a cum-sum is performed to retrieve the decompressed data
 */

kernel void mark_exceptions(global char* raw,
                            int size,
                            global int* mask,
                            global int* cnt,
                            global int* exc)
{
    int gid, maxi;
    gid = get_global_id(0);
    maxi = size - 1;
    if (gid<size)
    {
        int value, position;
        value = raw[gid];
        if (value == -128)
        {
            position = atomic_inc(cnt);
            exc[position] = gid;
            atomic_inc(&mask[gid]);
            atomic_inc(&mask[min(gid+1, maxi)]);
            atomic_inc(&mask[min(gid+2, maxi)]);
            if (((int) raw[min(gid+1, maxi)] == 0) &&
                ((int) raw[min(gid+2, maxi)] == -128))
            {
                atomic_inc(&mask[min(gid+3, maxi)]);
                atomic_inc(&mask[min(gid+4, maxi)]);
                atomic_inc(&mask[min(gid+5, maxi)]);
                atomic_inc(&mask[min(gid+6, maxi)]);
            }
        }
    }
}

//run with WG=1, as may as exceptions
kernel void treat_exceptions(global char* raw, //raw compressed stream
                             int size,         //size of the raw compressed stream
                             global int* mask, //tells if the value is masked
                             global int* exc,  //array storing the position of the start of exception zones
                             global int* values)// stores decompressed values.
{
    int gid = get_global_id(0);
    int position = exc[gid];
    if ((position>0) && ((int)mask[position-1] !=0))
    { // This is actually not another exception, remove from list.
      // Those data will be treated by another workgroup.
        exc[gid] = -1;
    }
    else
    {
        int inp_pos, out_pos, value, is_masked, next_value, inc;
        inp_pos = position;
        out_pos = position;
        is_masked = mask[inp_pos];
        while ((is_masked) && (inp_pos<size) && (out_pos<size))
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
            mask[inp_pos] = -1; // mark the processed data in the mask
            inp_pos += inc;
            out_pos += 1 ;
            is_masked = mask[inp_pos];
        }
    }
}

// copy the values of the non-exceptions elements based on the mask.
// Set the value of the mask to the current position  or -2
kernel void treat_simple(global char* raw,
                         int size,
                         global int* mask,
                         global int* values)
{
    int gid = get_global_id(0);
    if (gid < size)
    {
        if (mask[gid] > 0)
        { // convert the masked position from a positive value to a negative one
            mask[gid] = -2;
            values[gid] = 0; //this prevents a full memset of the array
        }
        else if (mask[gid] == -1)
        { // this is an exception and has already been treated.
            mask[gid] = gid;
        }
        else
        { // normal case
            values[gid] = raw[gid];
            mask[gid] = gid;
        }
    }
}

// copy the values of the elements to definitive position
kernel void copy_result_int(global int* values,
                            global int* indexes,
                            int size,
                            global int* output)
{
    int gid = get_global_id(0);
    if (gid<size)
    {
        output[gid] = values[indexes[gid]];
    }
}

// copy the values of the elements to definitive position
kernel void copy_result_float(global int* values,
                              global int* indexes,
                              int size,
                              global float* output)
{
    int gid = get_global_id(0);
    if (gid<size)
    {
        output[gid] = (float) values[indexes[gid]];
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

