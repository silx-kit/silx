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

kernel void mark_exceptions(global char* raw, int size, global int* mask,  global int* cnt, global int* exc)
{
    int ws, gid, lid;
    ws = get_local_size(0);
    lid = get_local_id(0);
    gid = get_global_id(0);
    if (gid<size)
    {
        int value, position;
        value = raw[gid];
        if (value == -128)
        {
            position = atomic_inc(cnt);
            exc[position] = gid;
            atomic_inc(&mask[gid]);
            atomic_inc(&mask[gid+1]);
            atomic_inc(&mask[gid+2]);
            if (((int) raw[gid+1] == 0) && ((int) raw[gid+2] == -128))
            {
                atomic_inc(&mask[gid+3]);
                atomic_inc(&mask[gid+4]);
                atomic_inc(&mask[gid+5]);
                atomic_inc(&mask[gid+6]);
            }

        }

    }
}
//run with WG=1, as may as exceptions
kernel void treat_exceptions(global char* raw,
                             int size,
                             global int* mask,
                             global int* cnt,
                             global int* exc,
                             global int* values,
                             global int* delta)
{
    int gid = get_global_id(0);
    int position = exc[gid];
    if ((position>0) && ((int)mask[position-1] !=0))
    { // this is actually not another exception, remove from list
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
            {
                uchar low_byte = raw[inp_pos+1];
                char high_byte = raw[inp_pos+2] ;

                next_value = high_byte<<8 | low_byte;
                if (next_value == -32768)
                {
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
            delta[inp_pos] = inc;
            inp_pos += inc;
            out_pos += 1 ;
            is_masked = mask[inp_pos];
        }
    }
}

kernel void calc_size(int size,
                      global int* mask,
                      global int* delta)
{
    int gid = get_global_id(0);
    if (gid < size)
    {
        int is_masked;
        is_masked = mask[gid];
        if (!is_masked)
        {
            delta[gid] = 1;
        }
    }
}
kernel void copy_values(global char* raw,
                        int size,
                        int datasize,
                        global int* values,
                        global int* exception_values,
                        global int* positions,
                        global int* delta)
{
    int gid = get_global_id(0);
    int in_pos, value_size;
    if (gid < datasize)
    {
        if (gid == 0)
        {
            in_pos = 0;
        }
        else
        {
            in_pos = positions[gid-1];
        }

        if (in_pos < size)
        {
            value_size = delta[in_pos];
            if (value_size == 1)
            {
                values[gid] = (int) raw[in_pos];
            }
            else if (value_size >1)
            {
                values[gid] = exception_values[in_pos];
            }
        }
    }
}
