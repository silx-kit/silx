/*
 *   Project: SILX: Bitshuffle LZ4 compressor
 *
 *   Copyright (C) 2023 European Synchrotron Radiation Facility
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

// define the min and the max of the absolute value
#define MAXA(a,b) (abs(a)>abs(b))?a:b;
#define MINA(a,b) (abs(a)>abs(b))?b:a;

/* Function is called at the end by the last running wg to compact the output
 * 
 * Maybe would be more efficient to call another kernel to do this in parallel ?
 */
inline void compact_output(global uchar *output_buffer,
                           int output_size,
                           global uchar *output_ptr, // Length of all output from different wg
                           int buffer_size // 1.2x the input buffer size!
                           ){
    int tid = get_local_id(0); // thread id
    int wg = get_local_size(0);// workgroup size
    int start_read = 0;
    int start_write = output_ptr[0];
    int to_copy;
    for (int i=1; i<get_num_groups(0); i++){
        start_read += buffer_size;
        to_copy = output_ptr[i];
        for (int t=tid; t<to_copy; t+=wg){
            output_buffer[start_write+t] = output_buffer[start_read+t];
        }
        start_write += to_copy;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


/* Odd–even parallel sort on the magnitude of values in shared memory
 * Not the fastest algorithm but it is parallel and inplace.
 * data are sorted accoding to their absolute values (the sign indictes if it is a litteral or a match)   
 */
inline void sort_odd_even(int start,
                          int stop,
                          volatile local short *lbuffer){
    int size = stop - start;
    if (size <2){
        return;
    }
    int cycle = (int)(size/2.0+0.5);
    int tid = get_local_id(0); // thread id
    int wg = get_local_size(0);// workgroup size
    short here, there;
    int pid = start + tid;
    for (int i=0; i<cycle; i++){ 
        //loop over the number of cycle to perform:
        // first case: even test above, odd test below
        if (tid<size){
            here = lbuffer[pid];
            if (tid%2){
                // odd tid test above
                if (tid+1 == size)
                   there = here;
                else
                   there = lbuffer[pid+1];
            }else{
                // even tid test below
                if (tid==0)
                    there = here;
                else
                    there = lbuffer[pid-1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid<size){
            if (tid%2){
                lbuffer[pid] = MINA(here, there);
            }              
            else{
                lbuffer[pid] = MAXA(here, there);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // second case: odd test above, even test below
        if (tid<size){
            here = lbuffer[pid];
            if (tid%2){
                // odd tid test below
                if (tid == 0)
                   there = here;
                else
                   there = lbuffer[pid-1];
            }
            else{
                // even tid test above
                if (tid+1==size)
                    there = here;
                else
                    there = lbuffer[pid+1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid<size){
            if (tid%2){
                lbuffer[pid] = MAXA(here, there);
            }
            else{
                lbuffer[pid] = MINA(here, there);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// test kernel to ensure `sort_odd_even` works
kernel void test_sort(global short *buffer,
                      int start,
                      int stop,
                      volatile local short *lbuffer){
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    for (int i=tid; i<stop; i+=wg){
        lbuffer[i] = buffer[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sort_odd_even(start, stop, lbuffer);
    for (int i=tid; i<stop; i+=wg){
        buffer[i] = lbuffer[i];
    }
}                       

/* Main kernel for lz4 compression
 */
kernel void lz4_cmp(   global uchar *input_buffer,
                       int input_size,
                       global uchar *output_buffer,
                       int output_size,
                       global uchar *output_ptr, // Length of all output from different wg
                       global int *running_grp,  // counter with the number of wg still running
                       local uchar *buffer,
                       int buffer_size,
                       local short *match_buffer // size of the workgroup
                      )
{
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    //copy input data to buffer
    int actual_buffer_size = min(buffer_size, input_size - ((gid+1) * buffer_size));
    int start_block = gid * buffer_size;
    for (int i=tid; i<actual_buffer_size; i+=wg){
        buffer[i] = input_buffer[start_block+i];
    }
    
    local int running[1];
    running[0] = wg;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    uchar here = buffer[tid];
    int match = 1;
    uchar valid = 1;
    for (size_t i=tid+1; i<buffer_size; i++){
        if (valid && (buffer[i] == here)){
            match++;
        }
        else{
            valid = 0;
            atomic_dec(running);
            if (match>4){
                match_buffer[tid] = match;
            }
            else{
                match_buffer[tid] = 0;
            }
            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (running[0] == 0){
            break;
        }
    }
    // retrieve the match_buffer
    int out_block = (int) ceil(buffer_size * 1.2);
    int write_block = gid * out_block;
    for (size_t i=tid; i<actual_buffer_size; i+=wg){
        output_buffer[write_block+i] = buffer[i];
        }
    
    return;
}