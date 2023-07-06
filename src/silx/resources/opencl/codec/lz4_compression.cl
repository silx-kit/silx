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

// short compare and swap function used in sort_odd_even
inline short8 _order_short4(short4 a, short4 b){
    return (a.s0<b.s0)?(short8)(a,b):(short8)(b,a);
}

/* Odd–even parallel sort on the magnitude of values in shared memory
 * Not the fastest algorithm but it is parallel and inplace.
 * data are sorted accoding to their absolute values (the sign indictes if it is a litteral or a match)   
 */
inline void sort_odd_even(int start,
                          int stop,
                          volatile local short4 *lbuffer){
    int size = stop - start;
    if (size <2){
        return;
    }
    int cycle = (int)ceil(size/2.0);
    int tid = get_local_id(0); // thread id
    int wg = get_local_size(0);// workgroup size
    short8 swapped;
    int pid = start + tid + tid;
    for (int i=0; i<cycle; i++){ 
        //loop over the number of cycle to perform:
        // first case: even test above, odd test below
        
        if (pid+1<stop){
            swapped = _order_short4(lbuffer[pid], lbuffer[pid+1]);
            lbuffer[pid] = (short4)(swapped.s0, swapped.s1, swapped.s2, swapped.s3);
            lbuffer[pid+1] = (short4)(swapped.s4, swapped.s5, swapped.s6, swapped.s7);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (pid+2<stop){
            swapped = _order_short4(lbuffer[pid+1], lbuffer[pid+2]);
            lbuffer[pid+1] = (short4)(swapped.s0, swapped.s1, swapped.s2, swapped.s3);
            lbuffer[pid+2] = (short4)(swapped.s4, swapped.s5, swapped.s6, swapped.s7);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/* compact segments
 * After the scan, begining of litterals and of match are noted and stored in segments.
 * In this function one takes 2 segments, starting with a litteral and concatenate the subsequent match  
 * as a consequence, the number of segments is divided by 2 !
 */
inline int compact_segments(local volatile short4 *segments,
                            int nb){
    int tid = get_local_id(0); // thread id
    short4 merge;
    if (2*tid<nb){
        short4 lit = segments[2*tid];
        short4 mat = segments[2*tid+1];        
        if ((lit.s1 == 0) && (lit.s2 == 0) && (mat.s2 !=0)){
            merge = (short4)(lit.s0, mat.s0-lit.s0, mat.s2, 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (2*tid<nb){
        segments[tid] = merge;
    }
    return nb/2;
}

/* This function scans the input data searching for litterals and matches.
 */
inline int scan4match(  local uchar *buffer,       // buffer with input data in it, as large as possible, limited by shared memory space.
                        int start,
                        int stop,
                        local short *match_buffer, // size of the wg is enough
                        volatile local int* cnt             // size 2 is enough
                       ){
    
    int wg = get_local_size(0);// workgroup size
    int tid = get_local_id(0); // thread id
    int size = stop-start;
    cnt[0] = min(wg, size);
    cnt[1] = 0;
    
    // memset match_buffer
    match_buffer[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int pid = tid + start;
    uchar here = (pid < stop)?buffer[pid]:255;
    int match = 1;
    uchar valid = 1;
    for (int i=pid+1; i<stop; i++){
        if (valid){
            if (buffer[i] == here){
                match++;
            }
            else{
//                printf("thread %d starting at %d found match %d up to position %d max is %d still running %d\n", tid, pid, match, i, cnt[1], cnt[0]);
                valid = 0;
                atomic_dec(cnt);
                match_buffer[tid] = match;
                atomic_max(&cnt[1], i);
            }            
        }
        if (cnt[0] == 0){
            break;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return cnt[1];
}

// segment over one wg size. returns the number of segments found
inline int segmentation(int start, //index where scan4match did start
                        int stop,
                        local short *match_buffer,      // size of the workgroup
                        local volatile short4 *segments,// size of the workgroup
                        local volatile int* cnt                  // size 1 is enough 
                        ){
    int wg = get_local_size(0);// workgroup size
    int tid = get_local_id(0); // thread id
    int pid = tid + start;
    cnt[0] = 1;
    segments[0] = (short4)(start, 0, 0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
         
    if ((tid>0) && (pid<stop)){
        short here = match_buffer[tid],
              there= match_buffer[tid-1];
        if ((there==1) && (here>4)){
            segments[atomic_inc(cnt)] = (short4)(pid, 0, here, 0);
        } else
        if ((here==1) && (there>1)){
            segments[atomic_inc(cnt)] = (short4)(pid, 0, 0, 0);
        }
    }
//    if (cnt[0] == 1){
//        // noting occured, just complete segment
//        segments[0] = (short4)(start, stop-start, 0, 0);
//    }else{
//        // sort segments
//        if (tid == 0){
//            cnt[0] += 1;
//            
//        }
//        sort_odd_even(0, cnt[0], segments);
//        // compact segments  TODO       
//    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return cnt[0];
}


//  Build token, concatenation of a litteral and a match 
inline uchar build_token(short4 segment){
    int lit = segment.s1;
    int mat = segment.s2;
    int token = ((lit & 15)<<4)|((mat-4)&15);
    return token;
}

// copy collaborative, return the position in output stream.
inline int copy(global uchar* dest,
                 const int dest_position,
                 local uchar* source,
                 const int src_position,
                 const int length){
    for (int i=get_local_id(0); i<length; i+=get_local_size(0)) {
        dest[dest_position+i] = source[src_position+i];
    }
    return dest_position+length;
}

/*
 * Perform the actual compression by copying
 * 
 * return the end-position in the output stream 
 */
inline int write_lz4(local uchar *buffer,
                     local volatile short4 *segments, // size of the workgroup
                     int nb_segments,
                     int start_cmp,
                     global uchar *output_buffer,
                     int stop
                    )
{
    for (int i=0; i<nb_segments; i++){
        short4 segment = segments[i];
        
        //write token
        output_buffer[start_cmp] = build_token(segment);
        start_cmp++;
        
        //write litteral overflow
        if (segment.s1>=15){
            int rem = segment.s1-15;
            while (rem>=255){
                output_buffer[start_cmp] = 255;
                start_cmp++;
                rem -=255;
            }
            output_buffer[start_cmp] = rem;
            start_cmp++;
        }
        
        //copy litteral. This is collaborative.
        start_cmp = copy(output_buffer, start_cmp,
                         buffer, segment.s0, segment.s1);
        
        //write offset, here always 1 in 16 bits little endian !
        output_buffer[start_cmp] = 1;
        output_buffer[start_cmp+1] = 0;
        start_cmp+=2;
        
        //write match overflow
        if (segment.s2>=19){
            int rem = segment.s2-19;
            while (rem>=255){
                output_buffer[start_cmp] = 255;
                start_cmp++;
                rem -=255;
            }
            output_buffer[start_cmp] = rem;
            start_cmp++;            
        }
    }//loop over segments
    return start_cmp;
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
                       local short *match_buffer, // size of the buffer
                       local volatile short4 *segments      // contains: start of segment (uncompressed), number of litterals, number of match (offset is enforced to 1) and start of segment (compressed) 
                      ){
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    
    //copy input data to buffer
    int actual_buffer_size = min(buffer_size, input_size - ((gid+1) * buffer_size));
    int start_block = gid * buffer_size;
    for (int i=tid; i<actual_buffer_size; i+=wg){
        buffer[i] = input_buffer[start_block+i];
    }
    local int cnt[2]; // small counters
    
    /// divide the work in parts, one wg has enough threads
    int start = 0;
//    while (start<actual_buffer_size){
        //scan for matching
//        int next_start = scan4match(buffer, start, actual_buffer_size, match_buffer);
        // extract from matching the sequence
        
        
//        start = next_start;
//    }
    
    
    
    
    
}

// test kernel to ensure `sort_odd_even` works
kernel void test_sort(global short4 *buffer,
                      int start,
                      int stop,
                      volatile local short4 *lbuffer){
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
// test kernel to ensure `scan4match` works
kernel void test_scan4match(
        global uchar *buffer,       // buffer with input data in it, as large as possible, limited by shared memory space.
        global short *match,        // buffer with output data in it, matches the buffer array
        int start,
        int stop,
        global int *end,
        local uchar *lbuffer,
        local short *lmatch){
    local volatile int cnt[2];    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    for (int i=tid; i<stop; i+=wg){
            lbuffer[i] = buffer[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int res = scan4match(lbuffer,
                  start, stop,
                  lmatch,
                  cnt);
    if ((tid==0) && (gid==0))printf("scanned up to %d\n", res);
    //copy back
    if (tid<stop-start){
        match[tid] = lmatch[tid];
    }
    end[0] = res;
    
}

// kernel to test scan4match+segmentation works
kernel void test_segmentation(global uchar *buffer,
                              int start, //index where scan should start
                              int stop,
                              global int *nbsegment,
                              global short4 *segments // size of the workgroup
){
    local volatile int cnt[2];    
    local volatile short4 lsegments[64];
    local uchar lbuffer[1024];
    local short lmatch[64];
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    for (int i=tid; i<stop; i+=wg){
            lbuffer[i] = buffer[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int res = scan4match(lbuffer,
                  start, stop,
                  lmatch,
                  cnt);
    if ((tid==0) && (gid==0))printf("scanned up to %d\n", res);
    int res2 = segmentation(start, stop, lmatch, lsegments, cnt);
    nbsegment[0] = res2;
    if (tid<res2){
        segments[tid] = lsegments[tid];
    }
}
