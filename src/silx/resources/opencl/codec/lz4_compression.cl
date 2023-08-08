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

// This is used in tests to simplify the signature of those test kernels. 
#ifndef TEST_WG
#define TEST_WG 64
#endif
#ifndef TEST_BUFFER
#define TEST_BUFFER 1024
#endif
#ifndef MIN_MATCH
#define MIN_MATCH 4
#endif

// TODO generalize test methods to use this 


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
    int cycle = (size+1)/2;
    int tid = get_local_id(0); // thread id
    int wg = get_local_size(0);// workgroup size
    short8 swapped;
    int pid = start + tid + tid;
    for (int i=0; i<cycle; i++){ 
        //loop over the number of cycle to perform:
        // first case: even test above, odd test below
        
        if (pid+1<stop){
            swapped = _order_short4(lbuffer[pid], lbuffer[pid+1]);
            lbuffer[pid] = swapped.lo;
            lbuffer[pid+1] = swapped.hi;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (pid+2<stop){
            swapped = _order_short4(lbuffer[pid+1], lbuffer[pid+2]);
            lbuffer[pid+1] = swapped.lo;
            lbuffer[pid+2] = swapped.hi;
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
                            local volatile int* cnt){
    int tid = get_local_id(0); // thread id
    int nb = cnt[0];
    short4 merge, current, next;
    int w = 0; //write positions
    int r = 0; //read position
    //single threaded for safety ...
    if (tid == 0){
        short4 current, next, merged;
        current = segments[r++];
        while (r<nb){
            if (current.s2){//match exist, just copy
                segments[w++] = current;
                current = segments[r++];
            }
            else{//no match, just merge 2 neighbors
                next = segments[r++];
                current = (short4)(current.s0, next.s0-current.s0, next.s2, next.s3);
            }
        }
//        finally write current
        segments[w++] = current;
        cnt[0] = w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // memset the remaining
    w = cnt[0];
    if ((tid>=w) && (tid<nb)){
        segments[tid] = (short4)(0,0,0,0);
    }
// attempt for mutli-threaded...
//    if (2*tid<nb){
//        short4 lit = segments[2*tid];
//        short4 mat = segments[2*tid+1];        
//        if ((lit.s1 == 0) && (lit.s2 == 0)){
//            merge = (short4)(lit.s0, mat.s0-lit.s0, mat.s2, 0);
//        }            
//    }
    
    return cnt[0];
}

/* This function scans the input data searching for litterals and matches. return the end-of-scan position.
 */
inline int scan4match(  local uchar *buffer,       // buffer with input data in it, as large as possible, limited by shared memory space.
                        int start,
                        int stop,
                        local short *match_buffer, // size of the wg is enough
                        volatile local int* cnt    // size 1 is enough, idx0: largest index value found
                       ){
    
    int wg = get_local_size(0);// workgroup size
    int tid = get_local_id(0); // thread id
    int size = stop-start;
    cnt[0] = 0;
    
    // memset match_buffer
    match_buffer[tid] = -1;
    barrier(CLK_LOCAL_MEM_FENCE);
    int i; // position index
    int pid = tid + start;
    uchar here = (pid < stop)?buffer[pid]:255;
    int match = 0;
    uchar valid = 1;
    for (i=pid+1; i<stop; i++){
        if (valid){
            if (buffer[i] == here){
                match++;
            }
            else{
                atomic_max(cnt, i);
                match_buffer[tid] = match;
                valid = 0;
            }
        }
    }
    if ((valid) && (i==stop)){ // we reached the end of the block: stop anyway
        cnt[0] = stop;
        match_buffer[tid] = match;
        valid = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return cnt[0];
}

// segment over one wg size. returns the number of segments found
inline int segmentation(int start, //index where scan4match did start
                        int stop,  //size of the buffer
                        int end,   //index where scan4match did stop
                        local short *match_buffer,      // size of the workgroup
                        local volatile short4 *segments,// size of the workgroup
                        local volatile int* cnt                  // size 1 is enough 
                        ){
    int wg = get_local_size(0);// workgroup size
    int tid = get_local_id(0); // thread id
    int pid = tid + start;
    // Ensure we have at least 1 (pre-segment) defined at the begining
    if (tid == 0){
        //start with the begining of the stream
        segments[atomic_inc(cnt)] = (short4)(start, 0, 0, 0);            
    }

    barrier(CLK_LOCAL_MEM_FENCE);
         
    if ((tid>0) && (pid<stop)){
        short here = match_buffer[tid],
              there= match_buffer[tid-1];
        if ((there==0) && (here>=MIN_MATCH)){
            segments[atomic_inc(cnt)] = (short4)(pid+1, 0, here, 1);
        } else
//        if ((here==0) && (there>0) && (tid>5) && match_buffer[tid-5]>4){
        if ((here==0) && (there>0) && (tid>=MIN_MATCH) && match_buffer[tid-MIN_MATCH]>=MIN_MATCH){
            segments[atomic_inc(cnt)] = (short4)(pid+1, 0, 0, 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (cnt[0] == 1){
        // nothing occured in considered 
        if (tid == 0){
            // noting occured, just complete former segment
            short4 seg=segments[0];
            if (seg.s2 == 0){ //there was no match, just complete the former segment
                seg.s1 += end-start;
                segments[0] = seg;
            }
            else{ // noting occured, but former segment has already some match !
                if (tid==0){
                    segments[atomic_inc(cnt)] = (short4)(start, end-start, 0, 0); 
                }
            }
        }        
    }
    else{
        // sort segments
        sort_odd_even(0, cnt[0], segments);
        //add end position as a litteral
        if (tid==0){
            segments[cnt[0]] = (short4)(end, 0, 0, 0);
            atomic_inc(cnt);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid==0){
            printf("after match scan, before compaction, cnt=%d start=%d end=%d stop=%d\n",cnt[0], start, end, stop);
        }
        // compact segments
        cnt[0] = compact_segments(segments, cnt);
    }
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
    local volatile int cnt[1];    
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
    local volatile int seg[1];
    local volatile short4 lsegments[TEST_WG];
    local uchar lbuffer[TEST_BUFFER];
    local short lmatch[TEST_WG];
    seg[0] = 0;
    
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
    int res2 = segmentation(start, stop, res, lmatch, lsegments, seg);
    nbsegment[0] = res2;
    if (tid<res2){
        segments[tid] = lsegments[tid];
    }
}

// kernel to test multiple scan4match+segmentation WG<64 buffer<1024.
kernel void test_multi(global uchar *buffer,
                              int start, //index where scan should start
                              int stop,
                              global int *nbsegment,
                              global short4 *segments // size of the workgroup
){
    local volatile int seg[2]; // #0:number of segments in local mem, #1 in global mem
    local volatile int cnt[1]; // end position of the scan   
    local volatile short4 lsegments[TEST_WG];
    local uchar lbuffer[TEST_BUFFER];
    local short lmatch[TEST_WG];
    
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int actual_buffer_size = min(TEST_BUFFER, stop);
    int watchdog = (stop-start+wg-1)/wg; //prevent code from running way !
    int res, res2;
    //copy input to local buffer
    for (int i=tid; i<stop; i+=wg){
            lbuffer[i] = buffer[i];
    }
    if (tid<2){
        seg[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    while ((watchdog)&&(start+1<actual_buffer_size)){
        if ((tid==0) && (gid==0)) printf("watchdog: %d start: %d buffer_size %d\n", watchdog, start, actual_buffer_size);
        watchdog--;
        
        
        //scan for matching
        res = scan4match(lbuffer, start, stop, lmatch, cnt);
        if ((tid==0) && (gid==0))printf("### scanned input buffer at position  %d-%d\n", start, res);
        res2 = segmentation(start, stop, res, lmatch, lsegments, seg);
        if ((tid==0) && (gid==0)){
            printf("Extracted %d segments\n", res2);
            for (int i=0; i<res2; i++){
                short4 seg = lsegments[i];
                printf("seg#%d (%d, %d, %d, %d)\n",i,seg.s0,seg.s1,seg.s2,seg.s3);
            }
        }
        
        
        // copy segments to global memory:
        if (tid+1<res2){
                segments[seg[1] + tid] = lsegments[tid];
        }
        if (tid==0)printf("copy segments -> %d to memory %d-%d\n",res2-1,seg[1], seg[1]+res2-1);
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (tid == 0){
            seg[1] += res2-1;
            lsegments[0] = lsegments[res2-1];
            seg[0] = 1;
            short4 seg = lsegments[0];
            printf("copied seg[0] (was %d) (%d, %d, %d, %d)\n",res2-1,seg.s0,seg.s1,seg.s2,seg.s3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //memset local segments above first one,
        if (tid>1) lsegments[tid] = (short4)(0,0,0,0);
        barrier(CLK_LOCAL_MEM_FENCE);
        start = res;
        if (tid==5)printf("end of loop, start=%d res=%d size=%d\n\n", start, res, actual_buffer_size);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid == 0){
        segments[seg[1]++] = lsegments[0];
        nbsegment[0] = seg[1];
        printf("last copy, total segments: %d\n", seg[1]);
    }
}
  