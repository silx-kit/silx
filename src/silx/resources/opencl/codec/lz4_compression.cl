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
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 64
#endif
//segment size should be buffer_size/4
#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 256
#endif

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 1024
#endif
#ifndef MIN_MATCH
#define MIN_MATCH 4
#endif



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

/* Compact litterals and matches into segments containing a litteral and a match section (non null)
 * 
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
            // nothing occured, just complete former segment
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
//        if (tid==0){
//            printf("after match scan, before compaction, cnt=%d start=%d end=%d stop=%d\n",cnt[0], start, end, stop);
//        }
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
                     int stop, //output buffer max size
                     int continuation // set to 0 to indicate this is the last segment
                    )
{
    for (int i=0; i<nb_segments; i++){
        short4 segment = segments[i];
        if ((segment.s1==0) && (segment.s2==0)){// this was the last segment
            break;
        }
        //write token
        int token_idx = start_cmp++;
        int rem;
        int litter = segment.s1;
        int match = segment.s2;
        if (litter >= 15){
            segment.s1 = 15;  
            rem = litter - 15;
            while (rem>=255){
                output_buffer[start_cmp++] = 255;
                rem -= 255;
            }
            output_buffer[start_cmp++] = rem;
        }
        if (match >= 19){
            segment.s2 = 19; 
        }
        output_buffer[token_idx] = build_token(segment);

        //copy litteral. This is collaborative.
        start_cmp = copy(output_buffer, start_cmp,
                         buffer, segment.s0, litter);
        
        if ((continuation)||(i+1<nb_segments)){ // last block has no offset, nor match
            //write offset, here always 1 in 16 bits little endian !
            output_buffer[start_cmp++] = 1;
            output_buffer[start_cmp++] = 0;
            
            //write match overflow
            if (segment.s2>=19){
                rem = segment.s2-19;
                while (rem>=255){
                    output_buffer[start_cmp++] = 255;
                    rem -= 255;
                }
                output_buffer[start_cmp++] = rem;
            }
        }
    }//loop over segments
    return start_cmp;
}

// calculate the length of a segment in compressed form
inline int len_segment(int4 segment){
    int lit = segment.s1;
    int mat = segment.s2-4;    
    int size = 3+lit;
    if (lit>=15){
        size++;
        lit -= 15;
    }
    while (lit>255){
        size++;
        lit-=255;
    }
    if (mat>=15){
        size++;
        mat -= 15;
    }
    while (mat>255){
        size++;
        mat-=255;
    }
    return size;
}

/* store several local segments into the global memory starting at position. 
 * return the position in the output stream
 */
inline int store_segments(local volatile short4 *local_segments,
                                 int nb_segments,
                          global int4 *global_segments,
                                 int max_idx, // last position achievable in global segment array
                                 int global_idx,
                                 int input_stream_idx,
                                 int output_stream_idx,
                                 int block_size, // size of the block under analysis 
                                 int last, // set to true to concatenate the match and the litteral for last block
                          local volatile int* cnt //size=1 is eough  
                          ){
    cnt[0] = output_stream_idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx!=max_idx){
        //this is serial for conviniance !    
        if (get_local_id(0)==0){
            for (int i=0; i<nb_segments; i++){
                int4 segment = convert_int4(local_segments[i]);
                
                // manage too few space in segment storage
                int emergency = (global_idx+1 == max_idx); 
                if (emergency){ // store all the remaining of the block in current segment
                    printf("gid %lu emergency %d %d, segment starts at %d -> %d\n", get_group_id(0), global_idx, max_idx, segment.s0, block_size);
                    segment.s1 = block_size - segment.s0;
                    segment.s2 = 0;
                }
                // manage last segment in block
                if (last){
                    segment.s1+=segment.s2;
                    segment.s2 = 0;
                }
                segment.s0 += input_stream_idx;
                segment.s3 = output_stream_idx;
                
                output_stream_idx += len_segment(segment);
                global_segments[global_idx++]=segment;
                if (emergency) break;
            }
            cnt[0] = output_stream_idx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);        
    }
    return cnt[0];
}

/* concatenate all segments (stored in global memory) in such a way that they are adjacent.
 * This function is to be called by the latest workgroup running.
 * 
 * There are tons of synchro since data are read and written from same buffer. 
 */  
inline int concatenate_segments(
        global int2 *segment_ptr,        // size = number of workgroup launched, contains start and stop position
        global int4 *segments,           // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  

        global int *output_size,         // output buffer size, max in input, actual value in output
        local volatile int *lsegment_idx, // index of segment offset, shared
        local volatile int4 *last_segment // shared memory with the last segment to share between threads
        ){
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int ng = get_num_groups(0);// number of groups
    
//    if (tid==0) printf("gid %d, running concat_segments \n", gid);
    int4 segment;
    barrier(CLK_GLOBAL_MEM_FENCE);
    int output_idx = output_size[0];
    lsegment_idx[0] = segment_ptr[0].s1;
    segment = segments[max(0, lsegment_idx[0]-1)];
    if ((tid==0) && (segment.s0>0) && (segment.s2==0) && (ng>1)){
        last_segment[0] = segment;
        lsegment_idx[0] -= 1;
    }
    
    last_segment[0] = (int4)(0,0,0,0);
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (tid==0) printf("groups range from 1 to %d. segment_idx=%d, output_ptr=%d\n",ng, lsegment_idx[0], output_idx); 
    for (int grp=1; grp<ng; grp++){
        int2 seg_ptr = segment_ptr[grp];
        int low = seg_ptr.s0 + tid;
        int high = (seg_ptr.s1+wg-1)&~(wg-1);
//        if (tid==0) printf("grp %d read from %d to %d and writes to %d to %d\n",grp, low, high, lsegment_idx[0], lsegment_idx[0]+seg_ptr.s1-seg_ptr.s0);
        // concatenate last segment with first one if needed
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid == 0) && (last_segment[0].s0>0) && (last_segment[0].s2==0)){
            segment = segments[seg_ptr.s0];
            segment.s0 = last_segment[0].s0;
            segment.s1 = segment.s0+segment.s1-last_segment[0].s0;
            output_idx += len_segment(segment)-len_segment(last_segment[0]);
            last_segment[0] = (int4)(0,0,0,0);
            segments[seg_ptr.s0] = segment;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i=low; i<high;i+=wg){
            
            if (i<seg_ptr.s1){
                segment = segments[i];
            }
            else
                segment = (int4)(0,0,0,0);
            barrier(CLK_GLOBAL_MEM_FENCE);                
            segment.s3+=output_idx;
            
            if (i<seg_ptr.s1){
//                printf("tid %d read at %d write at %d (%d, %d, %d, %d)\n",tid, i, lsegment_idx[0]+i-seg_ptr.s0, segment.s0, segment.s1, segment.s2, segment.s3);
                segments[lsegment_idx[0]+i-seg_ptr.s0] = segment;
                //segments[i] = (int4)(0,0,0,0);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            // if last block has match==0, concatenate with next one
            if ((i+1==seg_ptr.s1) &&  (segment.s0>0) && (segment.s2==0)&&(grp+1<ng)){
                last_segment[0] = segment;
                lsegment_idx[0] -= 1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid==0){
            lsegment_idx[0]+=seg_ptr.s1-seg_ptr.s0;
        }
        segment_ptr[grp] = (int2)(0,0);
        output_idx += output_size[grp];
        output_size[grp] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);    
    return lsegment_idx[0];
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
    local volatile short4 lsegments[SEGMENT_SIZE];
    local uchar lbuffer[BUFFER_SIZE];
    local short lmatch[WORKGROUP_SIZE];
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
    local volatile short4 lsegments[SEGMENT_SIZE];
    local uchar lbuffer[BUFFER_SIZE];
    local short lmatch[WORKGROUP_SIZE];
    
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int actual_buffer_size = min(BUFFER_SIZE, stop);
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
        barrier(CLK_LOCAL_MEM_FENCE);
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
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0){
        segments[seg[1]++] = lsegments[0];
        nbsegment[0] = seg[1];
        printf("last copy, total segments: %d\n", seg[1]);
    }
}
  
// kernel to test multiple scan4match+segmentation+write WG<64 buffer<1024.
kernel void test_write(global uchar *buffer,
                              int start, //index where scan should start
                              int stop,
                       global int *nbsegment,
                       global short4 *segments, // size of the workgroup
                       global uchar *output,    // output buffer
                       global int *output_size  // output buffer size, max in input, actual value in output
){
    local volatile int seg[2]; // #0:number of segments in local mem, #1 in global mem
    local volatile int cnt[1]; // end position of the scan   
    local volatile short4 lsegments[SEGMENT_SIZE];
    local uchar lbuffer[BUFFER_SIZE];
    local short lmatch[WORKGROUP_SIZE];
    
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int actual_buffer_size = min(BUFFER_SIZE, stop);
    int watchdog = (stop-start+wg-1)/wg; //prevent code from running way !
    int res, res2, out_ptr=0, max_out=output_size[0];
    
    //copy input to local buffer
    for (int i=tid; i<stop; i+=wg){
            lbuffer[i] = buffer[i];
    }
    if (tid<2){
        seg[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    while ((watchdog--) && (start+1<actual_buffer_size)){
        //scan for matching
        res = scan4match(lbuffer, start, stop, lmatch, cnt);
        res2 = segmentation(start, stop, res, lmatch, lsegments, seg);
        if ((tid==0) && (gid==0)){
            for (int i=0; i<res2; i++){
                short4 seg = lsegments[i];
            }
        }
        // copy segments to global memory:
        if (tid+1<res2){
                segments[seg[1] + tid] = lsegments[tid];
        }
        // copy data to compressed buffer
        if (res2>1)
            out_ptr = write_lz4(lbuffer, lsegments,
                                res2-1, // -1? to keep the last for concatenation
                                out_ptr, output,max_out, 1);
                
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid == 0){
            seg[1] += res2-1;
            lsegments[0] = lsegments[res2-1];
            seg[0] = 1;
//            short4 seg = lsegments[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //memset local segments above first one,
        if (tid>1) lsegments[tid] = (short4)(0,0,0,0);
        barrier(CLK_LOCAL_MEM_FENCE);
        start = res;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0){
        short4 segment = lsegments[0];
        segment.s1 += segment.s2;
        segment.s2 = 0;
        lsegments[0] = segment;
                            
        segments[seg[1]++] = segment;
        nbsegment[0] = seg[1];
        printf("last segment %d %d %d %d\n", segment.s0, segment.s1, segment.s2, segment.s3);
    }
    // write last segment

    out_ptr = write_lz4(lbuffer, lsegments,
                            1, out_ptr, output, max_out, 0);

    output_size[0] =  out_ptr;   
        
    
}

// kernel to test multiple blocks in parallel with last to finish who concatenates segments WG<64 buffer<1024.
// segment description: s0: position in input buffer s1: number of litterals, s2: number of match, s3: position in output buffer
kernel void test_multiblock(global uchar *buffer,
                                   int input_size,
                            global int2 *segment_ptr, // size = number of workgroup launched, contains start and stop position
                            global int4 *segments,    // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
                            global uchar *output,     // output buffer
                            global int *output_size,  // output buffer size, max in input, actual value in output
                            global int *wgcnt         // counter with workgroups still running
){
    local volatile int seg[2]; // #0:number of segments in local mem, #1 in global mem
    local volatile int cnt[1]; // end position of the scan   
    local volatile short4 lsegments[SEGMENT_SIZE];
    local uchar lbuffer[BUFFER_SIZE];
    local short lmatch[WORKGROUP_SIZE];
    local volatile int4 last_segment[1];
    
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int ng = get_num_groups(0);// number of groups
    
    int output_block_size = 0;
    int output_idx = output_block_size*gid;
    int2 seg_ptr = segment_ptr[gid];
    int segment_idx = seg_ptr.s0;
    int segment_max = seg_ptr.s1;
//    if (tid==0)printf("gid %d writes segments in range %d-%d\n", gid, segment_idx, segment_max);
    int local_start = 0; 
    int global_start = BUFFER_SIZE*gid;
    int local_stop = min(BUFFER_SIZE, input_size - global_start);
    if (local_stop<=0){
        if (tid==0)printf("gid %d local_stop: %d \n",gid, local_stop);
            return;
    }
    
//    int actual_buffer_size = min(BUFFER_SIZE, local_stop) ;
        
    int watchdog = (local_stop + wg-1)/wg; //prevent code from running way !
    int res, res2, out_ptr=0, max_out=output_size[0];
    
    //copy input to local buffer
    for (int i=tid; i<local_stop; i+=wg){
            lbuffer[i] = buffer[global_start+i];
    }
    if (tid<2){
        seg[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    while ((watchdog--) && (local_start+1<local_stop)){
        //scan for matching
        res = scan4match(lbuffer, local_start, local_stop, lmatch, cnt);
        res2 = segmentation(local_start, local_stop, res, lmatch, lsegments, seg);
        // copy segments to global memory:
        int segment_to_copy = res2 - 1;
//        if (tid==0)printf("gid %d store %d segments at %d\n",gid, segment_to_copy, segment_idx);
        output_idx = store_segments(lsegments, segment_to_copy, // last segment is kept for the future ...
                                    segments, segment_max, segment_idx, global_start, output_idx,  local_stop, 0, cnt);
        segment_idx += segment_to_copy;
                
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid == 0){
            seg[1] += segment_to_copy;
            lsegments[0] = lsegments[segment_to_copy];
            seg[0] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //memset local segments above first one,
        if (tid>1) lsegments[tid] = (short4)(0,0,0,0);
        barrier(CLK_LOCAL_MEM_FENCE);
        local_start = res;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (tid==0)printf("gid %d store final segments\n",gid);
    output_idx = store_segments(lsegments, 1, // last segment is treated here
                                segments, segment_max, segment_idx, global_start, output_idx, local_stop, gid+1==ng, cnt);
    output_size[gid] =  output_idx;
    seg_ptr.s1 = ++segment_idx;
    segment_ptr[gid] = seg_ptr; 

    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    // last group running performs the cumsum and compaction of indices
    if (tid==0){
        cnt[0] = (atomic_dec(wgcnt)==1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (cnt[0]){
        int end_ptr = concatenate_segments(segment_ptr,         // size = number of workgroup launched, contains start and stop position
                                           segments,            // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
                                           output_size,         // output buffer size, max in input, actual value in output
                                           cnt,                 // index of segment offset, shared
                                           last_segment         // shared memory with the last segment to share between threads
                                           );
        segment_ptr[0] = (int2)(0, end_ptr);        
    }
}