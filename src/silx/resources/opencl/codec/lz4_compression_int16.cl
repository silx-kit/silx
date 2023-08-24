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
#define WORKGROUP_SIZE 1024
#endif
//segment size should be buffer_size/4
//#ifndef SEGMENT_SIZE
//#define SEGMENT_SIZE 512
//#endif

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 16384
#endif
#ifndef MIN_MATCH
#define MIN_MATCH 4
#endif

/***************************
 * Odd-Even Sort algorithm * 
 ***************************/

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


/**************************************
 * Cumsum based on Hillis Steele Scan * 
 **************************************/
// calculate the cumulative sum of element in the array inplace.
inline void cumsum_short(local volatile short *array,
                                       int   size){
    int oid, tid = get_local_id(0);
    short here, there;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < size; offset *= 2){
        here = (tid < size) ? array[tid] : 0;
        oid = tid-offset;
        there = ((tid < size)&&(oid>=0)) ? array[oid] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid<size){
            if (tid >= offset)
                array[tid] = here+there;
            else
                array[tid] = here;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


/* *****************************************************************************************************
 * Compact litterals and matches into segments containing a litteral and a match section (non null)
 * 
 * After the scan, begining of litterals and of match are noted and stored in segments.
 * In this function one takes 2 segments, starting with a litteral and concatenate the subsequent match  
 * as a consequence, the number of segments is divided by 2 !
 *******************************************************************************************************/
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
inline int scan4match(local uchar *buffer,       // buffer with input data in it, as large as possible, limited by shared memory space.
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
//    if (tid==0)printf("workgroup size is %d\n",WORKGROUP_SIZE);
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


// calculate the length of a segment in compressed form
inline short length_segment(short4 segment){
    short lit = segment.s1;
    short mat = segment.s2-4;    
    if ((lit==0) && (mat==-4)) 
        return 0;
            
    short size = 3+lit;
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


// fill the s3 index with the compressed size of each segment
inline void calculate_output_size(local volatile short4 *segments,// size of the workgroup
                                  int start, int stop){
    int tid = get_local_id(0);
    if ((tid>=start) && (tid<stop)){
        short4 seg = segments[tid];
        seg.s3 = length_segment(seg);
        segments[tid] = seg;                              
    }
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
    // update the segment with the compressed size:
    calculate_output_size(segments, 0, cnt[0]);
    return cnt[0];
}


//  Build token, concatenation of a litteral and a match 
inline uchar build_token(int4 segment){
    int lit = segment.s1;
    int mat = segment.s2;
    int token = ((lit & 15)<<4)|((mat-4)&15);
    return token;
}


// copy collaborative, return the position in output stream.
inline int copy_local(global uchar* dest,
                      const int dest_position,
                      local uchar* source,
                      const int src_position,
                      const int length){
    for (int i=get_local_id(0); i<length; i+=get_local_size(0)) {
        dest[dest_position+i] = source[src_position+i];
    }
    return dest_position+length;
}
// copy collaborative, return the position in output stream.
inline int copy_global(global uchar* dest,
                       const int dest_position,
                       global uchar* source,
                       const int src_position,
                       const int length){
    for (int i=get_local_id(0); i<length; i+=get_local_size(0)) {
        dest[dest_position+i] = source[src_position+i];
    }
    return dest_position+length;
}


/*
 * Perform the actual compression by copying a single segment
 * 
 * return the end-position in the output stream 
 */

inline int write_segment(global uchar *input_buffer, // buffer with input uncompressed data
                                int input_size,      // size of the  data to be compressed 
                                int4 segment,        // segment to be compressed
                         global uchar *output_buffer,// destination buffer for compressed data
                                int output_size,     //  
                                int last_segment     // set to 1 to indicate this is the last segment
){

    int rem;
    int start_dec = segment.s0;
    int litter = segment.s1;
    int match = segment.s2;
    int start_cmp = segment.s3;
    
    if ((litter==0) && (match==0)){// this was the last segment
        return -1;
    }
    if ((start_dec>=input_size) || (start_cmp>=output_size)){// this segment read/write outsize boundaries
        return -1;
    }
    
    if (last_segment){
        litter += match;
        match = 0;
        segment.s1 = litter;
        segment.s2 = match;
//        if(tid==0)printf("last segment %d %d %d %d\n", segment.s0, segment.s1, segment.s2, segment.s3);
    }
    
    //write token
    int token_idx = start_cmp++;
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
    start_cmp = copy_global(output_buffer, start_cmp,
                            input_buffer, start_dec, litter);
    
    if (!last_segment){ // last block has no offset, nor match
        //write offset, here always 1 in 16 bits little endian !
        output_buffer[start_cmp++] = 1;
        output_buffer[start_cmp++] = 0;
        
        //write match overflow
        if (match>=19){
            rem = match-19;
            while (rem>=255){
                output_buffer[start_cmp++] = 255;
                rem -= 255;
            }
            output_buffer[start_cmp++] = rem;
        }
    }
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
                          global short4 *global_segments,
                                 int max_idx, // last position achievable in global segment array
                                 int global_idx,
                                 int input_stream_idx,
                                 int output_stream_idx,
                                 int block_size,   // size of the block under analysis 
                                 int last,         // set to true to concatenate the match and the litteral for last block
                          local volatile int* cnt, //size=2 is needed, index 0 for size, index 1 for emergency   
                          local short* tmparray    // size: workgroup size
                          ){
    int tid = get_local_id(0);
    cnt[0] = output_stream_idx;
    cnt[1] = 0;
    short4 segment;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx<max_idx){
        if (tid<nb_segments){
            segment = local_segments[tid];
            // manage too few space in segment storage
            int emergency = (global_idx+1+tid == max_idx); 
            if (emergency){ // store all the remaining of the block in current segment
                cnt[1] = tid+1;
                printf("gid %lu emergency %d %d, segment starts at %d -> %d\n", get_group_id(0), global_idx, max_idx, segment.s0, block_size);
                segment.s1 = block_size - segment.s0;
                segment.s2 = 0;
                segment.s3 = length_segment(segment);
            }
            // manage last segment in block, i.e. transform match in litteral.
            if ((last) && (tid+1==nb_segments)){
                segment.s1+=segment.s2;
                segment.s2 = 0;
                segment.s3 = length_segment(segment);
            }
            tmparray[tid] = segment.s3;
        }
        else tmparray[tid] = 0;
        cumsum_short(tmparray, nb_segments);
        nb_segments = cnt[1]?cnt[1]:nb_segments;
        if (tid==0){
            segment.s3 = output_stream_idx;
        } 
        else if (tid<nb_segments) {
            segment.s3 = output_stream_idx + tmparray[tid-1];            
        }
        if (tid<nb_segments)
            global_segments[global_idx+tid]=segment;
        output_stream_idx += tmparray[nb_segments-1];
    }
                                      
    return output_stream_idx;
}

/* concatenate all segments (stored in global memory) in such a way that they are adjacent.
 * This function is to be called by the latest workgroup running.
 * 
 * Returns the number of segments and the number of bytes to be written.
 * 
 * There are tons of synchro since data are read and written from same buffer. 
 */  
inline int2 concatenate_segments(
        global int2 *segment_ptr,        // size = number of workgroup launched, contains start and stop position
        global int4 *segments,           // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
        global int *output_size,         // output buffer size, max in input, actual value in output
        local volatile int *shared_idx, // shared indexes with segment offset(0), output_idx(1)
        local volatile int4 *last_segment // shared memory with the last segment to share between threads
        ){
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int ng = get_num_groups(0);// number of groups
    
//    if (tid==0) printf("gid %d, running concat_segments \n", gid);
    int4 segment;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (tid==0){ 
        shared_idx[0] = segment_ptr[0].s1;
        shared_idx[1] = output_size[0];
        segment = segments[max(0, shared_idx[0]-1)];
        if ((segment.s0>0) && (segment.s2==0) && (ng>1)){
            last_segment[0] = segment;
            shared_idx[0] -= 1;
        }
        else{
            last_segment[0] = (int4)(0,0,0,0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (tid==0) printf("groups range from 1 to %d. segment_idx=%d, output_ptr=%d\n",ng, shared_idx[0], shared_idx[1]); 
    for (int grp=1; grp<ng; grp++){
        int2 seg_ptr = segment_ptr[grp];
        int low = seg_ptr.s0 + tid;
        int high = (seg_ptr.s1+wg-1)&~(wg-1);
//        if (tid==0) printf("grp %d read from %d to %d and writes to %d to %d\n",grp, low, high, shared_idx[0], shared_idx[0]+seg_ptr.s1-seg_ptr.s0);
        // concatenate last segment with first one if needed
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid == 0) && (last_segment[0].s0>0) && (last_segment[0].s2==0)){
            segment = segments[seg_ptr.s0];
            segment.s0 = last_segment[0].s0;
            segment.s1 = segment.s0+segment.s1-last_segment[0].s0;
            shared_idx[1] += len_segment(segment)-len_segment(last_segment[0]);
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
            segment.s3+=shared_idx[1];
            
            if (i<seg_ptr.s1){
//                printf("tid %d read at %d write at %d (%d, %d, %d, %d)\n",tid, i, shared_idx[0]+i-seg_ptr.s0, segment.s0, segment.s1, segment.s2, segment.s3);
                segments[shared_idx[0]+i-seg_ptr.s0] = segment;
                //segments[i] = (int4)(0,0,0,0);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            // if last block has match==0, concatenate with next one
            if ((i+1==seg_ptr.s1) && (segment.s0>0) && (segment.s2==0) && (grp+1<ng)){
                last_segment[0] = segment;
                shared_idx[0] -= 1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid==0){
            shared_idx[0]+=seg_ptr.s1-seg_ptr.s0;
            segment_ptr[grp] = (int2)(0,0);
            shared_idx[1] += output_size[grp];
            output_size[grp] = 0;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);  
    if (tid==0){
        segment_ptr[0] = (int2)(0, shared_idx[0]);
        output_size[0] = shared_idx[1];
    }

    return (int2) (shared_idx[0], shared_idx[1]);
} // end concatenate_segments

    
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

// kernel to validate the cumsum implementation
kernel void test_cumsum(global short *buffer, int size, volatile local short *lbuffer){
    int tid = get_local_id(0);
    if (tid<size) lbuffer[tid] = buffer[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    cumsum_short(lbuffer, size);
    if (tid<size) buffer[tid] = lbuffer[tid];
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
//    if ((tid==0) && (gid==0))printf("scanned up to %d\n", res);
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
                              global int4 *segments // size of the workgroup
){
    local volatile int cnt[2];
    local volatile int seg[1];
    local volatile short4 lsegments[WORKGROUP_SIZE];
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
//    if ((tid==0) && (gid==0))printf("scanned up to %d\n", res);
    int res2 = segmentation(start, stop, res, lmatch, lsegments, seg);
    nbsegment[0] = res2;
    if (tid<res2){
        segments[tid] = convert_int4(lsegments[tid]);
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
    local volatile short4 lsegments[WORKGROUP_SIZE];
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
  

// kernel to test the function `concatenate_segments`, run on only one workgroup
kernel void test_concatenate_segments(                            
        global int2 *segment_ptr, // size = number of workgroup launched, contains start and stop position
        global int4 *segments,    // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
        global int *output_size  // output buffer size, max in input, actual value in output, size should be at least the 
        ){
    local volatile int cnt[2]; //0:segment_ptr, 1:output_ptr
    local volatile int4 last_segment[1];
    
    int gid = get_group_id(0); // group id
    int tid = get_local_id(0); // thread id
    if (gid == 0){
        int2 end_ptr = concatenate_segments(segment_ptr,         // size = number of workgroup launched, contains start and stop position
                                           segments,            // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
                                           output_size,         // output buffer size, max in input, actual value in output
                                           cnt,                 // index of segment offset, shared
                                           last_segment         // shared memory with the last segment to share between threads
                                           );
    }
}

// kernel to test multiple blocks in parallel with last to finish which manages the junction between blocks
// segment description: s0: position in input buffer s1: number of litterals, s2: number of match, s3: size/position in output buffer
kernel void LZ4_cmp_stage1(global uchar *buffer,
                                   int input_size,
                           local  uchar *lbuffer,     // local buffer of size block_size for caching buffer.
                                   int block_size,    // size of the block
                            global int4 *block_ptr,   // size = number of workgroup launched, i.e. number of LZ4-blocks. contains, start+end segment, start+end write 
                            global short4 *segments,    // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
                            int final_compaction,     // set to 0 to prevent the final compaction. allows the analysis of intermediate results
                            global int *output_size,  // output buffer size, max in input, actual value in output, size should be at least the 
                            global int *wgcnt         // counter with workgroups still running
){
    local volatile int seg[2]; // #0:number of segments in local mem, #1 in global mem
    local volatile int cnt[2]; // end position of the scan   
    local volatile short4 lsegments[WORKGROUP_SIZE];
    local short lmatch[WORKGROUP_SIZE];
    local volatile short4 last_segment[1];
    
    
    int tid = get_local_id(0); // thread id
    int gid = get_group_id(0); // group id
    int wg = get_local_size(0);// workgroup size
    int ng = get_num_groups(0);// number of groups
    
    int output_block_size = 0;
    int output_idx = output_block_size*gid;
    int4 seg_ptr = block_ptr[gid];
    int segment_idx = seg_ptr.s0;
    int segment_max = seg_ptr.s1;
    int local_start = 0; 
    int global_start = block_size * gid;
    int local_stop = min(block_size, input_size - global_start);
    if (local_stop<=0){
        if (tid==0)printf("gid %d local_stop: %d \n",gid, local_stop);
            return;
    }
           
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
        if (tid==0) printf("gid %d watchdog %d scan4match gave %d\n",gid, watchdog, res);
        res2 = segmentation(local_start, local_stop, res, lmatch, lsegments, seg);
        if (tid==0) printf("gid %d watchdog %d segmentation gave %d\n",gid, watchdog, res2);

        // copy segments to global memory:
        int segment_to_copy = res2 - 1;
        if (tid==0) printf("gid %d watchdog %d about to save %d segments\n",gid, watchdog, segment_to_copy);
        output_idx = store_segments(lsegments, segment_to_copy, // last segment is kept for the future ...
                                    segments, segment_max, segment_idx, global_start, output_idx,  local_stop, 0, cnt, lmatch);
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
                                segments, segment_max, segment_idx, global_start, output_idx, local_stop, gid+1==ng, cnt, lmatch);
    output_size[gid] =  output_idx;
    seg_ptr.s1 = ++segment_idx;
    seg_ptr.s3 = output_idx;
    block_ptr[gid] = seg_ptr; 

    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    // last group running performs the cumsum and compaction of indices
    if (tid==0){
        cnt[0] = (atomic_dec(wgcnt)==1);
    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (cnt[0] && final_compaction){//TODO: redo
//        int2 end_ptr = concatenate_segments(block_ptr,         // size = number of workgroup launched, contains start and stop position
//                                            segments,            // size of the block-size (i.e. 1-8k !wg) / 4 * number of workgroup  
//                                            output_size,         // output buffer size, max in input, actual value in output
//                                            cnt,                 // index of segment offset, shared
//                                            last_segment         // shared memory with the last segment to share between threads
//                                            );
//    }
}


// kernel launched with one block per workgroup. 
//If the segment has large litterals, having many threads per group is interesting. 

kernel void LZ4_cmp_stage2(global uchar *input_buffer,   // bufffer with data to be compressed
                                     int input_size,        // size of the  data to be compressed 
                                     int block_size,        // size of each block
                              global int4 *block_ptr,       // size = numblocks, contains contains the start and end index in segment array and start and end position in the output array 
                              global short4 *segments,      // size defined by segment_ptr, constains segments relative to the begining on the block
                              global uchar *output_buffer,  // destination buffer for compressed data
                              global  int *output_size,       // size of the destination buffer                              
                                     int prefix_header      // if set, put in header the input buffer size (increases the output_size[0] by 4)
){
    int gid = get_group_id(0);
    int tid = get_local_id(0);
    int wg = get_local_size(0);
    int ng = get_num_groups(0);
    int4 segment_range =  block_ptr[gid];
    int input_offset = block_size*gid;
    int output_offset = segment_range.s2 + (prefix_header) ? 4 : 0;
    short4 short_segment;
    int4   int_segment; 
    int r_size = output_size[0];

    if (prefix_header){
        if ((gid == 0) && (tid==0)){//write 
            output_buffer[0] = input_size & 0xFF;
            output_buffer[1] = (input_size>>8) & 0xFF;
            output_buffer[2] = (input_size>>16) & 0xFF;
            output_buffer[3] = (input_size>>24) & 0xFF;
        }
    }
    
    for (int i=segment_range.s0; i<segment_range.s1; i++){
        short_segment = segments[i];
        int_segment = (int4)(short_segment.s0+input_offset,
                             short_segment.s1,
                             short_segment.s2,
                             short_segment.s3+output_offset);
        if ((gid+1==segment_range.s1)&&(gid+1==ng)){//last segment
            int actual_size = write_segment(input_buffer,  // buffer with input uncompressed data
                                   input_size,    // size of the  data to be compressed 
                                   int_segment,   // segment to be compressed
                                   output_buffer, // destination buffer for compressed data
                                   r_size,   // size of the output buffer  
                                   1);
            if (tid==0) output_size[0] = actual_size;
        }
        else{
            write_segment(input_buffer,  // buffer with input uncompressed data
                          input_size,    // size of the  data to be compressed 
                          int_segment,   // segment to be compressed
                          output_buffer, // destination buffer for compressed data
                          r_size,   //  size of the output buffer
                          0);
        }
        
    }//loop over all segments in a block.
}
