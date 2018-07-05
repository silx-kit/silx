/*
 *   Project: SIFT: An algorithm for image alignement
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

#define DOUBLEMIN(a,b,c,d) ((a) < (c) ? ((b) < (c) ? (int2)(a,b) : (int2)(a,c)) : ((a) < (d) ? (int2)(c,a) : (int2)(c,d)))

#define ABS4(q1,q2) (int) (((int) (q1.s0 < q2.s0 ? q2.s0-q1.s0 : q1.s0-q2.s0)) + ((int) (q1.s1 < q2.s1 ? q2.s1-q1.s1 : q1.s1-q2.s1))+ ((int) (q1.s2 < q2.s2 ? q2.s2-q1.s2 : q1.s2-q2.s2)) + ((int) (q1.s3 < q2.s3 ? q2.s3-q1.s3 : q1.s3-q2.s3)))

#ifndef WORKGROUP_SIZE
    #define WORKGROUP_SIZE 64
#endif



/*
 *
 * \brief Compute SIFT descriptors matching for two lists of descriptors.
 *
 *  This version is optimized for GPU (vectorized instructions).
 *  As a descriptor value is a 1-byte data, we are reading 4 descriptor values at the same time in order to get coalesced memory access aligned on 32bits.
 *  This kernel works well if launched with a workgroup size of 64.
 *
 * :param keypoints1 Pointer to global memory with the first list of keypoints
 * :param keypoints: Pointer to global memory with the second list of keypoints
 * :param matchings: Pointer to global memory with the output pair of matchings (keypoint1, keypoint2)
 * :param counter: Pointer to global memory with the resulting number of matchings
 * :param max_nb_match: Absolute size limit for the resulting list of pairs
 * :param ratio_th: threshold for distances ; two descriptors whose distance is below this will not be considered as near. Default for sift.cpp implementation is 0.73*0.73 for L1 distance
 * :param size1: end index for processing list 1
 * :param size2: end index for processing list 2
 *
 * NOTE: a keypoint is (x,y,s,angle,[descriptors])
 *
*/




kernel void matching(
        global featured_keypoint* keypoints1,
        global featured_keypoint* keypoints2,
        global int2* matchings,
        global int* counter,
        int max_nb_match,
        float ratio_th,
        int size1,
        int size2)
{
    int gid0 = (int) get_global_id(0);
    if (!(0 <= gid0 && gid0 < size1))
    {
        return;
    }

    float dist1 = MAXFLOAT, dist2 = MAXFLOAT;
    int current_min = 0;
    int old;

    //pre-fetch
    uchar4 desc1[32];
    for (int i = 0; i<32; i++)
        desc1[i] = (uchar4) (((keypoints1[gid0]).desc)[4*i],
                             ((keypoints1[gid0]).desc)[4*i+1],
                             ((keypoints1[gid0]).desc)[4*i+2],
                             ((keypoints1[gid0]).desc)[4*i+3]);

    //each thread gid0 makes a loop on the second list
    for (int i = 0; i<size2; i++) {

        //L1 distance between desc1[gid0] and desc2[i]
        int dist = 0;
        for (int j=0; j<32; j++)
        { //1 thread handles 4 values
            uchar4 dval1 = desc1[j];
            uchar4 dval2 = (uchar4) (((keypoints2[i]).desc)[4*j],
                                    ((keypoints2[i]).desc)[4*j+1],
                                    ((keypoints2[i]).desc)[4*j+2],
                                    ((keypoints2[i]).desc)[4*j+3]);
            dist += ABS4(dval1,dval2);

        }
        
        if (dist < dist1)
        { //candidate better than the first
            dist2 = dist1;
            dist1 = dist;
            current_min = i;
        }
        else if (dist < dist2)
        { //candidate better than the second (but not the first)
            dist2 = dist;
        }
        
    }//end "i loop"
    
    if (dist2 != 0 && dist1/dist2 < ratio_th) {
        int2 pair = 0;
        pair.s0 = gid0;
        pair.s1 = current_min;
        old = atomic_inc(counter);
        if (old < max_nb_match) matchings[old] = pair;
    }
}



/*
 *
 * \brief Compute SIFT descriptors matching for two lists of descriptors, discarding descriptors outside a region of interest.
 *
 *  This version is optimized for GPU (vectorized instructions).
 *  As a descriptor value is a 1-byte data, we are reading 4 descriptor values at the same time in order to get coalesced memory access aligned on 32bits.
 *  This kernel works well if launched with a workgroup size of 64.
 *
 * :param keypoints1 Pointer to global memory with the first list of keypoints
 * :param keypoints: Pointer to global memory with the second list of keypoints
 * :param valid: Pointer to global memory with the region of interest (binary picture)
 * :param roi_width: Width of the Region Of Interest
 * :param roi_height: Height of the Region Of Interest
 * :param matchings: Pointer to global memory with the output pair of matchings (keypoint1, keypoint2)
 * :param counter: Pointer to global memory with the resulting number of matchings
 * :param max_nb_match: Absolute size limit for the resulting list of pairs
 * :param ratio_th: threshold for distances ; two descriptors whose distance is below this will not be considered as near. Default for sift.cpp implementation is 0.73*0.73 for L1 distance
 * :param size1: end index for processing list 1
 * :param size2: end index for processing list 2
 *
 * NOTE: a keypoint is (x,y,s,angle,[descriptors])
 *
*/


kernel void matching_valid(
    global featured_keypoint* keypoints1,
    global featured_keypoint* keypoints2,
    global char* valid,
    int roi_width,
    int roi_height,
    global int2* matchings,
    global int* counter,
    int max_nb_match,
    float ratio_th,
    int size1,
    int size2)
{
    int gid0 = (int) get_global_id(0);
    if (!(0 <= gid0 && gid0 < size1))
        return;

    float dist1 = MAXFLOAT, dist2 = MAXFLOAT;
    int current_min = 0;
    int old;

    actual_keypoint kp = keypoints1[gid0].keypoint;
    int c = kp.col, r = kp.row;
    //processing only valid keypoints
    if (r < roi_height && c < roi_width && valid[r*roi_width+c] == 0) return;

    //pre-fetch
    uchar4 desc1[32];
    for (int i = 0; i<32; i++)
        desc1[i] = (uchar4) (((keypoints1[gid0]).desc)[4*i],
                             ((keypoints1[gid0]).desc)[4*i+1],
                             ((keypoints1[gid0]).desc)[4*i+2],
                             ((keypoints1[gid0]).desc)[4*i+3]);

    //each thread gid0 makes a loop on the second list
    for (int i = 0; i<size2; i++) {

        //L1 distance between desc1[gid0] and desc2[i]
        int dist = 0;
        for (int j=0; j<32; j++) { //1 thread handles 4 values
            kp = keypoints2[i].keypoint;
            c = kp.col;
            r = kp.row;
            if (r < roi_height && c < roi_width && valid[r*roi_width+c] != 0)
            {
                uchar4 dval1 = desc1[j];
                uchar4 dval2 = (uchar4) (((keypoints2[i]).desc)[4*j],
                                        ((keypoints2[i]).desc)[4*j+1],
                                        ((keypoints2[i]).desc)[4*j+2],
                                        ((keypoints2[i]).desc)[4*j+3]);
                dist += ABS4(dval1,dval2);
            }
        }
        
        if (dist < dist1) { //candidate better than the first
            dist2 = dist1;
            dist1 = dist;
            current_min = i;
        }
        else if (dist < dist2) { //candidate better than the second (but not the first)
            dist2 = dist;
        }
        
    }//end "i loop"

    if (dist2 != 0 && dist1/dist2 < ratio_th)
    {
        int2 pair = 0;
        pair.s0 = gid0;
        pair.s1 = current_min;
        old = atomic_inc(counter);
        if (old < max_nb_match) matchings[old] = pair;
    }
}



















/*

    DO NOT USE ! Slow version.

    Let L2 be the length of "keypoints2" and W be the workgroup size.
    Each thread of the workgroup handles L2/W keypoints : [lid0*L2/W, (lid0+1)*L2/W[ ,
     and gives a pair of "best distance / second-best distance" (d1,d2)
    Then, we take d1 = min{(d1,d2) | all threads} and d2 = second_min {(d1,d2) | all threads}

     -----------------------------------------------
    |  thread 0 | thread 1 | ... | thread (W-1)    |
     -----------------------------------------------
     <---------->
    L2/W keypoints

    For this kernel W = 64


    DO NOT USE ! This version is actually slower than the first one, certainly the fact that we are reading "unsigned char".


*/



kernel void matching_v2(
        global featured_keypoint* keypoints1,
        global featured_keypoint* keypoints2,
        global int2* matchings,
        global int* counter,
        int max_nb_keypoints,
        float ratio_th,
        int end)
{

    int gid = get_group_id(0);
    int lid0 = get_local_id(0);
    if (!(0 <= gid && gid < end))
    {
        return;
    }
    float dist1 = MAXFLOAT, dist2 = MAXFLOAT;
    int current_min = 0;
    int old;

    local unsigned char desc1[64]; //store the descriptor of keypoint we are looking (in list 1)
    local int3 candidates[64];
    //local int3 parallel[64]; //for the parallel reduction


    for (int i = 0; i < 2; i++)
        desc1[i*64+lid0] = ((keypoints1[gid]).desc)[i*64+lid0];
    barrier(CLK_LOCAL_MEM_FENCE);
    int frac = (end >> 6)+1; //fraction of the list that will be processed by a thread
    int low_bound = lid0*frac;
    int up_bound = min(low_bound+frac,end);
    for (int i = low_bound; i<up_bound; i++) 
    {
        unsigned int dist = 0;
        for (int j=0; j<128; j++) 
        {
            unsigned char dval1 = desc1[j], dval2 = ((keypoints2[i]).desc)[j];
            dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2));
        }
        if (dist < dist1) 
        {
            dist2 = dist1;
            dist1 = dist;
            current_min = i;
        }
        else if (dist < dist2) 
        {
            dist2 = dist;
        }
    }//end "i loop"

    candidates[lid0] = (int3) (dist1, dist2, current_min);
    barrier(CLK_LOCAL_MEM_FENCE);

    //Now each block has its pair of best candidates (dist1,dist2) at position current_min
    //Find the global minimum and the "second minimum" : (min1,min2)


    int d1_0, d2_0, d1_1, d2_1, cmin_0, cmin_1;
    int2 sol;

    //parallel reduction

    if (lid0 < 32) 
    {
        d1_0 = candidates[lid0].s0;
        d2_0 = candidates[lid0].s1;
        d1_1 = candidates[lid0+32].s0;
        d2_1 = candidates[lid0+32].s1;
        cmin_0 = candidates[lid0].s2;
        cmin_1 = candidates[lid0+32].s2;
        sol = (int2) DOUBLEMIN(d1_0,d2_0,d1_1,d2_1);
        candidates[lid0] = (int3) (sol.s0, sol.s1, (sol.s0 == d1_0 ? cmin_0 : cmin_1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0 < 16) 
    {
        d1_0 = candidates[lid0].s0; d2_0 = candidates[lid0].s1; cmin_0 = candidates[lid0].s2;
        d1_1 = candidates[lid0+16].s0; d2_1 = candidates[lid0+16].s1; cmin_1 = candidates[lid0+16].s2;
        sol = (int2) DOUBLEMIN(d1_0,d2_0,d1_1,d2_1);
        candidates[lid0] = (int3) (sol.s0, sol.s1, (sol.s0 == d1_0 ? cmin_0 : cmin_1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0 < 8) 
    {
        d1_0 = candidates[lid0].s0; d2_0 = candidates[lid0].s1; cmin_0 = candidates[lid0].s2;
        d1_1 = candidates[lid0+8].s0; d2_1 = candidates[lid0+8].s1; cmin_1 = candidates[lid0+8].s2;
        sol = (int2) DOUBLEMIN(d1_0,d2_0,d1_1,d2_1);
        candidates[lid0] = (int3) (sol.s0, sol.s1, (sol.s0 == d1_0 ? cmin_0 : cmin_1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0 < 4) 
    {
        d1_0 = candidates[lid0].s0; d2_0 = candidates[lid0].s1; cmin_0 = candidates[lid0].s2;
        d1_1 = candidates[lid0+4].s0; d2_1 = candidates[lid0+4].s1; cmin_1 = candidates[lid0+4].s2;
        sol = (int2) DOUBLEMIN(d1_0,d2_0,d1_1,d2_1);
        candidates[lid0] = (int3) (sol.s0, sol.s1, (sol.s0 == d1_0 ? cmin_0 : cmin_1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0 < 2) 
    {
        d1_0 = candidates[lid0].s0; d2_0 = candidates[lid0].s1; cmin_0 = candidates[lid0].s2;
        d1_1 = candidates[lid0+2].s0; d2_1 = candidates[lid0+2].s1; cmin_1 = candidates[lid0+2].s2;
        sol = (int2) DOUBLEMIN(d1_0,d2_0,d1_1,d2_1);
        candidates[lid0] = (int3) (sol.s0, sol.s1, (sol.s0 == d1_0 ? cmin_0 : cmin_1));
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    if (lid0 == 0) 
    {
        d1_0 = candidates[lid0].s0; d2_0 = candidates[lid0].s1; cmin_0 = candidates[lid0].s2;
        d1_1 = candidates[lid0+1].s0; d2_1 = candidates[lid0+1].s1; cmin_1 = candidates[lid0+1].s2;
        sol = (int2) DOUBLEMIN(d1_0,d2_0,d1_1,d2_1);
        float dist10 = (float) sol.s0, dist20 = (float) sol.s1;
        int index_abs_min = (sol.s0 == d1_0 ? cmin_0 : cmin_1);
        if (dist20 != 0 && dist10/dist20 < ratio_th && gid <= index_abs_min) {
            int2 pair = 0;
            pair.s0 = gid;
            pair.s1 = index_abs_min;
            old = atomic_inc(counter);
            if (old < max_nb_keypoints) matchings[old] = pair;
        }
    }//end lid0 == 0
}
