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
 *  "Slow version", should work on CPU
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
 * NOTE: a keypoint is (col, row, scale, angle,[descriptors])
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
     int gid0 = get_global_id(0);
    if (!(0 <= gid0 && gid0 < size1))
    {
        return;
    }

    float dist1 = MAXFLOAT, dist2 = MAXFLOAT; 
    int current_min = 0;
    int old;

    //pre-fetch
    unsigned char desc1[128];
    for (int i = 0; i<128; i++)
    {
        desc1[i] = ((keypoints1[gid0]).desc)[i];
    }

    //each thread gid0 makes a loop on the second list
    for (int i = 0; i<size2; i++) 
    {
        //L1 distance between desc1[gid0] and desc2[i]
        int dist = 0;
        for (int j=0; j<128; j++) //1 thread handles 4 values (uint4) = 
        {
            
            unsigned char dval1 = desc1[j], dval2 = ((keypoints2[i]).desc)[j];
            dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2));
        }
        
        if (dist < dist1) //candidate better than the first
        {
            dist2 = dist1;
            dist1 = dist;
            current_min = i;
        }
        else if (dist < dist2) 
        { //candidate better than the second (but not the first)
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
 *
 * \brief Compute SIFT descriptors matching for two lists of descriptors, discarding descriptors outside a region of interest.
 *
 *  This version should work on CPU.
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
    int gid0 = get_global_id(0);
    if (!(0 <= gid0 && gid0 < size1))
        return;

    float dist1 = MAXFLOAT, dist2 = MAXFLOAT; 
    int current_min = 0;
    int old;

    actual_keypoint kp = keypoints1[gid0].keypoint;
    int c = kp.col, r = kp.row;
    //processing only valid keypoints
    if (r < roi_height && c < roi_width && valid[r*roi_width+c] == 0) 
    {
        return;
    }

    //pre-fetch
    unsigned char desc1[128];
    for (int i = 0; i<128; i++)
    {
        desc1[i] = ((keypoints1[gid0]).desc)[i];
    }
        
    //each thread gid0 makes a loop on the second list
    for (int i = 0; i<size2; i++)
    {
        //L1 distance between desc1[gid0] and desc2[i]
        int dist = 0;
        for (int j=0; j<128; j++) 
        { //1 thread handles 4 values
            kp = keypoints2[i].keypoint;
            c = kp.col, r = kp.row;
            if (r < roi_height && c < roi_width && valid[r*roi_width+c] != 0) 
            {
                unsigned char dval1 = desc1[j], dval2 = ((keypoints2[i]).desc)[j];
                dist += ((dval1 > dval2) ? (dval1 - dval2) : (-dval1 + dval2));
            }
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

    if ((dist2 != 0) && (dist1/dist2 < ratio_th)) 
    {
        int2 pair = 0;
        pair.s0 = gid0;
        pair.s1 = current_min;
        old = atomic_inc(counter);
        if (old < max_nb_match) matchings[old] = pair;
    }
}
