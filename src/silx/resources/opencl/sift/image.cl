/*
 *   Project: SIFT: An algorithm for image alignement
 *
 *   Copyright (C) 2013-2017 European Synchrotron Radiation Facility
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

/**
 *
 * Kernels for images processing
 *
 * A thread handles one keypoint -- any group size can do
 *
 *
*/


/*
 Keypoint structure : (amplitude, row, column, sigma)

 k.x == k.s0 : amplitude
 k.y == k.s1 : row
 k.z == k.s2 : column
 k.w == k.s3 : sigma

*/
typedef float4 keypoint;

#ifndef WORKGROUP_SIZE
    #define WORKGROUP_SIZE 128
#endif

/*
 Do not use __constant memory for large (usual) images
*/
#ifndef MAX_CONST_SIZE
    #define MAX_CONST_SIZE 16384
#endif

/**
 * \brief Gradient of a grayscale image
 *
 * The gradient is computed using central differences in the interior and first differences at the boundaries.
 *
 * :param igray: Pointer to global memory with the input data of the grayscale image
 * :param grad: Pointer to global memory with the output norm of the gradient
 * :param ori: Pointer to global memory with the output orientation of the gradient
 * :param width: integer number of columns of the input image
 * :param height: integer number of lines of the input image
 */



kernel void compute_gradient_orientation(
    global float* igray, // __attribute__((max_constant_size(MAX_CONST_SIZE))),
    global float *grad,
    global float *ori,
    int width,
    int height)
{

    int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);

    if (gid1 < height && gid0 < width) {

        float xgrad, ygrad;
        int pos = gid1*width+gid0;

        if (gid0 == 0)
            xgrad = 2.0f * (igray[pos+1] - igray[pos]);
        else if (gid0 == width-1)
            xgrad = 2.0f * (igray[pos] - igray[pos-1]);
        else
            xgrad = igray[pos+1] - igray[pos-1];
        if (gid1 == 0)
            ygrad = 2.0f * (igray[pos] - igray[pos + width]);
        else if (gid1 == height-1)
            ygrad = 2.0f * (igray[pos - width] - igray[pos]);
        else
            ygrad = igray[pos - width] - igray[pos + width];

        grad[pos] = sqrt((xgrad * xgrad + ygrad * ygrad));
//TODO use atan2pi and remove the division by pi later on
        ori[pos] = atan2 (-ygrad, xgrad);

      }
}





/**
 * \brief Local minimum or maximum detection in scale space
 *
 * IMPORTANT:
 *    -The output have to be Memset to (-1,-1,-1,-1)
 *    -This kernel must not be launched with s = 0 or s = nb_of_dogs (=4 for SIFT)
 *
 * :param DOGS: Pointer to global memory with ALL the coutiguously pre-allocated Differences of Gaussians
 * :param border_dist: integer, distance between inner image and borders (SIFT takes 5)
 * :param peak_thresh: float, threshold (SIFT takes 255.0 * 0.04 / 3.0)
 * :param output: Pointer to global memory output *filled with (-1,-1,-1,-1)* by default for invalid keypoints
 * :param octsize: initially 1 then twiced at each new octave
 * :param EdgeThresh0: initial upper limit of the curvatures ratio, to test if the point is on an edge
 * :param EdgeThresh: upper limit of the curvatures ratio, to test if the point is on an edge
 * :param counter: pointer to the current position in keypoints vector -- shared between threads
 * :param nb_keypoints: Maximum number of keypoints: size of the keypoints vector
 * :param scale: the scale in the DoG, i.e the index of the current DoG (this is not the std !)
 * :param total_width: integer number of columns of ALL the (contiguous) DOGs. We have total_height = height
 * :param width: integer number of columns of a DOG.
 * :param height: integer number of lines of a DOG

TODO:
-check fabs(val) outside this kernel ? It would avoid the "if"
-confirm usage of fabs instead of fabsf
-confirm the need to return -atan2() rather than atan2 ; to be coherent with python
-use OriHistThresh instead of the harcoded value = 0.8
*/


kernel void local_maxmin(
    global float* DOGS,
    global guess_keypoint* output,
    int border_dist,
    float peak_thresh,
    int octsize,
    float EdgeThresh0,
    float EdgeThresh,
    global int* counter,
    int nb_keypoints,
    int scale,
    int width,
    int height)
{

    int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);
    /*
        As the DOGs are contiguous, we have to test if (gid0,gid1) is actually in DOGs[s]
    */

    if ((gid1 < height - border_dist) && (gid0 < width - border_dist) && (gid1 >= border_dist) && (gid0 >= border_dist)) {
        int index_dog_prev = (scale-1)*(width*height);
        int index_dog =scale*(width*height);
        int index_dog_next =(scale+1)*(width*height);

        float res = 0.0f;
        float val = DOGS[index_dog + gid0 + width*gid1];

        /*
        The following condition is part of the keypoints refinement: we eliminate the low-contrast points
        NOTE: "fabsf" instead of "fabs" should be used, for "fabs" if for doubles. Used "fabs" to be coherent with python
        */
        if (fabs(val) > (0.8f * peak_thresh)) {

            int c,r,pos;
            int ismax = 0, ismin = 0;
            if (val > 0.0) ismax = 1;
            else ismin = 1;
            for (r = gid1  - 1; r <= gid1 + 1; r++) {
                for (c = gid0 - 1; c <= gid0 + 1; c++) {
                
                    pos = r*width + c;
                    if (ismax == 1) //if (val > 0.0)
                        if (DOGS[index_dog_prev+pos] > val || DOGS[index_dog+pos] > val || DOGS[index_dog_next+pos] > val) ismax = 0;
                    if (ismin == 1) //else
                        if (DOGS[index_dog_prev+pos] < val || DOGS[index_dog+pos] < val || DOGS[index_dog_next+pos] < val) ismin = 0;
                }
            }

            if (ismax == 1 || ismin == 1) res = val;

            /*
             At this point, we know if "val" is a local extremum or not
             We have to test if this value lies on an edge (keypoints refinement)
              This is done by testing the ratio of the principal curvatures, given by the product and the sum of the
               Hessian eigenvalues
            */

            pos = gid1*width+gid0;

            float H00 = DOGS[index_dog+(gid1-1)*width+gid0] - 2.0f * DOGS[index_dog+pos] + DOGS[index_dog+(gid1+1)*width+gid0],
            H11 = DOGS[index_dog+pos-1] - 2.0f * DOGS[index_dog+pos] + DOGS[index_dog+pos+1],
            H01 = ( (DOGS[index_dog+(gid1+1)*width+gid0+1]
                    - DOGS[index_dog+(gid1+1)*width+gid0-1])
                    - (DOGS[index_dog+(gid1-1)*width+gid0+1] - DOGS[index_dog+(gid1-1)*width+gid0-1])) / 4.0f;

            float det = H00 * H11 - H01 * H01, 
                trace = H00 + H11;

            /*
               If (trace^2)/det < thresh, the Keypoint is OK.
               Note that the following "EdgeThresh" seem to be the inverse of the ratio upper limit
            */

            float edthresh = (octsize <= 1 ? EdgeThresh0 : EdgeThresh);

            if (det < edthresh * trace * trace)
                res = 0.0f;

            /*
             At this stage, res != 0.0f iff the current pixel is a good keypoint
            */
            if (res != 0.0f)
            {
                int old = atomic_inc(counter);
//                keypoint k = 0.0; //no malloc, for this is a float4
//                k.s0 = val;
//                k.s1 = (float) gid1;
//                k.s2 = (float) gid0;
//                k.s3 = (float) scale;
//                guess_keypoint:  value, row, col, scale
                guess_keypoint kp;
                kp.value = val;
                kp.row = (float) gid1;
                kp.col = (float) gid0;
                kp.scale = (float) scale;
                if (old < nb_keypoints)
                    output[old]=kp;
            }//end if res

        }//end "value >thresh"
    }//end "in the inner image"
}





/**
 * \brief From the (temporary) keypoints, create a vector of interpolated keypoints
 *             (this is the last step of keypoints refinement)
 *
 *     We use the unified_keypoint type which contains:
 *         unified_keypoint.raw which is a guess_keypoint: value, row, column, scale
 *      unified_keypoint.ref which is an actual_keypoint: col, row, scale, angle
 *
 *       (-1,-1,-1) is used to flag invalid keypoints.
 *       This creates "holes" in the vector. which may be copacted afterwards
 *
 * :param DOGS: Pointer to global memory with ALL the coutiguously pre-allocated Differences of Gaussians
 * :param keypoints: Pointer to global memory with current keypoints vector. It will be modified with the interpolated points
 * :param actual_nb_keypoints: actual number of keypoints previously found, i.e previous "counter" final value
 * :param peak_thresh: we are not counting the interpolated values if below the threshold (par.PeakThresh = 255.0*0.04/3.0)
 * :param InitSigma: float "par.InitSigma" in SIFT (1.6 by default)
 * :param width: integer number of columns of the DoG
 * :param height: integer number of lines of the DoG

TODO: replace hard-coded 3.0f with par.Scales

 */


kernel void interp_keypoint(
    global float* DOGS,
    global guess_keypoint* keypoints,
    int start_keypoints,
    int end_keypoints,
    float peak_thresh,
    float InitSigma,
    int width,
    int height)
{

    //int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);

    if ((gid0 >= start_keypoints) && (gid0 < end_keypoints)) {
        guess_keypoint raw_kp = keypoints[gid0];
        int r = (int) raw_kp.row;
        int c = (int) raw_kp.col;
        int scale = (int) raw_kp.scale;
        if (r != -1)
        { //the keypoint is valid
            int index_dog_prev = (scale-1)*(width*height);
            int index_dog =scale*(width*height);
            int index_dog_next =(scale+1)*(width*height);

            //pre-allocating variables before entering into the loop
            float g0, g1, g2,
                H00, H11, H22, H01, H02, H12, H10, H20, H21,
                K00, K11, K22, K01, K02, K12, K10, K20, K21,
                solution0, solution1, solution2, det, peakval;
            int pos = r*width+c,
                loop = 1, 
                movesRemain = 5,
                newr = r, 
                newc = c;

            //this loop replaces the recursive "InterpKeyPoint"
            while (loop == 1) {

                r = newr, c = newc; //values got as parameters of InterpKeyPoint()" in sift.cpp
                pos = newr*width+newc;

                //Fill in the values of the gradient from pixel differences
                g0 = (DOGS[index_dog_next+pos] - DOGS[index_dog_prev+pos]) / 2.0f;
                g1 = (DOGS[index_dog+(newr+1)*width+newc] - DOGS[index_dog+(newr-1)*width+newc]) / 2.0f;
                g2 = (DOGS[index_dog+pos+1] - DOGS[index_dog+pos-1]) / 2.0f;

                //Fill in the values of the Hessian from pixel differences
                H00 = DOGS[index_dog_prev+pos]   - 2.0f * DOGS[index_dog+pos] + DOGS[index_dog_next+pos];
                H11 = DOGS[index_dog+(newr-1)*width+newc] - 2.0f * DOGS[index_dog+pos] + DOGS[index_dog+(newr+1)*width+newc];
                H22 = DOGS[index_dog+pos-1] - 2.0f * DOGS[index_dog+pos] + DOGS[index_dog+pos+1];

                H01 = ( (DOGS[index_dog_next+(newr+1)*width+newc] - DOGS[index_dog_next+(newr-1)*width+newc])
                        - (DOGS[index_dog_prev+(newr+1)*width+newc] - DOGS[index_dog_prev+(newr-1)*width+newc])) / 4.0f;

                H02 = ( (DOGS[index_dog_next+pos+1] - DOGS[index_dog_next+pos-1])
                        -(DOGS[index_dog_prev+pos+1] - DOGS[index_dog_prev+pos-1])) / 4.0f;

                H12 = ( (DOGS[index_dog+(newr+1)*width+newc+1] - DOGS[index_dog+(newr+1)*width+newc-1])
                        - (DOGS[index_dog+(newr-1)*width+newc+1] - DOGS[index_dog+(newr-1)*width+newc-1])) / 4.0f;

                H10 = H01; H20 = H02; H21 = H12;


                //inversion of the Hessian    : det*K = H^(-1)

                det = -(H02*H11*H20) + H01*H12*H20 + H02*H10*H21 - H00*H12*H21 - H01*H10*H22 + H00*H11*H22;

                K00 = H11*H22 - H12*H21;
                K01 = H02*H21 - H01*H22;
                K02 = H01*H12 - H02*H11;
                K10 = H12*H20 - H10*H22;
                K11 = H00*H22 - H02*H20;
                K12 = H02*H10 - H00*H12;
                K20 = H10*H21 - H11*H20;
                K21 = H01*H20 - H00*H21;
                K22 = H00*H11 - H01*H10;


                /*
                    x = -H^(-1)*g
                 As the Taylor Serie is calcualted around the current keypoint,
                 the position of the true extremum x_opt is exactly the "offset" between x and x_opt ("x" is the origin)
                */
                solution0 = -(g0*K00 + g1*K01 + g2*K02)/det; //"offset" in sigma
                solution1 = -(g0*K10 + g1*K11 + g2*K12)/det; //"offset" in r
                solution2 = -(g0*K20 + g1*K21 + g2*K22)/det; //"offset" in c

                //interpolated DoG magnitude at this peak
                peakval = DOGS[index_dog+pos] + 0.5f * (solution0*g0+solution1*g1+solution2*g2);


            /* Move to an adjacent (row,col) location if quadratic interpolation is larger than 0.6 units in some direction.                 The movesRemain counter allows only a fixed number of moves to prevent possibility of infinite loops.
            */

                if (solution1 > 0.6f && newr < height - 3)
                    newr++; //if the extremum is too far (along "r" here), we get closer if we can
                else if (solution1 < -0.6f && newr > 3)
                    newr--;
                if (solution2 > 0.6f && newc < width - 3)
                    newc++;
                else if (solution2 < -0.6f && newc > 3)
                    newc--;

                /*
                    Loop test
                */
                if (movesRemain > 0  &&  (newr != r || newc != c))
                    movesRemain--;
                else
                    loop = 0;

            }//end of the "keypoints interpolation" big loop


            /* Do not create a keypoint if interpolation still remains far outside expected limits,
                or if magnitude of peak value is below threshold (i.e., contrast is too low).

                guess_keypoint:  value, row, col, scale
            */
            guess_keypoint ref_kp;
            if (fabs(solution0) <= 1.5f && fabs(solution1) <= 1.5f && fabs(solution2) <= 1.5f && fabs(peakval) >= peak_thresh)
            { // keypoint properly interpolated
                ref_kp.value = peakval;
                ref_kp.row = r + solution1;
                ref_kp.col = c + solution2;
                ref_kp.scale = InitSigma * pow(2.0f, (((float) scale) + solution0) / 3.0f); //3.0 is "par.Scales"
            }// endif keypoint properly interpolated
            else
            { //the keypoint was not correctly interpolated : marked as bad
                ref_kp.value = -1.0f;
                ref_kp.row = -1.0f;
                ref_kp.col = -1.0f;
                ref_kp.scale = -1.0f;
            } //end bad kp

            keypoints[gid0] = ref_kp;

        /*
            Better return here and compute histogram in another kernel
        */
        }

    }
}



