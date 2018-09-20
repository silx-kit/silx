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

/*

    Kernels for keypoints orientation processing

    A *group of threads* handles one keypoint, for additional information is required in the keypoint neighborhood

    WARNING: local workgroup size must be at least 128 for orientation_assignment
    
    Workgroup Size: (128,)

    For descriptors (so far) :
    we use shared memory to store temporary 128-histogram (1 per keypoint)
      therefore, we need 128*N*4 bytes for N keypoints. We have
      -- 16 KB per multiprocessor for <=1.3 compute capability (GTX <= 295), that allows to process N<=30 keypoints per thread
      -- 48 KB per multiprocessor for >=2.x compute capability (GTX >= 465, Quadro 4000), that allows to process N<=95 keypoints per thread

*/

#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )



/**
 * \brief Assign an orientation to the keypoints.  This is done by creating a Gaussian weighted histogram
 *   of the gradient directions in the region.  The histogram is smoothed and the largest peak selected.
 *    The results are in the range of -PI to PI.
 *
 * Warning:
 *             -At this stage, a keypoint is: (peak,r,c,sigma)
              After this function, it will be (c,r,sigma,angle)
 *
 * :param keypoints: Pointer to global memory with current keypoints vector.
 * :param grad: Pointer to global memory with gradient norm previously calculated
 * :param ori: Pointer to global memory with gradient orientation previously calculated
 * :param counter: Pointer to global memory with actual number of keypoints previously found
 * :param hist: Pointer to shared memory with histogram (36 values per thread)
 * :param octsize: initially 1 then twiced at each octave
 * :param OriSigma : a SIFT parameter, default is 1.5. Warning : it is not "InitSigma".
 * :param nb_keypoints : maximum number of keypoints
 * :param grad_width: integer number of columns of the gradient
 * :param grad_height: integer num of lines of the gradient
 */


/*

par.OriBins = 36
par.OriHistThresh = 0.8;
-replace "36" by an external paramater ?
-replace "0.8" by an external parameter ?

TODO:
-Memory optimization
    --Use less registers (re-use, calculation instead of assignation)
    --Use local memory for float histogram[36]
-Speed-up
    --Less access to global memory (raw_kp.row is OK because this is a register)
    --leave the loops as soon as possible
    --Avoid divisions


*/

kernel void orientation_gpu(
                            global unified_keypoint* keypoints,
                            global float* grad,
                            global float* ori,
                            global int* counter,
                            int octsize,
                            float OriSigma, //WARNING: (1.5), it is not "InitSigma (=1.6)"
                            int nb_keypoints,
                            int keypoints_start,
                            int keypoints_end,
                            int grad_width,
                            int grad_height,
                            local float* hist, //size 36
                            local volatile float* hist2, //size 128
                            local volatile int* pos) //size 128
{
    int lid0 = (int) get_local_id(0);
    int groupid = (int) get_group_id(0);

    if ((groupid< keypoints_start) || (groupid >= keypoints_end))
    {   //    Process only valid points
        return;
    }
    guess_keypoint raw_kp = keypoints[groupid].raw;
    if (raw_kp.row < 0.0f )
    {
        return;
    }

    int bin, prev=0, next=0;
    int old;
    float distsq, gval, angle, interp=0.0;
    float hist_prev, hist_curr, hist_next;
    float ONE_3 = 1.0f / 3.0f;
    float ONE_18 = 1.0f / 18.0f;
    //memset for "pos" and "hist2"
    pos[lid0] = -1;
    hist2[lid0] = 0.0f;
    if (lid0 <36)
    {
        hist[lid0] = 0.0f;
    }

    int row = (int) (raw_kp.row + 0.5f),
        col = (int) (raw_kp.col + 0.5f);

    /* Look at pixels within 3 sigma around the point and sum their
      Gaussian weighted gradient magnitudes into the histogram. */

    float sigma = OriSigma * raw_kp.scale;
    int radius = (int) (sigma * 3.0f);
    int rmin = MAX(0,row - radius);
    int cmin = MAX(0,col - radius);
    int rmax = MIN(row + radius,grad_height - 2);
    int cmax = MIN(col + radius,grad_width - 2);
    int i,j,r,c;
    for (r = rmin; r <= rmax; r++)
    {
        //memset for "pos" and "hist2"
        pos[lid0] = -1;
        hist2[lid0] = 0.0f;

        c = cmin + lid0;
        pos[lid0] = -1;
        hist2[lid0] = 0.0f; //do not forget to memset before each re-use...
        if (c <= cmax)
        {
            gval = grad[r*grad_width+c];
            //distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);
            float dif;
            dif = (r - raw_kp.row);
            distsq = dif*dif;
            dif = (c - raw_kp.col);
            distsq += dif*dif;

            if (gval > 0.0f  &&  distsq < ((radius*radius) + 0.5f))
            {
                // Ori is in range of -PI to PI.
                angle = ori[r*grad_width+c];
                bin = (int) (18.0f * (angle + M_PI_F) *  M_1_PI_F);
                if (bin<0) bin+=36;
                if (bin>35) bin-=36;
                hist2[lid0] = exp(- distsq / (2.0f*sigma*sigma)) * gval;
                pos[lid0] = bin;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //We are missing atomic operations on floats in OpenCL...
        if (lid0 == 0)
        { //this has to be done here ! if not, pos[] is erased !
            for (i=0; i < 128; i++)
                if (pos[i] != -1) hist[pos[i]] += hist2[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

//        Apply smoothing 6 times for accurate Gaussian approximation

    for (j=0; j<6; j++)
    {
        if (lid0 == 0)
        {
            hist2[0] = hist[0]; //save unmodified hist
            hist[0] = (hist[35] + hist[0] + hist[1]) * ONE_3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (0 < lid0 && lid0 < 35)
        {
            hist2[lid0]=hist[lid0];
            hist[lid0] = (hist2[lid0-1] + hist[lid0] + hist[lid0+1]) * ONE_3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid0 == 35)
        {
            hist[35] = (hist2[34] + hist[35] + hist[0]) * ONE_3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    hist2[lid0] = 0.0f;


    /* Find maximum value in histogram */

    float maxval = 0.0f;
    int argmax = 0;
    //memset for "pos" and "hist2"
    pos[lid0] = -1;
    hist2[lid0] = 0.0f;

    //    Parallel reduction
    if (lid0<32)
    {
        if (lid0+32<36)
        {
            if (hist[lid0]>hist[lid0+32]){
                hist2[lid0] = hist[lid0];
                pos[lid0] = lid0;
            }
            else
            {
                hist2[lid0] = hist[lid0+32];
                pos[lid0] = lid0+32;
            }
        }
        else
        {
            hist2[lid0] = hist[lid0];
            pos[lid0] = lid0;
        }
    } //now we have hist2[0..32[ that takes [32..36[ into account
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0<16)
    {
        if (hist2[lid0+16]>hist2[lid0])
        {
            hist2[lid0] = hist2[lid0+16];
            pos[lid0] = pos[lid0+16];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0<8)
    {
        if (hist2[lid0+ 8]>hist2[lid0])
        {
            hist2[lid0] = hist2[lid0+ 8];
            pos[lid0] = pos[lid0+ 8];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0<04)
    {
        if (hist2[lid0+04]>hist2[lid0])
        {
            hist2[lid0] = hist2[lid0+04];
            pos[lid0] = pos[lid0+04];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0<02)
    {
        if (hist2[lid0+02]>hist2[lid0])
        {
            hist2[lid0] = hist2[lid0+02];
            pos[lid0] = pos[lid0+02];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid0==0)
    {
        if (hist2[1]>hist2[0])
        {
            hist2[0]=hist2[1];
            pos[0] = pos[1];
        }
        argmax = pos[0];
        maxval = hist2[0];

        /*
        This maximum value in the histogram is defined as the orientation of our current keypoint
        NOTE: a "true" keypoint has his coordinates multiplied by "octsize" (cf. SIFT)
        */

        prev = (argmax == 0 ? 35 : argmax - 1);
        next = (argmax == 35 ? 0 : argmax + 1);
        hist_prev = hist[prev];
        hist_next = hist[next];

        /* //values are positive...
        if (maxval < 0.0f) {
            hist_prev = -hist_prev; //do not directly use hist[prev] which is shared
            maxval = -maxval;
            hist_next = -hist_next;
        }
        */
        interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * maxval + hist_next);
        angle = (argmax + 0.5f + interp) * ONE_18;
        if (angle<0.0f)
            angle+=2.0f;
        else if (angle>2.0f)
            angle-=2.0f;

        actual_keypoint ref_kp;
        ref_kp.col = raw_kp.col * octsize;           //c
        ref_kp.row = raw_kp.row * octsize;           //r
        ref_kp.scale = raw_kp.scale * octsize;       //sigma
        ref_kp.angle = (angle - 1.0f) * M_PI_F;      //angle
        keypoints[groupid].ref = ref_kp;
//        use local memory to communicate with other threads
        pos[0] = argmax;
        hist2[0] = maxval;
        hist2[1] = ref_kp.col;
        hist2[2] = ref_kp.row;
        hist2[3] = ref_kp.scale;
        hist2[4] = ref_kp.angle;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //broadcast these values to all threads
    actual_keypoint new_kp;
    argmax = pos[0];
    maxval = hist2[0];
    new_kp.col = hist2[1];
    new_kp.row = hist2[2];
    new_kp.scale = hist2[3];
    new_kp.angle = hist2[4];

    /*
        An orientation is now assigned to our current keypoint.
        We can create new keypoints of same (x,y,sigma) but a different angle.
        For every local peak in histogram, every peak of value >= 80% of maxval generates a new keypoint
    */
    if (lid0 < 36 && lid0 != argmax)
    {
        i = lid0;
        prev = (i == 0 ? 35 : i - 1);
        next = (i == 35 ? 0 : i + 1);
        hist_prev = hist[prev];
        hist_curr = hist[i];
        hist_next = hist[next];

        if (hist_curr > hist_prev  &&  hist_curr > hist_next && hist_curr >= 0.8f * maxval)
        {
        /* //all values are positive...
            if (hist_curr < 0.0f) {
                hist_prev = -hist_prev;
                hist_curr = -hist_curr;
                hist_next = -hist_next;
            }
        */
            interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * hist_curr + hist_next);
            angle = (i + 0.5f + interp) * ONE_18;
            if (angle<0.0f)
                angle+=2.0f;
            else if (angle>2.0f)
                angle-=2.0f;
            new_kp.angle = (angle - 1.0f)*M_PI_F;
            old  = atomic_inc(counter);
            if (old < nb_keypoints)
                keypoints[old].ref = new_kp;

        } //end "val >= 80%*maxval"
    }
}
