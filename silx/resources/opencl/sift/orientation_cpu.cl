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

    Kernels for keypoints processing

    For CPUs, one keypoint is handled by one thread
*/



/**
 * \brief Assign an orientation to the keypoints.  This is done by creating a Gaussian weighted histogram
 *   of the gradient directions in the region.  The histogram is smoothed and the largest peak selected.
 *    The results are in the range of -PI to PI.
 *
 * Warning:
 *             -At this stage, a keypoint is: (peak,r,c,sigma)
 *              After this function, it will be (c,r,sigma,angle)
 *
 *  Workgroup size: (1,)
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


kernel void orientation_cpu(
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
                            int grad_height)
{
    int gid0 = (int) get_global_id(0);
    guess_keypoint raw_kp = keypoints[gid0].raw;
    if (!(keypoints_start <= gid0 && gid0 < keypoints_end && raw_kp.row >=0.0f))
    {
        return;
    }
    int bin, prev=0, next=0;
    int i,j,r,c;
    int old;
    float distsq, gval, angle, interp=0.0;
    float hist_prev, hist_next;
    float hist[36];
    //memset
    for (i=0; i<36; i++) 
    {
        hist[i] = 0.0f;
    }

    int row = (int) (raw_kp.row + 0.5f),
        col = (int) (raw_kp.col + 0.5f);

    float sigma = OriSigma * raw_kp.scale;
    int    radius = (int) (sigma * 3.0f);
    int rmin = max(0,row - radius);
    int cmin = max(0,col - radius);
    int rmax = min(row + radius,grad_height - 2);
    int cmax = min(col + radius,grad_width - 2);

    for (r = rmin; r <= rmax; r++)
    {
        for (c = cmin; c <= cmax; c++)
        {
            gval = grad[r*grad_width+c];

            //distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);
            float dif;
            dif = (r - raw_kp.row);
            distsq = dif*dif;
            dif = (c - raw_kp.col);
            distsq += dif*dif;

            if (gval > 0.0f  &&  distsq < ((float) (radius*radius)) + 0.5f)
            {
                // Ori is in range of -PI to PI.
                angle = ori[r*grad_width+c];
                bin = (int) (18.0f * (angle + M_PI_F) / (M_PI_F));
                if (bin<0)
                    bin+=36;
                if (bin>35)
                    bin-=36;
                hist[bin] += exp(- distsq / (2.0f*sigma*sigma)) * gval;

            }
        }
    }



    /*
        Apply smoothing 6 times for accurate Gaussian approximation
    */

    for (j = 0; j < 6; j++) {
        float prev, temp; //it is CRUCIAL to re-define "prev" here, for the line below... otherwise, it won't work
        prev = hist[35];
        for (i = 0; i < 36; i++) {
            temp = hist[i];
            hist[i] = ( prev + hist[i] + hist[(i + 1 == 36) ? 0 : i + 1] ) / 3.0;
            prev = temp;
        }
    }


    /* Find maximum value in histogram */

    float maxval = 0.0f;
    int argmax = 0;
    for (i=0; i<36; i++) {
        if (maxval < hist[i]) {
            maxval = hist[i];
            argmax = i;
        }
    }

/*
    This maximum value in the histogram is defined as the orientation of our current keypoint
*/
    prev = (argmax == 0 ? 35 : argmax - 1);
    next = (argmax == 35 ? 0 : argmax + 1);
    hist_prev = hist[prev];
    hist_next = hist[next];
    if (maxval < 0.0f)
    {
        hist_prev = -hist_prev;
        maxval = -maxval;
        hist_next = -hist_next;
    }
    interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * maxval + hist_next);
    angle = 2.0f * (argmax + 0.5f + interp) / 36.0f;

    actual_keypoint ref_kp;
    ref_kp.col = raw_kp.col * octsize;           //c
    ref_kp.row = raw_kp.row * octsize;           //r
    ref_kp.scale = raw_kp.scale * octsize;       //sigma
    ref_kp.angle = (angle - 1.0f) * M_PI_F;      //angle
    keypoints[gid0].ref = ref_kp;


    /*
        An orientation is now assigned to our current keypoint.
        We can create new keypoints of same (x,y,sigma) but a different angle.
        For every local peak in histogram, every peak of value >= 80% of maxval generates a new keypoint
    */

    for (i=0; i < 36; i++)
    {
        int prev = (i == 0 ? 35 : i - 1);
        int next = (i == 35 ? 0 : i + 1);
        float hist_prev = hist[prev];
        float hist_curr = hist[i];
        float hist_next = hist[next];
        if (hist_curr > hist_prev  &&  hist_curr > hist_next && hist_curr >= 0.8f * maxval && i != argmax)
        {
            if (hist_curr < 0.0f)
            {
                hist_prev = -hist_prev;
                hist_curr = -hist_curr;
                hist_next = -hist_next;
            }
            float interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * hist_curr + hist_next);

            float angle = (i + 0.5f + interp) / 18.0f;
            if (angle<0.0f)
                angle+=2.0f;
            else if (angle>2.0f)
                angle-=2.0f;
            ref_kp.angle = (angle - 1.0f) * M_PI_F;
            old  = atomic_inc(counter);
            if (old < nb_keypoints)
            {
                keypoints[old].ref = ref_kp;
            }
        } //end "val >= 80%*maxval"
    }
}
