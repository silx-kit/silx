/*
 *   Project: SIFT: An algorithm for image alignement
 *   keypoints_cpu
 *
 *   Copyright (C) 2013-2018 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *
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

    Mind to include sift.cl
*/

/* Deprecated:
typedef float4 keypoint;

Defined in sift.cl
typedef struct actual_keypoint
{
    float col, row, scale, angle;
} actual_keypoint;
*/

/*
**
 * \brief Compute a SIFT descriptor for each keypoint.
 *
 * :param keypoints: Pointer to global memory with current keypoints vector
 * :param descriptor: Pointer to global memory with the output SIFT descriptor, cast to uint8
 * //:param tmp_descriptor: Pointer to shared memory with temporary computed float descriptors
 * :param grad: Pointer to global memory with gradient norm previously calculated
 * :param oril: Pointer to global memory with gradient orientation previously calculated
 * :param keypoints_start : index start for keypoints
 * :param keypoints_end: end index for keypoints
 * :param grad_width: integer number of columns of the gradient
 * :param grad_height: integer num of lines of the gradient
 *
 * Group size: It is important to have a workgroup size of 1
 * There is some shared memory used but it is to save registers.
 *
 *
 */

kernel void descriptor_cpu(
                            global actual_keypoint* keypoints,
                            global unsigned char *descriptors,
                            global float* grad,
                            global float* orim,
                            int octsize,
                            int keypoints_start,
                            //    int keypoints_end,
                            global int* keypoints_end, //passing counter value to avoid to read it each time
                            int grad_width,
                            int grad_height)
{
    int groupid = (int) get_group_id(0);
    if ((groupid < keypoints_start) || (groupid >= *keypoints_end))
    {
        return;
    }
    actual_keypoint kp = keypoints[groupid];
    if ((kp.row <= 0.0f) || (kp.col <= 0.0f))
    {
        return;
    }

    int i, j;
    
    local volatile float tmp_descriptors[128];
    for (i=0; i<128; i++)
    {
        tmp_descriptors[i] = 0.0f;
    }

    float rx, cx,
          col = kp.col/octsize,
          row = kp.row/octsize,
          angle = kp.angle,
          sine = sin((float) angle),
          cosine = cos((float) angle),
          spacing = kp.scale/octsize * 3.0f;
    int    irow = (int) (row + 0.5f),
        icol = (int) (col + 0.5f),
        iradius = (int) ((1.414f * spacing * 2.5f) + 0.5f);

    for (i = -iradius; i <= iradius; i++)
    {
        for (j = -iradius; j <= iradius; j++)
        {
             rx = ((cosine * i - sine * j) - (row - irow)) / spacing + 1.5f;
             cx = ((sine * i + cosine * j) - (col - icol)) / spacing + 1.5f;
            if ((rx > -1.0f && rx < 4.0f && cx > -1.0f && cx < 4.0f
                 && (irow +i) >= 0  && (irow +i) < grad_height && (icol+j) >= 0 && (icol+j) < grad_width))
            {
                float mag = grad[(int)(icol+j) + (int)(irow+i)*grad_width]
                             * exp(- 0.125f*((rx - 1.5f) * (rx - 1.5f) + (cx - 1.5f) * (cx - 1.5f)) );
                float ori = orim[(int)(icol+j)+(int)(irow+i)*grad_width] -  angle;
                while (ori > 2.0f*M_PI_F) ori -= 2.0f*M_PI_F;
                while (ori < 0.0f) ori += 2.0f*M_PI_F;
                int    orr, rindex, cindex, oindex;
                float cweight;

                float oval = 4.0f*ori*M_1_PI_F; 

                int    ri = (int)((rx >= 0.0f) ? rx : rx - 1.0f),
                    ci = (int)((cx >= 0.0f) ? cx : cx - 1.0f),
                    oi = (int)((oval >= 0.0f) ? oval : oval - 1.0f);

                float rfrac = rx - ri,    
                    cfrac = cx - ci,
                    ofrac = oval - oi;
                if ((ri >= -1  &&  ri < 4  && oi >=  0  &&  oi <= 8  && rfrac >= 0.0f  &&  rfrac <= 1.0f))
                {
                    for (int r = 0; r < 2; r++)
                    {
                        rindex = ri + r; 
                        if ((rindex >=0 && rindex < 4)) {
                            float rweight = (float) (mag * (float) ((r == 0) ? 1.0f - rfrac : rfrac));

                            for (int c = 0; c < 2; c++)
                            {
                                cindex = ci + c;
                                if ((cindex >=0 && cindex < 4))
                                {
                                    cweight = rweight * ((c == 0) ? 1.0f - cfrac : cfrac);
                                    for (orr = 0; orr < 2; orr++)
                                    {
                                        oindex = oi + orr;
                                        if (oindex >= 8)
                                        {  /* Orientation wraps around at PI. */
                                            oindex = 0;
                                        }
                                        tmp_descriptors[(rindex*4 + cindex)*8+oindex] 
                                            += cweight * ((orr == 0) ? 1.0f - ofrac : ofrac); //1.0f;
                                    } //end "for orr"
                                } //end "valid cindex"
                            } //end "for c"
                        } //end "valid rindex"
                    } //end "for r"
                }
            } //end "sample in boundaries"
        }
    } //end "i loop"


    /*
        At this point, we have a descriptor associated with our keypoint.
        We have to normalize it, then cast to 1-byte integer
    */

    // Normalization

    float norm = 0;
    for (i = 0; i < 128; i++) 
    {
        norm+=tmp_descriptors[i]*tmp_descriptors[i];
    }
    norm = rsqrt(norm);
    for (i=0; i < 128; i++) 
    {
        tmp_descriptors[i] *= norm;
    }


    //Threshold to 0.2 of the norm, for invariance to illumination
    int changed = 0;
    norm = 0;
    for (i = 0; i < 128; i++) {
        if (tmp_descriptors[i] > 0.2f)
        {
            tmp_descriptors[i] = 0.2f;
            changed = 1;
        }
        norm += tmp_descriptors[i]*tmp_descriptors[i];
    }

    //if values have been changed, we have to normalize again...
    if (changed)
    {
        norm = rsqrt(norm);
        for (i=0; i < 128; i++)
        {
            tmp_descriptors[i] *= norm;
        }
    }

    //finally, cast to integer
    int intval;
    for (i = 0; i < 128; i++)
    {
        intval =  (int)(512.0f * tmp_descriptors[i]);
        descriptors[128*groupid+i] = (uchar) min(255, intval);
    }
}
