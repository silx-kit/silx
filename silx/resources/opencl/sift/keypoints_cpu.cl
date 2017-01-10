/*

	Kernels for keypoints processing

	For CPUs, one keypoint is handled by one thread
*/

typedef float4 keypoint;
#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )
#ifndef WORKGROUP_SIZE
	#define WORKGROUP_SIZE 128
#endif



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
 *
 */



__kernel void descriptor(
	__global keypoint* keypoints,
	__global unsigned char *descriptors,
	__global float* grad,
	__global float* orim,
	int octsize,
	int keypoints_start,
	//	int keypoints_end,
	__global int* keypoints_end, //passing counter value to avoid to read it each time
	int grad_width,
	int grad_height)
{

	int gid0 = get_global_id(0);
	keypoint k = keypoints[gid0];
	if (!(keypoints_start <= gid0 && gid0 < *keypoints_end && k.s1 >=0.0f))
		return;
		
	int i,j,u,v,old;
	
	__local volatile float tmp_descriptors[128];
	for (i=0; i<128; i++) tmp_descriptors[i] = 0.0f;

	float rx, cx;
	float row = k.s1/octsize, col = k.s0/octsize, angle = k.s3;
	int	irow = (int) (row + 0.5f), icol = (int) (col + 0.5f);
	float sine = sin((float) angle), cosine = cos((float) angle);
	float spacing = k.s2/octsize * 3.0f;
	int iradius = (int) ((1.414f * spacing * 2.5f) + 0.5f);

	for (i = -iradius; i <= iradius; i++) { 
		for (j = -iradius; j <= iradius; j++) { 
			 rx = ((cosine * i - sine * j) - (row - irow)) / spacing + 1.5f;
			 cx = ((sine * i + cosine * j) - (col - icol)) / spacing + 1.5f;
			if ((rx > -1.0f && rx < 4.0f && cx > -1.0f && cx < 4.0f
				 && (irow +i) >= 0  && (irow +i) < grad_height && (icol+j) >= 0 && (icol+j) < grad_width)) {
				float mag = grad[(int)(icol+j) + (int)(irow+i)*grad_width]
							 * exp(- 0.125f*((rx - 1.5f) * (rx - 1.5f) + (cx - 1.5f) * (cx - 1.5f)) );
				float ori = orim[(int)(icol+j)+(int)(irow+i)*grad_width] -  angle;
				while (ori > 2.0f*M_PI_F) ori -= 2.0f*M_PI_F;
				while (ori < 0.0f) ori += 2.0f*M_PI_F;
				int	orr, rindex, cindex, oindex;
				float	rweight, cweight;

				float oval = 4.0f*ori*M_1_PI_F; 

				int	ri = (int)((rx >= 0.0f) ? rx : rx - 1.0f),
					ci = (int)((cx >= 0.0f) ? cx : cx - 1.0f),
					oi = (int)((oval >= 0.0f) ? oval : oval - 1.0f);

				float rfrac = rx - ri,	
					cfrac = cx - ci,
					ofrac = oval - oi;
				if ((ri >= -1  &&  ri < 4  && oi >=  0  &&  oi <= 8  && rfrac >= 0.0f  &&  rfrac <= 1.0f)) {
					for (int r = 0; r < 2; r++) {
						rindex = ri + r; 
						if ((rindex >=0 && rindex < 4)) {
							float rweight = (float) (mag * (float) ((r == 0) ? 1.0f - rfrac : rfrac));

							for (int c = 0; c < 2; c++) {
								cindex = ci + c;
								if ((cindex >=0 && cindex < 4)) {
									cweight = rweight * ((c == 0) ? 1.0f - cfrac : cfrac);
									for (orr = 0; orr < 2; orr++) {
										oindex = oi + orr;
										if (oindex >= 8) {  /* Orientation wraps around at PI. */
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
		norm+=tmp_descriptors[i]*tmp_descriptors[i];
	norm = rsqrt(norm);
	for (i=0; i < 128; i++) {
		tmp_descriptors[i] *= norm;
	}


	//Threshold to 0.2 of the norm, for invariance to illumination
	bool changed = false;
	norm = 0;
	for (i = 0; i < 128; i++) {
		if (tmp_descriptors[i] > 0.2f) {
			tmp_descriptors[i] = 0.2f;
			changed = true;
		}
		norm += tmp_descriptors[i]*tmp_descriptors[i];
	}

	//if values have been changed, we have to normalize again...
	if (changed == true) {
		norm = rsqrt(norm);
		for (i=0; i < 128; i++)
			tmp_descriptors[i] *= norm;
	}

	//finally, cast to integer
	int intval;
	for (i = 0; i < 128; i++) {
		intval =  (int)(512.0 * tmp_descriptors[i]);
		descriptors[128*gid0+i]
			= (unsigned char) MIN(255, intval);
	}
}




