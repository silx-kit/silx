/*

	Kernels for keypoints processing

	For CPUs, one keypoint is handled by one thread
*/

typedef float4 keypoint;
#define MIN(i,j) ( (i)<(j) ? (i):(j) )
#define MAX(i,j) ( (i)<(j) ? (j):(i) )
#ifndef WORKGROUP_SIZE
	#define WORKGROUP_SIZE 1
#endif



/**
 * \brief Assign an orientation to the keypoints.  This is done by creating a Gaussian weighted histogram
 *   of the gradient directions in the region.  The histogram is smoothed and the largest peak selected.
 *    The results are in the range of -PI to PI.
 *
 * Warning:
 * 			-At this stage, a keypoint is: (peak,r,c,sigma)
 * 			 After this function, it will be (c,r,sigma,angle)
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


__kernel void orientation_assignment(
	__global keypoint* keypoints,
	__global float* grad,
	__global float* ori,
	__global int* counter,
	int octsize,
	float OriSigma, //WARNING: (1.5), it is not "InitSigma (=1.6)"
	int nb_keypoints,
	int keypoints_start,
	int keypoints_end,
	int grad_width,
	int grad_height)
{
	int gid0 = get_global_id(0);
	keypoint k = keypoints[gid0];
	if (!(keypoints_start <= gid0 && gid0 < keypoints_end && k.s1 >=0.0f ))
		return;
	int	bin, prev=0, next=0;
	int i,j,r,c;
	int old;
	float distsq, gval, angle, interp=0.0;
	float hist_prev,hist_curr,hist_next;
	float hist[36];
	//memset
	for (i=0; i<36; i++) hist[i] = 0.0f;

	int	row = (int) (k.s1 + 0.5),
		col = (int) (k.s2 + 0.5);

	float sigma = OriSigma * k.s3;
	int	radius = (int) (sigma * 3.0);
	int rmin = MAX(0,row - radius);
	int cmin = MAX(0,col - radius);
	int rmax = MIN(row + radius,grad_height - 2);
	int cmax = MIN(col + radius,grad_width - 2);
	
	for (r = rmin; r <= rmax; r++) {
		for (c = cmin; c <= cmax; c++) {
			gval = grad[r*grad_width+c];
			
			float dif = (r - k.s1);	distsq = dif*dif;
			dif = (c - k.s2);	distsq += dif*dif;
			
			//distsq = (r-k.s1)*(r-k.s1) + (c-k.s2)*(c-k.s2);

			if (gval > 0.0f  &&  distsq < ((float) (radius*radius)) + 0.5f) {
				angle = ori[r*grad_width+c];
				bin = (int) (36.0f * (angle + M_PI_F + 0.001f) / (2.0f * M_PI_F)); //why this offset ?
				if (bin >= 0 && bin <= 36) {
					bin = MIN(bin, 35);
					hist[bin] += exp(- distsq / (2.0f*sigma*sigma)) * gval;
				}
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
	if (maxval < 0.0f) {
		hist_prev = -hist_prev;
		maxval = -maxval;
		hist_next = -hist_next;
	}
	interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * maxval + hist_next);
	angle = 2.0f * M_PI_F * (argmax + 0.5f + interp) / 36.0f - M_PI_F;


	k.s0 = k.s2 *octsize; //c
	k.s1 = k.s1 *octsize; //r
	k.s2 = k.s3 *octsize; //sigma
	k.s3 = angle; 		  //angle
	keypoints[gid0] = k;
	
	/*
		An orientation is now assigned to our current keypoint.
		We can create new keypoints of same (x,y,sigma) but a different angle.
		For every local peak in histogram, every peak of value >= 80% of maxval generates a new keypoint
	*/
	
	for (i=0; i < 36; i++) {
		int prev = (i == 0 ? 35 : i - 1);
		int next = (i == 35 ? 0 : i + 1);
		float hist_prev = hist[prev];
		float hist_curr = hist[i];
		float hist_next = hist[next];
		if (hist_curr > hist_prev  &&  hist_curr > hist_next && hist_curr >= 0.8f * maxval && i != argmax) {
			if (hist_curr < 0.0f) {
				hist_prev = -hist_prev;
				hist_curr = -hist_curr;
				hist_next = -hist_next;
			}
			float interp = 0.5f * (hist_prev - hist_next) / (hist_prev - 2.0f * hist_curr + hist_next);
			float angle = 2.0f * M_PI_F * (i + 0.5f + interp) /36.0 - M_PI_F;
			if (angle >= -M_PI_F && angle <= M_PI_F) {
				k.s3 = angle;
				old  = atomic_inc(counter);
				if (old < nb_keypoints) keypoints[old] = k;
			}
		} //end "val >= 80%*maxval"
	}
}
