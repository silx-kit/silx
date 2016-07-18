
typedef float4 keypoint;

/**
 * \brief Linear combination of two matrices
 *
 * @param u: Pointer to global memory with the input data of the first matrix
 * @param a: float scalar which multiplies the first matrix
 * @param v: Pointer to global memory with the input data of the second matrix
 * @param b: float scalar which multiplies the second matrix
 * @param w: Pointer to global memory with the output data
 * @param width: integer, number of columns the matrices
 * @param height: integer, number of lines of the matrices
 *
 * Nota: updated to have coalesced access on dim[0]
 */

__kernel void combine(
	__global float *u,
	float a,
	__global float *v,
	float b,
	__global float *w,
	int dog,
	int width,
	int height)
{

	int gid1 = (int) get_global_id(1);
	int gid0 = (int) get_global_id(0);

	if (gid0 < width && gid1 < height) {
		int index = gid0 + width * gid1;
		int index_dog = dog * width * height +  index;
		w[index_dog] = a * u[index] + b * v[index];
	}
}



/**
 * \brief Deletes the (-1,-1,-1,-1) in order to get a more "compact" keypoints vector
 		Also arranges the keypoints coordinates in the SIFT order : (x:col,y:row,sigma,angle)
 *		(initially we had (peak,r,c,sigma), but at this stage peak is not useful anymore)
 *
 *
 * @param keypoints: Pointer to global memory with the keypoints
 * @param output: Pointer to global memory with the output
 * @param counter: Pointer to global memory with the shared counter in the output
 * @param start_keypoint: start compaction at this index. counter should be equal to start at the begining.
 * @param end_keypoint: index of last keypoints
 *
 */



__kernel void compact(
	__global keypoint* keypoints,
	__global keypoint* output,
	__global int* counter,
	int start_keypoint,
	int end_keypoint)
{

	int gid0 = (int) get_global_id(0);
	if (gid0 < start_keypoint){
		output[gid0] = keypoints[gid0];
	}
	else if (gid0 < end_keypoint) {

		keypoint k = keypoints[gid0];

		if (k.s1 != -1) { //Coordinates are never negative

			/*k.s0 = (float) k.s2; //col
			k.s2 = k.s3; //sigma
			k.s3 = 0.0; //angle
			*/
			int old = atomic_inc(counter);
			if (old < end_keypoint) output[old] = k;

		}
	}
}















