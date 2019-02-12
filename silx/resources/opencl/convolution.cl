#define MAX_CONST_SIZE 16384


///
/// Horizontal convolution (along fast dim)
///

__kernel void horizontal_convolution(
    const __global float * input,  // input array
    __global float * output, // output array
    __global float * filter,// __attribute__((max_constant_size(MAX_CONST_SIZE))), // filter coefficients
    int hlen,       // filter size
    int IMAGE_W,
    int IMAGE_H,
    int IMAGE_Z
)
{
    int gidz = (int) get_global_id(2); // slow dim
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0); // fast dim

    if (gidy < IMAGE_H && gidx < IMAGE_W && gidz < IMAGE_Z) {

        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        int jx1 = c - gidx;
        int jx2 = IMAGE_W - 1 - gidx + c;
        float sum = 0.0f;

        // Convolution with boundaries extension
        for (int jx = 0; jx <= hR+hL; jx++) {
            int idx_x = gidx - c + jx;
            if (jx < jx1) idx_x = jx1-jx-1;
            if (jx > jx2) idx_x = IMAGE_W - (jx-jx2);

            sum += input[(gidz*IMAGE_H + gidy)*IMAGE_W + idx_x] * filter[hlen-1 - jx];
        }
        output[(gidz*IMAGE_H + gidy)*IMAGE_W + gidx] =  sum;
    }
}


///
/// Vertical convolution
///

__kernel void vertical_convolution(
    const __global float * input,  // input array
    __global float * output, // output array
    __global float * filter,// __attribute__((max_constant_size(MAX_CONST_SIZE))), // filter coefficients
    int hlen,       // filter size
    int IMAGE_W,
    int IMAGE_H,
    int IMAGE_Z
)
{
    int gidz = (int) get_global_id(2); // slow dim
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0); // fast dim

    if (gidy < IMAGE_H && gidx < IMAGE_W && gidz < IMAGE_Z) {

        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        int jy1 = c - gidy;
        int jy2 = IMAGE_H - 1 - gidy + c;
        float sum = 0.0f;

        // Convolution with boundaries extension
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + jy;
            if (jy < jy1) idx_y = jy1-jy-1;
            if (jy > jy2) idx_y = IMAGE_H - (jy-jy2);

            sum += input[(gidz*IMAGE_H + idx_y)*IMAGE_W + gidx] * filter[hlen-1 - jy];
        }
        output[(gidz*IMAGE_H + gidy)*IMAGE_W + gidx] =  sum;
    }
}




///
/// Depth convolution (along fast dim)
///

__kernel void depth_convolution(
    const __global float * input,  // input array
    __global float * output, // output array
    __global float * filter,// __attribute__((max_constant_size(MAX_CONST_SIZE))), // filter coefficients
    int hlen,       // filter size
    int IMAGE_W,
    int IMAGE_H,
    int IMAGE_Z
)
{
    int gidz = (int) get_global_id(2); // slow dim
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0); // fast dim

    if (gidy < IMAGE_H && gidx < IMAGE_W && gidz < IMAGE_Z) {

        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        int jz1 = c - gidz;
        int jz2 = IMAGE_Z - 1 - gidz + c;
        float sum = 0.0f;

        // Convolution with boundaries extension
        for (int jz = 0; jz <= hR+hL; jz++) {
            int idx_z = gidz - c + jz;
            if (jz < jz1) idx_z = jz1-jz-1;
            if (jz > jz2) idx_z = IMAGE_Z - (jz-jz2);

            sum += input[(idx_z*IMAGE_H + gidy)*IMAGE_W + gidx] * filter[hlen-1 - jz];
        }
        output[(gidz*IMAGE_H + gidy)*IMAGE_W + gidx] =  sum;
    }
}



