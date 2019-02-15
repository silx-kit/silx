#define MAX_CONST_SIZE 16384



/******************************************************************************/
/**************************** Macros ******************************************/
/******************************************************************************/

// Get the center index of the filter,
// and the "half-Left" and "half-Right" lengths.
// In the case of an even-sized filter, the center is shifted to the left.
#define GET_CENTER_HL(hlen){\
        if (hlen & 1) {\
            c = hlen/2;\
            hL = c;\
            hR = c;\
        }\
        else {\
            c = hlen/2 - 1;\
            hL = c;\
            hR = c+1;\
        }\
}\


// Get the lower and upper bound for the summation part of convolution.
//~ #define GET_SUM_BOUNDS_X int jx1 = c - gidx, jx2 = Nx - 1 - gidx + c;
//~ #define GET_SUM_BOUNDS_Y int jy1 = c - gidy, jy2 = Ny - 1 - gidy + c;
//~ #define GET_SUM_BOUNDS_Z int jz1 = c - gidz, jz2 = Nz - 1 - gidz + c;

// Get the moving convolution index for periodic boundary extension.
//~ #define CONV_PERIODIC_IDX_X int idx_x = gidx - c + jx; if (jx < jx1) idx_x = jx1-jx-1; if (jx > jx2) idx_x = Nx - (jx-jx2);
//~ #define CONV_PERIODIC_IDX_Y int idx_y = gidy - c + jy; if (jy < jy1) idx_y = jy1-jy-1; if (jy > jy2) idx_y = Ny - (jy-jy2);
//~ #define CONV_PERIODIC_IDX_Z int idx_z = gidz - c + jz; if (jz < jz1) idx_z = jz1-jz-1; if (jz > jz2) idx_z = Nz - (jz-jz2);
#define CONV_PERIODIC_IDX_X int idx_x = gidx - c + jx; if (idx_x < 0) idx_x += Nx; if (idx_x >= Nx) idx_x -= Nx;
#define CONV_PERIODIC_IDX_Y int idx_y = gidy - c + jy; if (idx_y < 0) idx_y += Ny; if (idx_y >= Ny) idx_y -= Ny;
#define CONV_PERIODIC_IDX_Z int idx_z = gidz - c + jz; if (idx_z < 0) idx_z += Nz; if (idx_z >= Nz) idx_z -= Nz;



/******************************************************************************/
/**************************** 1D Convolution **********************************/
/******************************************************************************/


// Convolution with 1D kernel along axis "X" (fast dimension)
// Works for batched 1D on 2D and batched 2D on 3D, along axis "X".
__kernel void convol_1D_X(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int L, // filter size
    int Nx, // input/output number of columns
    int Ny, // input/output number of rows
    int Nz  // input/output depth
)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    if ((gidx >= Nx) || (gidy >= Ny) || (gidz >= Nz)) return;

    int c, hL, hR;
    GET_CENTER_HL(L);
    float sum = 0.0f;

    for (int jx = 0; jx <= hR+hL; jx++) {
        CONV_PERIODIC_IDX_X; // Get index "x"
        sum += input[(gidz*Ny + gidy)*Nx + idx_x] * filter[L-1 - jx];
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 1D kernel along axis "Y"
// Works for batched 1D on 2D and batched 2D on 3D, along axis "Y".
__kernel void convol_1D_Y(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int L, // filter size
    int Nx, // input/output number of columns
    int Ny, // input/output number of rows
    int Nz  // input/output depth
)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    if ((gidx >= Nx) || (gidy >= Ny) || (gidz >= Nz)) return;

    int c, hL, hR;
    GET_CENTER_HL(L);
    float sum = 0.0f;

    for (int jy = 0; jy <= hR+hL; jy++) {
        CONV_PERIODIC_IDX_Y; // Get index "y"
        sum += input[(gidz*Ny + idx_y)*Nx + gidx] * filter[L-1 - jy];
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}




// Convolution with 1D kernel along axis "Z"
// Works for batched 1D on 2D and batched 2D on 3D, along axis "Z".
__kernel void convol_1D_Z(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int L, // filter size
    int Nx, // input/output number of columns
    int Ny, // input/output number of rows
    int Nz  // input/output depth
)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    if ((gidx >= Nx) || (gidy >= Ny) || (gidz >= Nz)) return;

    int c, hL, hR;
    GET_CENTER_HL(L);
    float sum = 0.0f;

    for (int jz = 0; jz <= hR+hL; jz++) {
        CONV_PERIODIC_IDX_Z; // Get index "z"
        sum += input[(idx_z*Ny + gidy)*Nx + gidx] * filter[L-1 - jz];
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}




// Convolution with 2D kernel along axis "X,Y"
// Works for batched 2D on 3D.
__kernel void convol_2D_XY(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int Lx, // filter number of columns,
    int Ly, // filter number of rows,
    int Nx, // input/output number of columns
    int Ny, // input/output number of rows
    int Nz  // input/output depth
)
{
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    if ((gidx >= Nx) || (gidy >= Ny) || (gidz >= Nz)) return;

    int c, hL, hR;
    GET_CENTER_HL(Lx);
    float sum = 0.0f;

    for (int jx = 0; jx <= hR+hL; jx++) {
        CONV_PERIODIC_IDX_X; // Get index "x"
        for (int jy = 0; jy <= hR+hL; jy++) {
            CONV_PERIODIC_IDX_Y; // Get index "y"
            sum += input[(gidz*Ny + idx_y)*Nx + idx_x] * filter[(Ly-1-jy)*Lx + (Lx-1 - jx)];
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}









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



