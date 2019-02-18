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


// Get the moving convolution index for periodic boundary extension.
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


/******************************************************************************/
/**************************** 2D Convolution **********************************/
/******************************************************************************/

// Convolution with 2D kernel
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

    for (int jy = 0; jy <= hR+hL; jy++) {
        CONV_PERIODIC_IDX_Y; // Get index "y"
        for (int jx = 0; jx <= hR+hL; jx++) {
            CONV_PERIODIC_IDX_X; // Get index "x"
            sum += input[(gidz*Ny + idx_y)*Nx + idx_x] * filter[(Ly-1-jy)*Lx + (Lx-1 - jx)];
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_XZ(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int Lx, // filter number of columns,
    int Lz, // filter number of rows,
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

    for (int jz = 0; jz <= hR+hL; jz++) {
        CONV_PERIODIC_IDX_Z; // Get index "z"
        for (int jx = 0; jx <= hR+hL; jx++) {
            CONV_PERIODIC_IDX_X; // Get index "x"
            sum += input[(idx_z*Ny + gidy)*Nx + idx_x] * filter[(Lz-1-jz)*Lx + (Lx-1 - jx)];
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_YZ(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int Ly, // filter number of columns,
    int Lz, // filter number of rows,
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
    GET_CENTER_HL(Ly);
    float sum = 0.0f;

    for (int jz = 0; jz <= hR+hL; jz++) {
        CONV_PERIODIC_IDX_Z; // Get index "z"
        for (int jy = 0; jy <= hR+hL; jy++) {
            CONV_PERIODIC_IDX_Y; // Get index "y"
            sum += input[(idx_z*Ny + idx_y)*Nx + gidx] * filter[(Lz-1-jz)*Ly + (Ly-1 - jy)];
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}



/******************************************************************************/
/**************************** 3D Convolution **********************************/
/******************************************************************************/

// Convolution with 3D kernel
__kernel void convol_3D_XYZ(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int Lx, // filter number of columns,
    int Ly, // filter number of rows,
    int Lz, // filter number of rows,
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

    for (int jz = 0; jz <= hR+hL; jz++) {
        CONV_PERIODIC_IDX_Z; // Get index "z"
        for (int jy = 0; jy <= hR+hL; jy++) {
            CONV_PERIODIC_IDX_Y; // Get index "y"
            for (int jx = 0; jx <= hR+hL; jx++) {
                CONV_PERIODIC_IDX_X; // Get index "x"
                sum += input[(idx_z*Ny + idx_y)*Nx + idx_x] * filter[((Lz-1-jz)*Ly + (Ly-1-jy))*Lx + (Lx-1 - jx)];
            }
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}

