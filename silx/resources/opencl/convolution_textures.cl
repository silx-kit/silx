/******************************************************************************/
/**************************** Macros ******************************************/
/******************************************************************************/

// Convolution index for filter (+0.5f for texture access)
#define FILTER_INDEX_X (Lx-1-jx+0.5f)
#define FILTER_INDEX_Y (Ly-1-jy+0.5f)
#define FILTER_INDEX_Z (Lz-1-jz+0.5f)
// Convolution index for image (+0.5f for texture access)
#define IMAGE_INDEX_X (gidx - c + jx + 0.5f)
#define IMAGE_INDEX_Y (gidy - c + jy + 0.5f)
#define IMAGE_INDEX_Z (gidz - c + jz + 0.5f)

// Filter access patterns
#define READ_FILTER_1D read_imagef(filter, sampler, (float) FILTER_INDEX_X).x;
#define READ_FILTER_2D read_imagef(filter, sampler, (float2) (FILTER_INDEX_X, FILTER_INDEX_Y)).x;
#define READ_FILTER_3D read_imagef(filter, sampler, (float4) (FILTER_INDEX_X, FILTER_INDEX_Y, FILTER_INDEX_Z), 0f).x;

// Image access patterns
#define READ_IMAGE_1D read_imagef(input, sampler, (float) IMAGE_INDEX_X).x

#define READ_IMAGE_2D_X read_imagef(input, sampler, (float2) (IMAGE_INDEX_X , gidy + 0.5f)).x
#define READ_IMAGE_2D_Y read_imagef(input, sampler, (float2) (gidx + 0.5f , IMAGE_INDEX_Y)).x
#define READ_IMAGE_2D_XY read_imagef(input, sampler, (float2) (IMAGE_INDEX_X, IMAGE_INDEX_Y)).x

#define READ_IMAGE_3D_X read_imagef(input, sampler, (float4) (IMAGE_INDEX_X, gidy+0.5f, gidz+0.5f, 0f)).x
#define READ_IMAGE_3D_Y read_imagef(input, sampler, (float4) (gidx+0.5f, IMAGE_INDEX_Y, gidz+0.5f, 0f)).x
#define READ_IMAGE_3D_Z read_imagef(input, sampler, (float4) (gidx+0.5f, gidy+0.5f, IMAGE_INDEX_Z, 0f)).x
#define READ_IMAGE_3D_XY read_imagef(input, sampler, (float4) (IMAGE_INDEX_X, IMAGE_INDEX_Y, gidz+0.5f, 0f)).x
#define READ_IMAGE_3D_XZ read_imagef(input, sampler, (float4) (IMAGE_INDEX_X, gidy+0.5f, IMAGE_INDEX_Z, 0f)).x
#define READ_IMAGE_3D_YZ read_imagef(input, sampler, (float4) (gidx+0.5f, IMAGE_INDEX_Y, IMAGE_INDEX_Z, 0f)).x
#define READ_IMAGE_3D_XYZ read_imagef(input, sampler, (float4) (IMAGE_INDEX_X, IMAGE_INDEX_Y, IMAGE_INDEX_Z, 0f)).x

#if FILTER_DIMS == 1
    #define FILTER_TYPE image1d_t
    #define READ_FILTER_VAL READ_FILTER_1D
#elif FILTER_DIMS == 2
    #define FILTER_TYPE image2d_t
    #define READ_FILTER_VAL READ_FILTER_2D
#elif FILTER_DIMS == 3
    #define FILTER_TYPE image3d_t
    #define READ_FILTER_VAL READ_FILTER_3D
#endif

#if IMAGE_DIMS == 1
    #define IMAGE_TYPE image1d_t
    #define READ_IMAGE_X READ_IMAGE_1D
#elif IMAGE_DIMS == 2
    #define IMAGE_TYPE image2d_t
    #define READ_IMAGE_X READ_IMAGE_2D_X
    #define READ_IMAGE_Y READ_IMAGE_2D_Y
    #define READ_IMAGE_XY READ_IMAGE_2D_XY
#elif IMAGE_DIMS == 3
    #define IMAGE_TYPE image3d_t
    #define READ_IMAGE_X READ_IMAGE_3D_X
    #define READ_IMAGE_Y READ_IMAGE_3D_Y
    #define READ_IMAGE_Z READ_IMAGE_3D_Z
    #define READ_IMAGE_XY READ_IMAGE_3D_XY
    #define READ_IMAGE_XZ READ_IMAGE_3D_XZ
    #define READ_IMAGE_YZ READ_IMAGE_3D_YZ
    #define READ_IMAGE_XYZ READ_IMAGE_3D_XYZ
#endif

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


/******************************************************************************/
/**************************** 1D Convolution **********************************/
/******************************************************************************/


// Convolution with 1D kernel along axis "X" (fast dimension)
// Works for batched 1D on 2D and batched 2D on 3D, along axis "X".
__kernel void convol_1D_X_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
        sum += READ_IMAGE_X * READ_FILTER_X;
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 1D kernel along axis "Y"
// Works for batched 1D on 2D and batched 2D on 3D, along axis "Y".
__kernel void convol_1D_Y_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
        sum += READ_IMAGE_Y * READ_FILTER_VAL;
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 1D kernel along axis "Z"
// Works for batched 1D on 2D and batched 2D on 3D, along axis "Z".
__kernel void convol_1D_Z_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
        sum += READ_IMAGE_Z * READ_FILTER_VAL;
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


/******************************************************************************/
/**************************** 2D Convolution **********************************/
/******************************************************************************/

// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_XY_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only image2d_t filter,
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
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int c, hL, hR;
    GET_CENTER_HL(Lx);
    float sum = 0.0f;

    for (int jy = 0; jy <= hR+hL; jy++) {
        for (int jx = 0; jx <= hR+hL; jx++) {
            sum += READ_IMAGE_XY * READ_FILTER_VAL;
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_XZ_tex(
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
        for (int jx = 0; jx <= hR+hL; jx++) {
            sum += READ_IMAGE_XZ * READ_FILTER_VAL;
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_YZ_tex(
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
        for (int jy = 0; jy <= hR+hL; jy++) {
            sum += READ_IMAGE_YZ * READ_FILTER_VAL;
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}



/******************************************************************************/
/**************************** 3D Convolution **********************************/
/******************************************************************************/

// Convolution with 3D kernel
__kernel void convol_3D_XYZ_tex(
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
        for (int jy = 0; jy <= hR+hL; jy++) {
            for (int jx = 0; jx <= hR+hL; jx++) {
                sum += READ_IMAGE_XYZ * READ_FILTER_VAL;
            }
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}

