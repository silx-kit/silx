/******************************************************************************/
/**************************** Macros ******************************************/
/******************************************************************************/

// Error handling
#ifndef IMAGE_DIMS
    #error "IMAGE_DIMS must be defined"
#endif
#ifndef FILTER_DIMS
    #error "FILTER_DIMS must be defined"
#endif
#if FILTER_DIMS > IMAGE_DIMS
    #error "Filter cannot have more dimensions than image"
#endif

// Boundary handling modes
#define CONV_MODE_REFLECT 0 // CLK_ADDRESS_MIRRORED_REPEAT : cba|abcd|dcb
#define CONV_MODE_NEAREST 1 // CLK_ADDRESS_CLAMP_TO_EDGE : aaa|abcd|ddd
#define CONV_MODE_WRAP 2 // CLK_ADDRESS_REPEAT : bcd|abcd|abc
#define CONV_MODE_CONSTANT 3 // CLK_ADDRESS_CLAMP : 000|abcd|000
#ifndef USED_CONV_MODE
    #define USED_CONV_MODE CONV_MODE_NEAREST
#endif
#if USED_CONV_MODE == CONV_MODE_REFLECT
    #define CLK_BOUNDARY CLK_ADDRESS_MIRRORED_REPEAT
    #define CLK_COORDS CLK_NORMALIZED_COORDS_TRUE
    #define USE_NORM_COORDS
#elif USED_CONV_MODE == CONV_MODE_NEAREST
    #define CLK_BOUNDARY CLK_ADDRESS_CLAMP_TO_EDGE
    #define CLK_COORDS CLK_NORMALIZED_COORDS_FALSE
#elif USED_CONV_MODE == CONV_MODE_WRAP
    #define CLK_BOUNDARY CLK_ADDRESS_REPEAT
    #define CLK_COORDS CLK_NORMALIZED_COORDS_TRUE
    #define USE_NORM_COORDS
#elif USED_CONV_MODE == CONV_MODE_CONSTANT
    #define CLK_BOUNDARY CLK_ADDRESS_CLAMP
    #define CLK_COORDS CLK_NORMALIZED_COORDS_FALSE
#else
    #error "Unknown convolution mode"
#endif



// Convolution index for filter
#define FILTER_INDEX(j) (Lx - 1 - j)

// Filter access patterns
#define READ_FILTER_1D(j) read_imagef(filter, (int2) (FILTER_INDEX(j), 0)).x;
#define READ_FILTER_2D(jx, jy) read_imagef(filter, (int2) (FILTER_INDEX(jx), FILTER_INDEX(jy))).x;
#define READ_FILTER_3D(jx, jy, jz) read_imagef(filter, (int4) (FILTER_INDEX(jx), FILTER_INDEX(jy), FILTER_INDEX(jz), 0)).x;


// Convolution index for image
#ifdef USE_NORM_COORDS
    #define IMAGE_INDEX_X (gidx*1.0f +0.5f - c + jx)/Nx
    #define IMAGE_INDEX_Y (gidy*1.0f +0.5f - c + jy)/Ny
    #define IMAGE_INDEX_Z (gidz*1.0f +0.5f - c + jz)/Nz
    #define RET_TYPE_1 float
    #define RET_TYPE_2 float2
    #define RET_TYPE_4 float4
    #define C_ZERO 0.5f
    #define GIDX (gidx*1.0f + 0.5f)/Nx
    #define GIDY (gidy*1.0f + 0.5f)/Ny
    #define GIDZ (gidz*1.0f + 0.5f)/Nz
#else
    #define IMAGE_INDEX_X (gidx - c + jx)
    #define IMAGE_INDEX_Y (gidy - c + jy)
    #define IMAGE_INDEX_Z (gidz - c + jz)
    #define RET_TYPE_1 int
    #define RET_TYPE_2 int2
    #define RET_TYPE_4 int4
    #define C_ZERO 0
    #define GIDX gidx
    #define GIDY gidy
    #define GIDZ gidz
#endif

static const sampler_t sampler = CLK_COORDS | CLK_BOUNDARY | CLK_FILTER_NEAREST;

// Image access patterns
#define READ_IMAGE_1D read_imagef(input, sampler, (RET_TYPE_2) (IMAGE_INDEX_X, C_ZERO)).x

#define READ_IMAGE_2D_X read_imagef(input, sampler, (RET_TYPE_2) (IMAGE_INDEX_X , GIDY)).x
#define READ_IMAGE_2D_Y read_imagef(input, sampler, (RET_TYPE_2) (GIDX, IMAGE_INDEX_Y)).x
#define READ_IMAGE_2D_XY read_imagef(input, sampler, (RET_TYPE_2) (IMAGE_INDEX_X, IMAGE_INDEX_Y)).x

#define READ_IMAGE_3D_X read_imagef(input, sampler, (RET_TYPE_4) (IMAGE_INDEX_X, GIDY, GIDZ, C_ZERO)).x
#define READ_IMAGE_3D_Y read_imagef(input, sampler, (RET_TYPE_4) (GIDX, IMAGE_INDEX_Y, GIDZ, C_ZERO)).x
#define READ_IMAGE_3D_Z read_imagef(input, sampler, (RET_TYPE_4) (GIDX, GIDY, IMAGE_INDEX_Z, C_ZERO)).x
#define READ_IMAGE_3D_XY read_imagef(input, sampler, (RET_TYPE_4) (IMAGE_INDEX_X, IMAGE_INDEX_Y, GIDZ, C_ZERO)).x
#define READ_IMAGE_3D_XZ read_imagef(input, sampler, (RET_TYPE_4) (IMAGE_INDEX_X, GIDY, IMAGE_INDEX_Z, C_ZERO)).x
#define READ_IMAGE_3D_YZ read_imagef(input, sampler, (RET_TYPE_4) (GIDX, IMAGE_INDEX_Y, IMAGE_INDEX_Z, C_ZERO)).x
#define READ_IMAGE_3D_XYZ read_imagef(input, sampler, (RET_TYPE_4) (IMAGE_INDEX_X, IMAGE_INDEX_Y, IMAGE_INDEX_Z, C_ZERO)).x


// NOTE: pyopencl and OpenCL < 1.2 do not support image1d_t
#if FILTER_DIMS == 1
    #define FILTER_TYPE image2d_t
    #define READ_FILTER_VAL(j) READ_FILTER_1D(j)
#elif FILTER_DIMS == 2
    #define FILTER_TYPE image2d_t
    #define READ_FILTER_VAL(jx, jy) READ_FILTER_2D(jx, jy)
#elif FILTER_DIMS == 3
    #define FILTER_TYPE image3d_t
    #define READ_FILTER_VAL(jx, jy, jz) READ_FILTER_3D(jx, jy, jz)
#endif

#if IMAGE_DIMS == 1
    #define IMAGE_TYPE image2d_t
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

#if FILTER_DIMS == 1
// Convolution with 1D kernel along axis "X" (fast dimension)
// Works for batched 1D on 2D and batched 2D on 3D, along axis "X".
__kernel void convol_1D_X_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
    int Lx, // filter size
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
        sum += READ_IMAGE_X * READ_FILTER_VAL(jx);
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


#if IMAGE_DIMS >= 2
// Convolution with 1D kernel along axis "Y"
// Works for batched 1D on 2D and batched 2D on 3D, along axis "Y".
__kernel void convol_1D_Y_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
    int Lx, // filter size
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
        sum += READ_IMAGE_Y * READ_FILTER_VAL(jy);
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}
#endif

#if IMAGE_DIMS == 3
// Convolution with 1D kernel along axis "Z"
// Works for batched 1D on 2D and batched 2D on 3D, along axis "Z".
__kernel void convol_1D_Z_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
    int Lx, // filter size
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
        sum += READ_IMAGE_Z * READ_FILTER_VAL(jz);
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}
#endif
#endif

/******************************************************************************/
/**************************** 2D Convolution **********************************/
/******************************************************************************/

#if IMAGE_DIMS >= 2 && FILTER_DIMS == 2
// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_XY_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
        for (int jx = 0; jx <= hR+hL; jx++) {
            sum += READ_IMAGE_XY * READ_FILTER_VAL(jx, jy);
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}
#endif

#if IMAGE_DIMS == 3 && FILTER_DIMS == 2
// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_XZ_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
            sum += READ_IMAGE_XZ * READ_FILTER_VAL(jx, jz);
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}


// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_YZ_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
        for (int jy = 0; jy <= hR+hL; jy++) {
            sum += READ_IMAGE_YZ * READ_FILTER_VAL(jy, jz);
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}
#endif


/******************************************************************************/
/**************************** 3D Convolution **********************************/
/******************************************************************************/

#if IMAGE_DIMS == 3 && FILTER_DIMS == 3
// Convolution with 3D kernel
__kernel void convol_3D_XYZ_tex(
    read_only IMAGE_TYPE input,
    __global float * output,
    read_only FILTER_TYPE filter,
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
                sum += READ_IMAGE_XYZ * READ_FILTER_VAL(jx, jy, jz);
            }
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}
#endif
