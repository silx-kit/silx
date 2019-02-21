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


/******************************************************************************/
/**************************** 2D Convolution **********************************/
/******************************************************************************/

// Convolution with 2D kernel
// Works for batched 2D on 3D.
__kernel void convol_2D_XY_tex(
    read_only image2d_t input,
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
            // Fun thing: "0.5" instead of "0.5f" plummets performances
            sum += read_imagef(
                input,
                sampler,
                (float2) (gidx - c + jx + 0.5f , gidy - c + jy + 0.5f)
            ).x * read_imagef(
                filter,
                sampler,
                (float2) (Lx-1-jx+0.5f, Ly-1-jy+0.5f)
            ).x;
        }
    }
    output[(gidz*Ny + gidy)*Nx + gidx] = sum;
}
