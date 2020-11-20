/*
 *   Project: silx: filtered backprojection
 *
 *   Copyright (C) 2016-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: A. Mirone
 *                      P. Paleo
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


/*******************************************************************************/
/************************ GPU VERSION (with textures) **************************/
/*******************************************************************************/

#ifndef DONT_USE_TEXTURES
kernel void backproj_kernel(
    int num_proj,
    int num_bins,
    float axis_position,
    global float *d_SLICE,
    read_only image2d_t d_sino,
    float gpu_offset_x,
    float gpu_offset_y,
    global float * d_cos_s,  // precalculated cos(theta[i])
    global float* d_sin_s,   // precalculated sin(theta[i])
    global float* d_axis_s,  // array of axis positions (n_projs)
    local float* shared2)    // 768B of local mem
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    const int tidx = get_local_id(0); //threadIdx.x;
    const int bidx = get_group_id(0); //blockIdx.x;
    const int tidy = get_local_id(1); //threadIdx.y;
    const int bidy = get_group_id(1); //blockIdx.y;

    local float sh_cos[256];
    local float sh_sin[256];
    local float sh_axis[256];

    float pcos, psin;
    float h0, h1, h2, h3;
    const float apos_off_x= gpu_offset_x - axis_position ;
    const float apos_off_y= gpu_offset_y - axis_position ;
    float acorr05;
    float res0 = 0, res1 = 0, res2 = 0, res3 = 0;

    const float bx00 = (32 * bidx + 2 * tidx + 0 + apos_off_x  ) ;
    const float by00 = (32 * bidy + 2 * tidy + 0 + apos_off_y  ) ;

    int read=0;
    for(int proj=0; proj<num_proj; proj++) {
        if(proj>=read) {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ip = tidy*16+tidx;
            if( read+ip < num_proj) {
                sh_cos [ip] = d_cos_s[read+ip] ;
                sh_sin [ip] = d_sin_s[read+ip] ;
                sh_axis[ip] = d_axis_s[read+ip] ;
            }
            read=read+256; // 256=16*16 block size
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        pcos = sh_cos[256-read + proj] ;
        psin = sh_sin[256-read + proj] ;

        acorr05 = sh_axis[256 - read + proj] ;

        h0 = (acorr05 + bx00*pcos - by00*psin);
        h1 = (acorr05 + (bx00+0)*pcos - (by00+1)*psin);
        h2 = (acorr05 + (bx00+1)*pcos - (by00+0)*psin);
        h3 = (acorr05 + (bx00+1)*pcos - (by00+1)*psin);

        if(h0>=0 && h0<num_bins) res0 += read_imagef(d_sino, sampler, (float2) (h0 +0.5f,proj +0.5f)).x; // tex2D(texprojs,h0 +0.5f,proj +0.5f);
        if(h1>=0 && h1<num_bins) res1 += read_imagef(d_sino, sampler, (float2) (h1 +0.5f,proj +0.5f)).x; // tex2D(texprojs,h1 +0.5f,proj +0.5f);
        if(h2>=0 && h2<num_bins) res2 += read_imagef(d_sino, sampler, (float2) (h2 +0.5f,proj +0.5f)).x; // tex2D(texprojs,h2 +0.5f,proj +0.5f);
        if(h3>=0 && h3<num_bins) res3 += read_imagef(d_sino, sampler, (float2) (h3 +0.5f,proj +0.5f)).x; // tex2D(texprojs,h3 +0.5f,proj +0.5f);
    }
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 0] = res0;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 0] = res1;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 1] = res2;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 1] = res3;
}
#endif




/*******************************************************************************/
/********************* CPU VERSION (without textures) **************************/
/*******************************************************************************/


#define CLIP_MAX(x, N) (fmin(fmax(x, 0.0f), (N - 1.0f)))

#define FLOORCEIL_x(x) {\
    xm = (int) floor(x);\
    xp = (int) ceil(x);\
}

#define ADJACENT_PIXELS_VALS(arr, Nx, y, xm, xp) ((float2) (arr[y*Nx+xm], arr[y*Nx+xp]))

//Simple linear interpolator for working on the GPU
static float linear_interpolation(float2 vals,
                                  float x,
                                  int xm,
                                  int xp)
{
    if (xm == xp)
        return vals.s0;
    else
        return (vals.s0 * (xp - x)) + (vals.s1 * (x - xm));
}

/**
 *
 *  Same kernel as backproj_kernel, but targets the CPU (no texture)
 *
**/
kernel void backproj_cpu_kernel(
    int num_proj,
    int num_bins,
    float axis_position,
    global float *d_SLICE,
    global float* d_sino,
    float gpu_offset_x,
    float gpu_offset_y,
    global float * d_cos_s, // precalculated cos(theta[i])
    global float * d_sin_s, // precalculated sin(theta[i])
    global float * d_axis_s, // array of axis positions (n_projs)
    local float* shared2)     // 768B of local mem
{
    const int tidx = get_local_id(0); //threadIdx.x;
    const int bidx = get_group_id(0); //blockIdx.x;
    const int tidy = get_local_id(1); //threadIdx.y;
    const int bidy = get_group_id(1); //blockIdx.y;

    local float sh_cos[256];
    local float sh_sin[256];
    local float sh_axis[256];

    float pcos, psin;
    float h0, h1, h2, h3;
    const float apos_off_x= gpu_offset_x - axis_position ;
    const float apos_off_y= gpu_offset_y - axis_position ;
    float acorr05;
    float res0 = 0, res1 = 0, res2 = 0, res3 = 0;

    const float bx00 = (32 * bidx + 2 * tidx + 0 + apos_off_x  ) ;
    const float by00 = (32 * bidy + 2 * tidy + 0 + apos_off_y  ) ;

    int read=0;
    for(int proj=0; proj<num_proj; proj++) {
        if(proj>=read) {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ip = tidy*16+tidx;
            if( read+ip < num_proj) {
                sh_cos [ip] = d_cos_s[read+ip] ;
                sh_sin [ip] = d_sin_s[read+ip] ;
                sh_axis[ip] = d_axis_s[read+ip] ;
            }
            read=read+256; // 256=16*16 block size
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        pcos = sh_cos[256-read + proj] ;
        psin = sh_sin[256-read + proj] ;

        acorr05 = sh_axis[256 - read + proj] ;

        h0 = (acorr05 + bx00*pcos - by00*psin);
        h1 = (acorr05 + (bx00+0)*pcos - (by00+1)*psin);
        h2 = (acorr05 + (bx00+1)*pcos - (by00+0)*psin);
        h3 = (acorr05 + (bx00+1)*pcos - (by00+1)*psin);


        float x;
        int ym, xm, xp;
        ym = proj;
        float2 vals;

        if(h0>=0 && h0<num_bins) {
            x = CLIP_MAX(h0, num_bins);
            FLOORCEIL_x(x);
            vals = ADJACENT_PIXELS_VALS(d_sino, num_bins, ym, xm, xp);
            res0 += linear_interpolation(vals, x, xm, xp);
        }
        if(h1>=0 && h1<num_bins) {
            x = CLIP_MAX(h1, num_bins);
            FLOORCEIL_x(x);
            vals = ADJACENT_PIXELS_VALS(d_sino, num_bins, ym, xm, xp);
            res1 += linear_interpolation(vals, x, xm, xp);
        }
        if(h2>=0 && h2<num_bins) {
            x = CLIP_MAX(h2, num_bins);
            FLOORCEIL_x(x);
            vals = ADJACENT_PIXELS_VALS(d_sino, num_bins, ym, xm, xp);
            res2 += linear_interpolation(vals, x, xm, xp);
        }
        if(h3>=0 && h3<num_bins) {
            x = CLIP_MAX(h3, num_bins);
            FLOORCEIL_x(x);
            vals = ADJACENT_PIXELS_VALS(d_sino, num_bins, ym, xm, xp);
            res3 += linear_interpolation(vals, x, xm, xp);
        }
    }
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 0] = res0;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 0] = res1;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 1] = res2;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 1] = res3;
}

