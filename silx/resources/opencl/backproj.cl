const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void backproj_kernel(
    int num_proj,
    int num_bins,
    float axis_position,
    __global float *d_SLICE,
    __read_only image2d_t d_sino,
    float gpu_offset_x,
    float gpu_offset_y,
    __global float * d_cos_s, // precalculated cos(theta[i])
    __global float * d_sin_s, // precalculated sin(theta[i])
    __global float * d_axis_s, // array of axis positions (n_projs)
    __local float* shared2)     // 768B of local mem
{
    const int tidx = get_local_id(0); //threadIdx.x;
    const int bidx = get_group_id(0); //blockIdx.x;
    const int tidy = get_local_id(1); //threadIdx.y;
    const int bidy = get_group_id(1); //blockIdx.y;

    //~ __local float shared[768];
    //~ float  * sh_sin  = shared;
    //~ float  * sh_cos  = shared+256;
    //~ float  * sh_axis = sh_cos+256;

    __local float sh_cos[256];
    __local float sh_sin[256];
    __local float sh_axis[256];

    float pcos, psin;
    float h0, h1, h2, h3;
    const float apos_off_x= gpu_offset_x - axis_position ;
    const float apos_off_y= gpu_offset_y - axis_position ;
    float acorr05;
    float res0 = 0, res1 = 0, res2 = 0, res3 = 0;

    const float bx00 = (32 * bidx + 2 * tidx + 0 + apos_off_x  ) ;
    const float by00 = (32 * bidy + 2 * tidy + 0 + apos_off_y  ) ;

    int lettofino=0;
    for(int proje=0; proje<num_proj; proje++) {
        if(proje>=lettofino) {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ip = tidy*16+tidx;
            if( lettofino+ip < num_proj) {
                sh_cos [ip] = d_cos_s[lettofino+ip] ;
                sh_sin [ip] = d_sin_s[lettofino+ip] ;
                sh_axis[ip] = d_axis_s[lettofino+ip] ;
            }
            lettofino=lettofino+256; // 256=16*16 block size
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        pcos = sh_cos[256-lettofino + proje] ;
        psin = sh_sin[256-lettofino + proje] ;

        acorr05 = sh_axis[256 - lettofino + proje] ;

        h0 = (acorr05 + bx00*pcos - by00*psin);
        h1 = (acorr05 + (bx00+0)*pcos - (by00+1)*psin);
        h2 = (acorr05 + (bx00+1)*pcos - (by00+0)*psin);
        h3 = (acorr05 + (bx00+1)*pcos - (by00+1)*psin);

        if(h0>=0 && h0<num_bins) res0 += read_imagef(d_sino, sampler, (float2) (h0 +0.5f,proje +0.5f)).x; // tex2D(texProjes,h0 +0.5f,proje +0.5f);
        if(h1>=0 && h1<num_bins) res1 += read_imagef(d_sino, sampler, (float2) (h1 +0.5f,proje +0.5f)).x; //tex2D(texProjes,h1 +0.5f,proje +0.5f);
        if(h2>=0 && h2<num_bins) res2 += read_imagef(d_sino, sampler, (float2) (h2 +0.5f,proje +0.5f)).x; //tex2D(texProjes,h2 +0.5f,proje +0.5f);
        if(h3>=0 && h3<num_bins) res3 += read_imagef(d_sino, sampler, (float2) (h3 +0.5f,proje +0.5f)).x; //tex2D(texProjes,h3 +0.5f,proje +0.5f);
    }
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 0] = res0;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 0] = res1;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 1] = res2;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 1] = res3;
}




/**
 * Needed for CPU version
 *
 */
float bilinear_interpolation(float* img, int Nx, int Ny, float x, float y) {
    /*
        (xm, ym)        (xp, ym)
                 (x, y)
        (xm, yp)        (xp, yp)
    */
    int xm = (int) floor(x);
    int xp = (int) ceil(x);
    int ym = (int) floor(y);
    int yp = (int) ceil(y);
    float val;
    float tol = 0.001f; // CHECKME
    if ((x - xm) < tol && (y - ym) < tol) val = img[ym*Nx + xm];
    else if ((xp - x) < tol && (yp - y) < tol) val = img[yp*Nx + xp];
    else {
        // Mirror - TODO: clamp ?
        if (xm < 0) xm = 0;
        if (xp >= Nx) xp = Nx - 1;
        if (ym < 0) ym = 0;
        if (yp >= Ny) yp = Ny -1;
        // Interp
        val = img[yp*Nx+xm]*(xp-x)*(y-ym)
                    + img[yp*Nx+xp]*(x-xm)*(y-ym)
                    + img[ym*Nx + xm]*(xp-x)*(yp-y)
                    + img[ym*Nx + xp]*(x-xm)*(yp-y);

    }
    return val;
}





/**
 *
 *  Same kernel as backproj_kernel, but targets the CPU (no texture)
 *
**/
__kernel void backproj_cpu_kernel(
    int num_proj,
    int num_bins,
    float axis_position,
    __global float *d_SLICE,
    __global float* d_sino,
    float gpu_offset_x,
    float gpu_offset_y,
    __global float * d_cos_s, // precalculated cos(theta[i])
    __global float * d_sin_s, // precalculated sin(theta[i])
    __global float * d_axis_s, // array of axis positions (n_projs)
    __local float* shared2)     // 768B of local mem
{
    const int tidx = get_local_id(0); //threadIdx.x;
    const int bidx = get_group_id(0); //blockIdx.x;
    const int tidy = get_local_id(1); //threadIdx.y;
    const int bidy = get_group_id(1); //blockIdx.y;

    //~ __local float shared[768];
    //~ float  * sh_sin  = shared;
    //~ float  * sh_cos  = shared+256;
    //~ float  * sh_axis = sh_cos+256;

    __local float sh_cos[256];
    __local float sh_sin[256];
    __local float sh_axis[256];

    float pcos, psin;
    float h0, h1, h2, h3;
    const float apos_off_x= gpu_offset_x - axis_position ;
    const float apos_off_y= gpu_offset_y - axis_position ;
    float acorr05;
    float res0 = 0, res1 = 0, res2 = 0, res3 = 0;

    const float bx00 = (32 * bidx + 2 * tidx + 0 + apos_off_x  ) ;
    const float by00 = (32 * bidy + 2 * tidy + 0 + apos_off_y  ) ;

    int lettofino=0;
    for(int proje=0; proje<num_proj; proje++) {
        if(proje>=lettofino) {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ip = tidy*16+tidx;
            if( lettofino+ip < num_proj) {
                sh_cos [ip] = d_cos_s[lettofino+ip] ;
                sh_sin [ip] = d_sin_s[lettofino+ip] ;
                sh_axis[ip] = d_axis_s[lettofino+ip] ;
            }
            lettofino=lettofino+256; // 256=16*16 block size
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        pcos = sh_cos[256-lettofino + proje] ;
        psin = sh_sin[256-lettofino + proje] ;

        acorr05 = sh_axis[256 - lettofino + proje] ;

        h0 = (acorr05 + bx00*pcos - by00*psin);
        h1 = (acorr05 + (bx00+0)*pcos - (by00+1)*psin);
        h2 = (acorr05 + (bx00+1)*pcos - (by00+0)*psin);
        h3 = (acorr05 + (bx00+1)*pcos - (by00+1)*psin);

        if(h0>=0 && h0<num_bins) res0 += bilinear_interpolation(d_sino, num_bins, num_proj, h0 +0.5f,proje +0.5f) // tex2D(texProjes,h0 +0.5f,proje +0.5f);
        if(h1>=0 && h1<num_bins) res1 += bilinear_interpolation(d_sino, num_bins, num_proj, h1 +0.5f,proje +0.5f) //tex2D(texProjes,h1 +0.5f,proje +0.5f);
        if(h2>=0 && h2<num_bins) res2 += bilinear_interpolation(d_sino, num_bins, num_proj, h2 +0.5f,proje +0.5f) //tex2D(texProjes,h2 +0.5f,proje +0.5f);
        if(h3>=0 && h3<num_bins) res3 += bilinear_interpolation(d_sino, num_bins, num_proj, h3 +0.5f,proje +0.5f) //tex2D(texProjes,h3 +0.5f,proje +0.5f);
    }
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 0] = res0;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 0] = res1;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+0) + bidx*32 + tidx*2 + 1] = res2;
    d_SLICE[ 32*get_num_groups(0)*(bidy*32+tidy*2+1) + bidx*32 + tidx*2 + 1] = res3;
}

