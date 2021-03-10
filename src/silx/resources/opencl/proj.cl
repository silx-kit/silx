/*
 *   Copyright (C) 2017-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 * Based on the projector of PyHST2 - https://forge.epn-campus.eu/projects/pyhst2
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

/*******************************************************************************/
/************************ GPU VERSION (with textures) **************************/
/*******************************************************************************/

#ifndef DONT_USE_TEXTURES
kernel void  forward_kernel(
        global float *d_Sino,
        read_only image2d_t d_slice,
        int dimslice,
        int num_bins,
        global float* angles_per_project ,
        float axis_position,
        global float *d_axis_corrections,
        global int* d_beginPos    ,
        global int* d_strideJoseph,
        global int* d_strideLine  ,
        int num_projections,
        int  dimrecx,
        int  dimrecy,
        float cpu_offset_x,
        float cpu_offset_y,
        int josephnoclip,
        int normalize)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    const int tidx = get_local_id(0);
    const int bidx = get_group_id(0);
    const int tidy = get_local_id(1);
    const int bidy = get_group_id(1);
    float angle;
    float cos_angle,sin_angle ;

    local float corrections[16];
    local int beginPos[16*2];
    local int strideJoseph[16*2];
    local int strideLine[16*2];

    // thread will use corrections[tidy]
    // All are read by first warp
    int offset, OFFSET;
    switch(tidy) {
    case 0:
        corrections[ tidx ]= d_axis_corrections[ bidy*16+tidx];
        break;
    case 1:
    case 2:
        offset = 16*(tidy-1);
        OFFSET = dimrecy*(tidy-1);
        beginPos    [offset + tidx ]=  d_beginPos[ OFFSET+ bidy*16+tidx]  ;
        break;
    case 3:
    case 4:
        offset = 16*(tidy-3);
        OFFSET = dimrecy*(tidy-3);
        strideJoseph[offset + tidx ]=  d_strideJoseph[OFFSET + bidy*16+tidx]  ;
        break;
    case 5:
    case 6:
        offset = 16*(tidy-5);
        OFFSET = dimrecy*(tidy-5);
        strideLine[offset + tidx ]=  d_strideLine[OFFSET + bidy*16+tidx]  ;
        break;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    angle = angles_per_project[ bidy*16+tidy ]   ;
    cos_angle = cos(angle);
    sin_angle = sin(angle);

    if(fabs(cos_angle) > 0.70710678f ) {
        if( cos_angle>0) {
            cos_angle = cos(angle);
            sin_angle = sin(angle);
        }
        else {
            cos_angle = -cos(angle);
            sin_angle = -sin(angle);
        }
    }
    else {
        if( sin_angle>0) {
            cos_angle =  sin(angle);
            sin_angle = -cos(angle);
        }
        else {
            cos_angle = -sin(angle);
            sin_angle =  cos(angle);
        }
    }
    float res=0.0f;
    float axis_corr = axis_position  + corrections[ tidy ];
    float axis      = axis_position ;
    float xpix = ( bidx*16+tidx )-cpu_offset_x;
    float posx = axis*(1.0f-sin_angle/cos_angle ) +(xpix-(axis_corr) )/cos_angle ;

    float shiftJ = sin_angle/cos_angle;
    float x1 = min(-sin_angle/cos_angle ,0.f);
    float x2 = max(-sin_angle/cos_angle ,0.f);

    float  Area;
    Area=1.0f/cos_angle;
    int stlA, stlB , stlAJ, stlBJ ;
    stlA=strideLine[16+tidy];
    stlB=strideLine[tidy];
    stlAJ=strideJoseph[16+tidy];
    stlBJ=strideJoseph[tidy];

    int beginA = beginPos[16+tidy ];
    int beginB = beginPos[tidy ];
    float add;
    int l;

    if(josephnoclip) {
        for(int j=0; j<dimslice; j++) {  // j: Joseph
            x1 = beginA +(posx)*stlA + (j)*stlAJ+1.5f;
            x2 = beginB +(posx)*stlB + (j)*stlBJ+1.5f;
            add = read_imagef(d_slice, sampler, (float2) (x1, x2)).x; // add = tex2D(texSlice, x1,x2);
            res += add;
            posx += shiftJ;
        }
    }
    else {
        for(int j=0; j<dimslice; j++) {  // j: Joseph
            x1 = beginA +(posx)*stlA + (j)*stlAJ+1.5f;
            x2 = beginB +(posx)*stlB + (j)*stlBJ+1.5f;
            l = (x1>=0.0f )*(x1<(dimslice+2))*( x2>=0.0f)*( x2<(dimslice+2) ) ;
            add = read_imagef(d_slice, sampler, (float2) (x1, x2)).x; // add = tex2D(texSlice, x1,x2);
            res += add*l;
            posx += shiftJ;
        }
    }

    if((bidy*16 + tidy) < num_projections && (bidx*16 + tidx) < num_bins) {
        res *= Area;
        if (normalize)
            res *= M_PI_F * 0.5f / num_projections;
        d_Sino[dimrecx*(bidy*16 + tidy) + (bidx*16 + tidx)] = res;
    }
}
#endif


/*******************************************************************************/
/********************* CPU VERSION (without textures) **************************/
/*******************************************************************************/


kernel void  forward_kernel_cpu(
        global float *d_Sino,
        global float* d_slice,
        int dimslice,
        int num_bins,
        global float* angles_per_project ,
        float axis_position,
        global float *d_axis_corrections,
        global int* d_beginPos    ,
        global int* d_strideJoseph,
        global int* d_strideLine  ,
        int num_projections,
        int  dimrecx,
        int  dimrecy,
        float cpu_offset_x,
        float cpu_offset_y,
        int josephnoclip,
        int normalize)
{

    const int tidx = get_local_id(0);
    const int bidx = get_group_id(0);
    const int tidy = get_local_id(1);
    const int bidy = get_group_id(1);
    float angle;
    float cos_angle,sin_angle ;

    local float corrections[16];
    local int beginPos[16*2];
    local int strideJoseph[16*2];
    local int strideLine[16*2];

    // thread will use corrections[tidy]
    // All are read by first warp
    int offset, OFFSET;
    switch(tidy) {
    case 0:
        corrections[ tidx ]= d_axis_corrections[ bidy*16+tidx];
        break;
    case 1:
    case 2:
        offset = 16*(tidy-1);
        OFFSET = dimrecy*(tidy-1);
        beginPos    [offset + tidx ]=  d_beginPos[ OFFSET+ bidy*16+tidx]  ;
        break;
    case 3:
    case 4:
        offset = 16*(tidy-3);
        OFFSET = dimrecy*(tidy-3);
        strideJoseph[offset + tidx ]=  d_strideJoseph[OFFSET + bidy*16+tidx]  ;
        break;
    case 5:
    case 6:
        offset = 16*(tidy-5);
        OFFSET = dimrecy*(tidy-5);
        strideLine[offset + tidx ]=  d_strideLine[OFFSET + bidy*16+tidx]  ;
        break;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    angle = angles_per_project[ bidy*16+tidy ]   ;
    cos_angle = cos(angle);
    sin_angle = sin(angle);

    if(fabs(cos_angle) > 0.70710678f ) {
        if( cos_angle>0) {
            cos_angle = cos(angle);
            sin_angle = sin(angle);
        }
        else {
            cos_angle = -cos(angle);
            sin_angle = -sin(angle);
        }
    }
    else {
        if( sin_angle>0) {
            cos_angle =  sin(angle);
            sin_angle = -cos(angle);
        }
        else {
            cos_angle = -sin(angle);
            sin_angle =  cos(angle);
        }
    }
    float res=0.0f;
    float axis_corr = axis_position  + corrections[ tidy ];
    float axis      = axis_position ;
    float xpix = ( bidx*16+tidx )-cpu_offset_x;
    float posx = axis*(1.0f-sin_angle/cos_angle ) +(xpix-(axis_corr) )/cos_angle ;

    float shiftJ = sin_angle/cos_angle;
    float x1 = min(-sin_angle/cos_angle ,0.f);
    float x2 = max(-sin_angle/cos_angle ,0.f);

    float  Area;
    Area=1.0f/cos_angle;
    int stlA, stlB , stlAJ, stlBJ ;
    stlA=strideLine[16+tidy];
    stlB=strideLine[tidy];
    stlAJ=strideJoseph[16+tidy];
    stlBJ=strideJoseph[tidy];

    int beginA = beginPos[16+tidy ];
    int beginB = beginPos[tidy ];
    int l;

    int ym, yp, xm, xp;
    float yc, xc;
    float val;
    if(josephnoclip) {
        for(int j=0; j<dimslice; j++) {  // j: Joseph
            x1 = beginA +(posx)*stlA + (j)*stlAJ+1.0f;
            x2 = beginB +(posx)*stlB + (j)*stlBJ+1.0f;
            /*
              Bilinear interpolation
            */
            yc = fmin(fmax(x2, 0.0f), ((dimslice+2) - 1.0f)); // y_clipped
            ym = (int) floor(yc); // y_minus
            yp = (int) ceil(yc);  // y_plus

            xc = fmin(fmax(x1, 0.0f), ((dimslice+2) - 1.0f));  // x_clipped
            xm = (int) floor(xc); // x_minus
            xp = (int) ceil(xc);  // x_plus

            if ((ym == yp) && (xm == xp)) val = d_slice[ym*(dimslice+2) + xm];
            else if (ym == yp) val = (d_slice[ym*(dimslice+2) + xm] * (xp - xc)) + (d_slice[ym*(dimslice+2) + xp] * (xc - xm));
            else if (xm == xp) val = (d_slice[ym*(dimslice+2) + xm] * (yp - yc)) + (d_slice[yp*(dimslice+2) + xm] * (yc - ym));
            else val = (d_slice[ym*(dimslice+2) + xm] * (yp - yc) * (xp - xc))
                       + (d_slice[yp*(dimslice+2) + xm] * (yc - ym) * (xp - xc))
                       + (d_slice[ym*(dimslice+2) + xp] * (yp - yc) * (xc - xm))
                       + (d_slice[yp*(dimslice+2) + xp] * (yc - ym) * (xc - xm));
            // ----------
            res += val;
            posx += shiftJ;
        }
    }
    else {
        for(int j=0; j<dimslice; j++) {  // j: Joseph
            x1 = beginA +(posx)*stlA + (j)*stlAJ+1.5f;
            x2 = beginB +(posx)*stlB + (j)*stlBJ+1.5f;
            l = (x1>=0.0f )*(x1<(dimslice+2))*( x2>=0.0f)*( x2<(dimslice+2) ) ;
            /*
              Bilinear interpolation
            */
            yc = fmin(fmax(x2, 0.0f), ((dimslice+2) - 1.0f)); // y_clipped
            ym = (int) floor(yc); // y_minus
            yp = (int) ceil(yc);  // y_plus

            xc = fmin(fmax(x1, 0.0f), ((dimslice+2) - 1.0f));  // x_clipped
            xm = (int) floor(xc); // x_minus
            xp = (int) ceil(xc);  // x_plus

            if ((ym == yp) && (xm == xp)) val = d_slice[ym*(dimslice+2) + xm];
            else if (ym == yp) val = (d_slice[ym*(dimslice+2) + xm] * (xp - xc)) + (d_slice[ym*(dimslice+2) + xp] * (xc - xm));
            else if (xm == xp) val = (d_slice[ym*(dimslice+2) + xm] * (yp - yc)) + (d_slice[yp*(dimslice+2) + xm] * (yc - ym));
            else val = (d_slice[ym*(dimslice+2) + xm] * (yp - yc) * (xp - xc))
                       + (d_slice[yp*(dimslice+2) + xm] * (yc - ym) * (xp - xc))
                       + (d_slice[ym*(dimslice+2) + xp] * (yp - yc) * (xc - xm))
                       + (d_slice[yp*(dimslice+2) + xp] * (yc - ym) * (xc - xm));
            // ----------
            res += val*l;
            posx += shiftJ;
        }
    }

    if((bidy*16 + tidy) < num_projections && (bidx*16 + tidx) < num_bins) {
        res *= Area;
        if (normalize)
            res *= M_PI_F * 0.5f / num_projections;
        d_Sino[dimrecx*(bidy*16 + tidy) + (bidx*16 + tidx)] = res;
    }
}
