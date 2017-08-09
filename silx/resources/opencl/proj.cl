const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void  forward_kernel(
        __global float *d_Sino,
        __read_only image2d_t d_slice,
        int dimslice,
        int num_bins,
        __global float* angles_per_project ,
        float axis_position,
        __global float *d_axis_corrections,
        __global int* d_beginPos    ,
        __global int* d_strideJoseph,
        __global int* d_strideLine  ,
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

    __local float corrections[16];
    __local int beginPos[16*2];
    __local int strideJoseph[16*2];
    __local int strideLine[16*2];

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
        if (normalize) res *= M_PI*0.5/num_projections;
        d_Sino[dimrecx*(bidy*16 + tidy) + (bidx*16 + tidx)] = res;
    }
}

