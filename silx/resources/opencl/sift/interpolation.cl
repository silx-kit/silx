/*
 *   Project: SIFT: An algorithm for image alignement
 *
 *   Copyright (C) 2013-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
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

void kernel interpolate(image3d_t volume,
                        sampler_t sampler,
                        global float* img,
                        int img_width,
                        int img_height,
                        global float* point,
                        global float3* norm)
{                       
    int pos_x = (int) get_global_id(0);
    int pos_y = (int) get_global_id(1);
    if ((pos_x>=img_width)||(pos_y>img_height))
    {
        return;
    }
    float3 n_norm = normalize(norm[0]);
    float3 u_norm, v_norm;
    float3 pos;
    float nx = n_norm.x,
          ny = n_norm.y,
          nz = n_norm.z;
    float ax = fabs(nx),
          ay = fabs(ny),
          az = fabs(nz);
    
    if ((ax>=az) && (ay>=az))       //z smallest
    {
        u_norm = (float3)( -ny, nx, 0.0f);
    }
    else if  ((ax>=ay) && (az>=ay)) //y smallest
    {
        u_norm = (float3)( -nz, 0.0f, nx);
    }
    else if  ((ay>=ax) && (az>=ax)) //x smallest
    {
        u_norm = (float3)( 0.0f, -nz, ny);
    }
    v_norm = cross(n_norm,u_norm);
    // u_norm, v_norm, n_norm is a direct orthonormal ref
    float3 tx=(float3)(u_norm.x,v_norm.x,n_norm.x);
    float3 ty=(float3)(u_norm.y,v_norm.y,n_norm.y);
    float3 tz=(float3)(u_norm.z,v_norm.z,n_norm.z);
    //transposed version
    float3 pos_uvn = (float3)(2.0f*((float)pos_x/(float)img_width)-1.0f,
                                      2.0f*((float)pos_y/(float)img_height)-1.0f,
                                      0.0f);
    float4 pos_xyz =  (float4)(dot(tx,pos_uvn)+point[0],
                               dot(ty,pos_uvn)+point[1],
                               dot(tz,pos_uvn)+point[2],
                               0.0f);
                       
    float4 res = read_imagef(volume, sampler, pos_xyz);
    img[pos_x+img_width*pos_y] = res.x;
}
