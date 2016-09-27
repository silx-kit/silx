__kernel void addition(__global float* a, __global float* b, __global float* res, int N)
{
    unsigned int i = get_global_id(0);
    if( i<N ){
        res[i] = a[i] + b[i];
    }
}