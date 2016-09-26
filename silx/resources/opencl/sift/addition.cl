__kernel void addition(__global float* a, __global float* b, __global float* c)
{
    unsigned int i = get_global_id(0);
    c[i] = a[i] + b[i];
}