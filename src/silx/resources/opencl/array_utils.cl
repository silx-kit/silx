/**
 *  2D Memcpy for float* arrays,
 * replacing pyopencl "enqueue_copy" which does not work for rectangular copies.
 * ALL THE SIZES/OFFSETS ARE SPECIFIED IN PIXELS, NOT IN BYTES.
 * In the (x, y) convention, x is the fast index (as in CUDA).
 *
 * :param dst: destination array
 * :param src: source array
 * :param dst_width: width of the dst array
 * :param src_width: width of the src array
 * :param dst_offset: tuple with the offset (x, y) in the dst array
 * :param src_offset: tuple with the offset (x, y) in the src array
 * :param transfer_shape: shape of the transfer array in the form (x, y)
 *
 */
kernel void cpy2d(
    global float* dst,
    global float* src,
    int dst_width,
    int src_width,
    int2 dst_offset,
    int2 src_offset,
    int2 transfer_shape)
{
    int gidx = get_global_id(0), gidy = get_global_id(1);
    if (gidx < transfer_shape.x && gidy < transfer_shape.y) {
        dst[(dst_offset.y + gidy)*dst_width + (dst_offset.x + gidx)] = src[(src_offset.y + gidy)*src_width + (src_offset.x + gidx)];
    }
}


// Looks like cfloat_t and cfloat_mul are not working, yet specified in
// pyopencl documentation. Here we are using float2 as in all available examples
// #include <pyopencl-complex.h>
// typedef cfloat_t complex;

static inline float2 complex_mul(float2 a, float2 b) {
    float2 res = (float2) (0, 0);
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.y * b.x + a.x * b.y;
    return res;
}

// arr2D *= arr1D (line by line, i.e along fast dim)
kernel void inplace_complex_mul_2Dby1D(
    global float2* arr2D,
    global float2* arr1D,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if ((x >= width) || (y >= height)) return;
    int i = y*width + x;
    arr2D[i] = complex_mul(arr2D[i], arr1D[x]);
}


// arr3D *= arr1D (along fast dim)
kernel void inplace_complex_mul_3Dby1D(
    global float2* arr3D,
    global float2* arr1D,
    int width,
    int height,
    int depth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if ((x >= width) || (y >= height) || (z >= depth)) return;
    int i = (z*height + y)*width + x;
    arr3D[i] = complex_mul(arr3D[i], arr1D[x]);
}
