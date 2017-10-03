/**
 *  2D Memcpy for float* arrays,
 * replacing pyopencl "enqueue_copy" which does not return the expected result
 * when dealing with rectangular buffers.
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
    int gidx = get_global_id(0),
        gidy = get_global_id(1);
    if (gidx < transfer_shape.x && gidy < transfer_shape.y)
    {
        dst[(dst_offset.y + gidy)*dst_width + (dst_offset.x + gidx)] = src[(src_offset.y + gidy)*src_width + (src_offset.x + gidx)];
    }
}

