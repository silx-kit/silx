kernel void densify_csr(
    const global float* data,
    const global int* ind,
    const global int* iptr,
    global float* output,
    int image_height
)
{
    uint tid = get_local_id(0);
    uint row_idx = get_global_id(1);
    if ((tid >= IMAGE_WIDTH) || (row_idx >= image_height)) return;

    local float line[IMAGE_WIDTH];

    // Memset
    //~ #pragma unroll
    for (int k = 0; tid+k < IMAGE_WIDTH; k += get_local_size(0)) {
        if (tid+k >= IMAGE_WIDTH) break;
        line[tid+k] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    uint start = iptr[row_idx], end = iptr[row_idx+1];
    //~ #pragma unroll
    for (int k = start; k < end; k += get_local_size(0)) {
        // Current work group handles one line of the final array
        // on the current line, write data[start+tid] at column index ind[start+tid]
        if (k+tid < end)
            line[ind[k+tid]] = data[k+tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write the current line (shared mem) into the output array (global mem)
    //~ #pragma unroll
    for (int k = 0; tid+k < IMAGE_WIDTH; k += get_local_size(0)) {
        output[row_idx*IMAGE_WIDTH + tid+k] = line[tid+k];
        if (k+tid >= IMAGE_WIDTH) return;
    }
}
