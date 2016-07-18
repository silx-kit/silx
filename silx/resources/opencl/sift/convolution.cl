/*
    Separate convolution with global memory access
    The borders are handled directly in the kernel (by symetrization), 
    so the input image does not need to be pre-processed

*/

#define MAX_CONST_SIZE 16384



__kernel void horizontal_convolution(
    const __global float * input,
    __global float * output,
    __constant float * filter __attribute__((max_constant_size(MAX_CONST_SIZE))),
    int FILTER_SIZE,
    int IMAGE_W,
    int IMAGE_H
)
{
    int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);

    int HALF_FILTER_SIZE = (FILTER_SIZE % 2 == 1 ? (FILTER_SIZE)/2 : (FILTER_SIZE+1)/2);

    if (gid1 < IMAGE_H && gid0 < IMAGE_W) {

//        int pos = gid0* IMAGE_W + gid1;
        int pos = gid1*IMAGE_W + gid0;
        int fIndex = 0;
        float sum = 0.0f;
        int c = 0;
        int newpos = 0;
        int debug=0;


        for (c = -HALF_FILTER_SIZE ; c < FILTER_SIZE-HALF_FILTER_SIZE ; c++) {

            newpos = pos + c;
            if (gid0 + c < 0) {
                //debug=1;
                newpos= pos - 2*gid0 - c - 1;
            }

            else if (gid0 + c > IMAGE_W -1 ) {
                newpos= (gid1+2)*IMAGE_W - gid0 -c -1;
                //newpos= pos - c+1; //newpos - 2*c;
                //debug = 1;
            }
            sum += input[ newpos ] * filter[ fIndex  ];

            fIndex += 1;

        }

        output[pos]=sum;
    }
}









__kernel void vertical_convolution(
    const __global float * input,
    __global float * output,
    __constant float * filter __attribute__((max_constant_size(MAX_CONST_SIZE))),
    int FILTER_SIZE,
    int IMAGE_W,
    int IMAGE_H
)
{

    int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);


    if (gid1 < IMAGE_H && gid0 < IMAGE_W) {

        int HALF_FILTER_SIZE = (FILTER_SIZE % 2 == 1 ? (FILTER_SIZE)/2 : (FILTER_SIZE+1)/2);

//        int pos = gid0 * IMAGE_W + gid1;
        int pos = gid1 * IMAGE_W + gid0;
        int fIndex = 0;
        float sum = 0.0f;
        int r = 0,newpos=0;
        int debug=0;

        for (r = -HALF_FILTER_SIZE ; r < FILTER_SIZE-HALF_FILTER_SIZE ; r++) {
            newpos = pos + r * (IMAGE_W);

            if (gid1+r < 0) {
                newpos = gid0 -(r+1)*IMAGE_W - gid1*IMAGE_W;
                //debug=1;
            }
            else if (gid1+r > IMAGE_H -1) {
                newpos= (IMAGE_H-1)*IMAGE_W + gid0 + (IMAGE_H - r)*IMAGE_W - gid1*IMAGE_W;
            }
            sum += input[ newpos ] * filter[ fIndex   ];
            fIndex += 1;

        }
        output[pos]=sum;
        if (debug == 1) output[pos]=0;
    }
}



/*
*/




