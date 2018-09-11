/*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
// __authors__ = ["H. Payno"]
// __license__ = "MIT"
// __date__ = "10/02/2017"

#ifndef MEDIAN_FILTER
#define MEDIAN_FILTER

#include <vector>
#include <assert.h>
#include <algorithm>
#include <signal.h>
#include <iostream>
#include <cmath>
#include <cfloat>

/* Needed for pytohn2.7 on Windows... */
#ifndef INFINITY
#define INFINITY (DBL_MAX+DBL_MAX)
#endif

#ifndef NAN
#define NAN (INFINITY-INFINITY)
#endif

// Modes for the median filter
enum MODE{
    NEAREST=0,  
    REFLECT=1,
    MIRROR=2,
    SHRINK=3,
    CONSTANT=4,
};

// Simple function browsing a deque and registering the min and max values
// and if those values are unique or not
template<typename T>
void getMinMax(std::vector<T>& v, T& min, T&max,
    typename std::vector<T>::const_iterator end){
    // init min and max values
    typename std::vector<T>::const_iterator it = v.begin();
    if (v.size() == 0){
        raise(SIGINT);
    }else{
        min = max = *it;
    }
    it++;

    // Browse all the deque
    while(it!=end){
        // check if repeated (should always be before min/max setting)
        T value = *it;
        if(value > max) max = value;
        if(value < min) min = value;

        it++;
    }
}


// apply the median filter only on limited part of the vector
// In case of even number of elements (either due to NaNs in the window
// or for image borders in shrink mode):
// the highest of the 2 central values is returned
template<typename T>
inline T median(std::vector<T>& v, int window_size) {
    int pivot = window_size / 2;
    std::nth_element(v.begin(), v.begin() + pivot, v.begin()+window_size);
    return v[pivot];
}


// return the index into 0, (length_max - 1) in reflect mode
inline int reflect(int index, int length_max){
    int res = index;
    // if the index is negative get the positive symmetrical value
    if(res < 0){
        res += 1;
        res = -res;
    }
    // then apply the reflect algorithm. Frequency is 2 max length
    res = res % (2*length_max);
    if(res >= length_max){
        res = 2*length_max - res -1;
        res = res % length_max;
    }
    return res;
}

// return the index into 0, (length_max - 1) in mirror mode
inline int mirror(int index, int length_max){
    int res = index;
    // if the index is negative get the positive symmetrical value
    if(res < 0){
        res = -res;
    }
    int rightLimit = length_max -1;
    // apply the redundancy each two right limit
    res = res % (2*rightLimit);
    if(res >= length_max){
        int distToRedundancy = (2*rightLimit) - res;
        res = distToRedundancy;
    }
    return res;
}

// Browse the column of pixel_x
template<typename T>
void median_filter(
    const T* input,
    T* output,
    int* kernel_dim,        // two values : 0:width, 1:height
    int* image_dim,         // two values : 0:width, 1:height
    int y_pixel,            // the x pixel to process
    int x_pixel_range_min,
    int x_pixel_range_max,
    bool conditional,
    int pMode,
    T cval) {
    
    assert(kernel_dim[0] > 0);
    assert(kernel_dim[1] > 0);
    assert(y_pixel >= 0);
    assert(image_dim[0] > 0);
    assert(image_dim[1] > 0);
    assert(y_pixel >= 0);
    assert(y_pixel < image_dim[0]);
    assert(x_pixel_range_max < image_dim[1]);
    assert(x_pixel_range_min <= x_pixel_range_max);
    // kernel odd assertion
    assert((kernel_dim[0] - 1)%2 == 0);
    assert((kernel_dim[1] - 1)%2 == 0);

    // # this should be move up to avoid calculation each time
    int halfKernel_x = (kernel_dim[1] - 1) / 2;
    int halfKernel_y = (kernel_dim[0] - 1) / 2;

    MODE mode = static_cast<MODE>(pMode);

    // init buffer
    std::vector<T> window_values(kernel_dim[0]*kernel_dim[1]);

    bool not_horizontal_border = (y_pixel >= halfKernel_y && y_pixel < image_dim[0] - halfKernel_y);

    for(int x_pixel=x_pixel_range_min; x_pixel <= x_pixel_range_max; x_pixel ++ ){
        typename std::vector<T>::iterator it = window_values.begin();
        // fill the vector

        if (not_horizontal_border &&
            x_pixel >= halfKernel_x && x_pixel < image_dim[1] - halfKernel_x) {
            //This is not a border, just fill it
            for(int win_y=y_pixel-halfKernel_y; win_y<= y_pixel+halfKernel_y; win_y++) {
                for(int win_x = x_pixel-halfKernel_x; win_x <= x_pixel+halfKernel_x; win_x++){
                    T value = input[win_y*image_dim[1] + win_x];
                    if (value == value) {  // Ignore NaNs
                        *it = value;
                        ++it;
                    }
                }
            }

        } else { // This is a border, handle the special case
            for(int win_y=y_pixel-halfKernel_y; win_y<= y_pixel+halfKernel_y; win_y++)
            {
                for(int win_x = x_pixel-halfKernel_x; win_x <= x_pixel+halfKernel_x; win_x++)
                {
                    T value = 0;
                    int index_x = win_x;
                    int index_y = win_y;

                    switch(mode){
                        case NEAREST:
                            index_x = std::min(std::max(win_x, 0), image_dim[1] - 1);
                            index_y = std::min(std::max(win_y, 0), image_dim[0] - 1);
                            value = input[index_y*image_dim[1] + index_x];
                            break;

                        case REFLECT:
                            index_x = reflect(win_x, image_dim[1]);
                            index_y = reflect(win_y, image_dim[0]);
                            value = input[index_y*image_dim[1] + index_x];
                            break;

                        case MIRROR:
                            index_x = mirror(win_x, image_dim[1]);
                            // deal with 1d case
                            if(win_y == 0 && image_dim[0] == 1){
                                index_y = 0;
                            }else{
                                index_y = mirror(win_y, image_dim[0]);
                            }
                            value = input[index_y*image_dim[1] + index_x];
                            break;

                        case SHRINK:
                            if ((index_x < 0) || (index_x > image_dim[1] -1) ||
                                (index_y < 0) || (index_y > image_dim[0] -1)) {
                                continue;
                            }
                            value = input[index_y*image_dim[1] + index_x];
                            break;
                        case CONSTANT:
                            if ((index_x < 0) || (index_x > image_dim[1] -1) ||
                                (index_y < 0) || (index_y > image_dim[0] -1)) {
                                value = cval;
                            } else {
                                value = input[index_y*image_dim[1] + index_x];
                            }
                            break;
                    }

                    if (value == value) {  // Ignore NaNs
                        *it = value;
                        ++it;
                    }
                }
            }
        }

        //window_size can be smaller than kernel size in shrink mode or if there is NaNs
        int window_size = std::distance(window_values.begin(), it);

        if (window_size == 0) {
            // Window is empty, this is the case when all values are NaNs
            output[image_dim[1]*y_pixel + x_pixel] = NAN;

        } else {
            // apply the median value if needed for this pixel
            const T currentPixelValue = input[image_dim[1]*y_pixel + x_pixel];
            if (conditional == true){
                typename std::vector<T>::iterator window_end = window_values.begin() + window_size;
                T min = 0;
                T max = 0;
                getMinMax(window_values, min, max, window_end);
                // NaNs are propagated through unchanged
                if ((currentPixelValue == max) || (currentPixelValue == min)){
                    output[image_dim[1]*y_pixel + x_pixel] = median<T>(window_values, window_size);
                }else{
                    output[image_dim[1]*y_pixel + x_pixel] = currentPixelValue;
                }
            }else{
                output[image_dim[1]*y_pixel + x_pixel] = median<T>(window_values, window_size);
            }
        }
    }
}

#endif // MEDIAN_FILTER
