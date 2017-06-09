/*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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

// Modes for the median filter
enum MODE{
    NEAREST=0,  
    REFLECT=1,
    MIRROR=2,
    SHRINK=3
};

// Simple function browsing a deque and registring the min and max values
// and if those values are unique or not
template<typename T>
void getMinMax(std::vector<const T*>& v, T& min, T&max,
    typename std::vector<const T*>::const_iterator end){
    // init min and max values
    typename std::vector<const T*>::const_iterator it = v.begin();
    if (v.size() == 0){
        raise(SIGINT);
    }else{
        min = max = *(*it);
    }
    it++;

    // Browse all the deque
    while(it!=end){
        // check if repeated (should always be before min/max setting)
        if(*(*it) > max) max = *(*it);
        if(*(*it) < min) min = *(*it);

        it++;
    }
}

template<typename T>
bool cmp(const T* a, const T* b){
    return *a < *b;
}

// apply the median filter only on limited part of the vector
template<typename T>
const T* median(std::vector<const T*>& v, int window_size) {
    std::nth_element(v.begin(), v.begin() + window_size/2, v.begin()+window_size, cmp<T>);
    return v[window_size/2];
}

template<typename T>
void print_window(std::vector<const T*>& v,
    typename std::vector<const T*>::const_iterator end){
    typename std::vector<const T*>::const_iterator it;
    for(it = v.begin(); it != end; ++it){
        std::cout << *(*it) << " ";
    }
    std::cout << std::endl;
}

// return the index into 0, (length_max - 1) in reflect mode
int reflect(int index, int length_max){
    int res = index;
    // if the index is negative get the positive symetrical value
    if(res < 0){
        res += 1;
        res = -res;
    }
    // then apply the reflect algorithm. Frequence is 2 max length
    res = res % (2*length_max);
    if(res >= length_max){
        res = 2*length_max - res -1;
        res = res % length_max;
    }
    return res;
}

// return the index into 0, (length_max - 1) in mirror mode
int mirror(int index, int length_max){
    int res = index;
    // if the index is negative get the positive symetrical value
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
    int pMode){
    
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
    std::vector<const T*> window_values(kernel_dim[0]*kernel_dim[1]);

    for(int x_pixel=x_pixel_range_min; x_pixel <= x_pixel_range_max; x_pixel ++ ){
        typename std::vector<const T*>::iterator it = window_values.begin();
        // fill the vector
        for(int win_y=y_pixel-halfKernel_y; win_y<= y_pixel+halfKernel_y; win_y++)
        {
            for(int win_x = x_pixel-halfKernel_x; win_x <= x_pixel+halfKernel_x; win_x++)
            {
                int index_x = win_x;
                int index_y = win_y;
                switch(mode){
                    case NEAREST:
                        index_x = std::min(std::max(win_x, 0), image_dim[1] - 1);
                        index_y = std::min(std::max(win_y, 0), image_dim[0] - 1);
                        break;

                    case REFLECT:
                        index_x = reflect(win_x, image_dim[1]);
                        index_y = reflect(win_y, image_dim[0]);
                        break;

                    case MIRROR:
                        index_x = mirror(win_x, image_dim[1]);
                        index_y = mirror(win_y, image_dim[0]);
                        break;
                    case SHRINK:
                        if((index_x < 0) || (index_x > image_dim[1] -1)){
                            continue;
                        }
                        if((index_y < 0) || (index_y > image_dim[0] -1)){
                            continue;
                        }
                        break;
                }
                *it = (&input[index_y*image_dim[1] + index_x]);
                ++it;
            }
        }

        // get end of the windows. This is needed since in shrink mode we are
        // not sure to fill the entire window.
        typename std::vector<const T*>::iterator window_end;
        int window_size = kernel_dim[0]*kernel_dim[1];
        if(mode == SHRINK){
            int x_shrink_ker_dim = std::min(x_pixel+halfKernel_x, image_dim[1]-1) - std::max(0, x_pixel-halfKernel_x)+1;
            int y_shrink_ker_dim = std::min(y_pixel+halfKernel_y, image_dim[0]-1) - std::max(0, y_pixel-halfKernel_y)+1;
            window_size = x_shrink_ker_dim*y_shrink_ker_dim;
            window_end = window_values.begin() + window_size;
        }else{
            window_end = window_values.end();
        }

        // apply the median value if needed for this pixel
        const T* currentPixelValue = &input[image_dim[1]*y_pixel + x_pixel];
        if (conditional == true){
            T min = 0;
            T max = 0;
            getMinMax(window_values, min, max, window_end);
            if ((*currentPixelValue == max) || (*currentPixelValue == min)){
                output[image_dim[1]*y_pixel + x_pixel] = *(median<T>(window_values, window_size));
            }else{
                output[image_dim[1]*y_pixel + x_pixel] = *currentPixelValue;
            }
        }else{
            output[image_dim[1]*y_pixel + x_pixel] = *(median<T>(window_values, window_size));
        }
    }
}

#endif // MEDIAN_FILTER
