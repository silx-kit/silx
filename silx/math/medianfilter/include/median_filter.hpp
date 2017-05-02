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

#include <deque>
#include <assert.h>
#include <algorithm>
#include <signal.h>

// Simple function browsing a deque and registring the min and max values
// and if those values are unique or not
template<typename T>
void getMinMax(std::deque<const T*>& v, T& min, T&max){
    // init min and max values
    typename std::deque<const T*>::const_iterator it = v.begin();
    if (v.size() == 0){
        raise(SIGINT);
    }else{
        min = max = *(*it);
    }
    it++;

    // Browse all the deque
    while(it!=v.end()){
        // check if repeated (should always be before min/max setting)
        if(*(*it) > max) max = *(*it);
        if(*(*it) < min) min = *(*it);

        it++;
    }
    // std::cout << " get min ax end" << std::endl;
}


template<typename T>
bool cmp(const T* a, const T* b){
    return *a < *b;
}

template<typename T>
const T* median(std::deque<const T*>& v) {
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end(), cmp<T>);
    return v[v.size()/2];
}

// Browse all pixels in the range of (pixel_x_min, pixel_y_min) to 
// (pixel_x_min, pixel_y_max)
template<typename T>
void median_filter(
    const T* input, 
    T* output,
    int* kernel_dim,        // two values : 0:width, 1:height
    int* image_dim,         // two values : 0:width, 1:height
    int x_pixel,            // the x pixel to process
    int y_pixel_range_min,
    int y_pixel_range_max,
    bool conditional){
    
    assert(kernel_dim[0] > 0);
    assert(kernel_dim[1] > 0);
    assert(x_pixel >= 0);
    assert(image_dim[0] > 0);
    assert(image_dim[1] > 0);
    assert(x_pixel_range_max < image_dim[0]);
    assert(y_pixel_range_max < image_dim[1]);
    assert(x_pixel_range_min <= x_pixel_range_max);
    assert(y_pixel_range_min <= y_pixel_range_max);
    // # kernel odd
    assert((kernel_dim[0] - 1)%2 == 0);
    assert((kernel_dim[1] - 1)%2 == 0);

    // # this should be move up to avoid calculation each time
    int halfKernel_x = (kernel_dim[1] - 1) / 2;
    int halfKernel_y = (kernel_dim[0] - 1) / 2;

    // init buffer
    // fill the buffer for the first iteration
    // we are treating
    // TODO : reserve size
    std::deque<const T*> window_values;

    for(int win_x = x_pixel-halfKernel_x; win_x <= x_pixel+halfKernel_x; win_x++)
    {
        // -1 to let the next iteration fill the buffer
        for(int win_y=-halfKernel_y; win_y<= halfKernel_y-1; win_y++)
        {
            int index_x = std::min(std::max(win_x, 0), image_dim[0] - 1);
            int index_y = std::min(std::max(win_y, 0), image_dim[1] - 1);
            window_values.push_back(&input[index_y*image_dim[0] + index_x]);
        }
    }

    for(int pixel_y=y_pixel_range_min; pixel_y <= y_pixel_range_max; pixel_y ++ ){
        // deque containing all the pixels in the neighbourhood
        int y_to_add = std::min(std::max(pixel_y + halfKernel_y, 0), image_dim[1] -1);
        
        // add new values to the buffer
        for(int win_x=x_pixel-halfKernel_x; win_x <= x_pixel+halfKernel_x; win_x++)
        {
            int index_x = std::min(std::max(win_x, 0), image_dim[0] -1);
            window_values.push_back(&input[y_to_add*image_dim[0] + index_x]);
        }

        const T* currentPixelValue = &input[image_dim[0]*pixel_y + x_pixel];
        // change value for the median, only if we don't intend to use the 
        // conditional or if the value of the pixel is one of the extrema
        // of the pixel value
        if (conditional == true){
            T min = 0;
            T max = 0;
            getMinMax(window_values, min, max);
            // In conditional point we are only setting the value to the pixel
            // if the value is the min or max and unique
            if ((*currentPixelValue == max) || (*currentPixelValue == min)){
                output[image_dim[0]*pixel_y + x_pixel] = *(median<T>(window_values));
            }else{
                output[image_dim[0]*pixel_y + x_pixel] = *currentPixelValue;
            }
        }else{
            output[image_dim[0]*pixel_y + x_pixel] = *(median<T>(window_values));
        }

        // remove the last line one the buffer
        for(int win_x=x_pixel-halfKernel_x; win_x <= x_pixel+halfKernel_x; win_x++)
        {
            window_values.pop_front();
        }
    }
}

#endif // MEDIAN_FILTER
