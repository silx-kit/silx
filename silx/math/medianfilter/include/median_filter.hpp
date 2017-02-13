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

#include <kth_smallest.hpp>
#include <vector>
#include <assert.h>


template<typename T>
void median_filter(
    const T* input, 
    T* output,
    int* kernel_dim,        // two values : 0:width, 1:height
    int* image_dim,         // two values : 0:width, 1:height
    int x_pixel_range_min,
    int x_pixel_range_max,
    int y_pixel_range_min,
    int y_pixel_range_max,
    bool conditioannal){
    // Browse all pixels in the range of (pixel_x_min, pixel_y_min) to 
    // (pixel_x_min, pixel_y_max)
    
    assert(kernel_dim[0] > 0);
    assert(kernel_dim[1] > 0);
    assert(x_pixel_range_min >= 0);
    assert(y_pixel_range_min >= 0);
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
    int halfKernel_x = (kernel_dim[0] - 1) / 2;
    int halfKernel_y = (kernel_dim[1] - 1) / 2;

    for(int pixel_x=x_pixel_range_min; pixel_x <= x_pixel_range_max; pixel_x ++ ){
        for(int pixel_y=y_pixel_range_min; pixel_y <= y_pixel_range_max; pixel_y ++ ){
            // define the window size
            int xmin = std::max(0, pixel_x-halfKernel_x);
            int xmax = std::min(image_dim[0]-1, pixel_x+halfKernel_x);

            int ymin = std::max(0, pixel_y-halfKernel_y);
            int ymax = std::min(image_dim[1]-1, pixel_y+halfKernel_y);

            // faire un set pour ordonner les valeurs
            std::vector<const T*> window_values;

            for(int win_x = xmin; win_x <= xmax; win_x++)
            {
                for(int win_y=ymin; win_y<= ymax; win_y++)
                {
                    window_values.push_back(&input[win_y*image_dim[0] + win_x]);
                }
            }
            const T* currentPixelValue = &input[image_dim[0]*pixel_y + pixel_x];
            // change value for the median, only if we don't intend to use the 
            // conditionnal or if the value of the pixel is one of the extrema
            // of the pixel value
            if (conditioannal == true){
                T min = 0;
                T max = 0;
                getMinMax(window_values, min, max);
                if ((*currentPixelValue == max) || (*currentPixelValue == min)){
                    output[image_dim[0]*pixel_y + pixel_x] = *(medianwirth<T>(window_values));
                }else{
                    output[image_dim[0]*pixel_y + pixel_x] = *currentPixelValue;
                }
            }else{
                output[image_dim[0]*pixel_y + pixel_x] = *(medianwirth<T>(window_values));
            }
        }
    }
}

#endif // MEDIAN_FILTER