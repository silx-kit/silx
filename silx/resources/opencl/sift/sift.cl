/*
 *   Project: SIFT: An algorithm for image alignement
 *
 *   Copyright (C) 2013-2017 European Synchrotron Radiation Facility
 *                           Grenoble, France
 *
 *   Principal authors: J. Kieffer (kieffer@esrf.fr)
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*
    guess_keypoint: amplitude, row, column, scale
	actual_keypoint: column, row, scale, angle
*/


typedef struct guess_keypoint
{
	float value, row, col, scale;
} guess_keypoint;


typedef struct actual_keypoint 
{
	float col, row, scale, angle;
} actual_keypoint;

/*
    This is an unified float4 which can be seen as a pre_keypoint (guess keypoint)
*/

typedef union
{
        guess_keypoint raw;
        actual_keypoint ref;
} unified_keypoint;

/*
	Keypoint with its descriptor
*/
typedef struct featured_keypoint {
	actual_keypoint keypoint;
	unsigned char desc[128];
} featured_keypoint;
