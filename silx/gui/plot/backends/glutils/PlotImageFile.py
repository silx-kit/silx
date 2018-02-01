# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2017 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Function to save an image to a file."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import base64
import struct
import sys
import zlib


# Image writer ################################################################

def convertRGBDataToPNG(data):
    """Convert a RGB bitmap to PNG.

    It only supports RGB bitmap with one byte per channel stored as a 3D array.
    See `Definitive Guide <http://www.libpng.org/pub/png/book/>`_ and
    `Specification <http://www.libpng.org/pub/png/spec/1.2/>`_ for details.

    :param data: A 3D array (h, w, rgb) storing an RGB image
    :type data: numpy.ndarray of unsigned bytes
    :returns: The PNG encoded data
    :rtype: bytes
    """
    height, width = data.shape[0], data.shape[1]
    depth = 8  # 8 bit per channel
    colorType = 2  # 'truecolor' = RGB
    interlace = 0  # No

    IHDRdata = struct.pack(">ccccIIBBBBB", b'I', b'H', b'D', b'R',
                           width, height, depth, colorType,
                           0, 0, interlace)

    # Add filter 'None' before each scanline
    preparedData = b'\x00' + b'\x00'.join(line.tostring() for line in data)
    compressedData = zlib.compress(preparedData, 8)

    IDATdata = struct.pack("cccc", b'I', b'D', b'A', b'T')
    IDATdata += compressedData

    return b''.join([
        b'\x89PNG\r\n\x1a\n',  # PNG signature
        # IHDR chunk: Image Header
        struct.pack(">I", 13),  # length
        IHDRdata,
        struct.pack(">I", zlib.crc32(IHDRdata) & 0xffffffff),  # CRC
        # IDAT chunk: Payload
        struct.pack(">I", len(compressedData)),
        IDATdata,
        struct.pack(">I", zlib.crc32(IDATdata) & 0xffffffff),  # CRC
        b'\x00\x00\x00\x00IEND\xaeB`\x82'  # IEND chunk: footer
    ])


def saveImageToFile(data, fileNameOrObj, fileFormat):
    """Save a RGB image to a file.

    :param data: A 3D array (h, w, 3) storing an RGB image.
    :type data: numpy.ndarray with of unsigned bytes.
    :param fileNameOrObj: Filename or object to use to write the image.
    :type fileNameOrObj: A str or a 'file-like' object with a 'write' method.
    :param str fileFormat: The type of the file in: 'png', 'ppm', 'svg', 'tiff'.
    """
    assert len(data.shape) == 3
    assert data.shape[2] == 3
    assert fileFormat in ('png', 'ppm', 'svg', 'tiff')

    if not hasattr(fileNameOrObj, 'write'):
        if sys.version_info < (3, ):
            fileObj = open(fileNameOrObj, "wb")
        else:
            if fileFormat in ('png', 'ppm', 'tiff'):
                # Open in binary mode
                fileObj = open(fileNameOrObj, 'wb')
            else:
                fileObj = open(fileNameOrObj, 'w', newline='')
    else:  # Use as a file-like object
        fileObj = fileNameOrObj

    if fileFormat == 'svg':
        height, width = data.shape[:2]
        base64Data = base64.b64encode(convertRGBDataToPNG(data))

        fileObj.write(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        fileObj.write('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n')
        fileObj.write(
            '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')
        fileObj.write('<svg xmlns:xlink="http://www.w3.org/1999/xlink"\n')
        fileObj.write('     xmlns="http://www.w3.org/2000/svg"\n')
        fileObj.write('     version="1.1"\n')
        fileObj.write('     width="%d"\n' % width)
        fileObj.write('     height="%d">\n' % height)
        fileObj.write('    <image xlink:href="data:image/png;base64,')
        fileObj.write(base64Data.decode('ascii'))
        fileObj.write('"\n')
        fileObj.write('           x="0"\n')
        fileObj.write('           y="0"\n')
        fileObj.write('           width="%d"\n' % width)
        fileObj.write('           height="%d"\n' % height)
        fileObj.write('           id="image" />\n')
        fileObj.write('</svg>')

    elif fileFormat == 'ppm':
        height, width = data.shape[:2]

        fileObj.write(b'P6\n')
        fileObj.write(b'%d %d\n' % (width, height))
        fileObj.write(b'255\n')
        fileObj.write(data.tostring())

    elif fileFormat == 'png':
        fileObj.write(convertRGBDataToPNG(data))

    elif fileFormat == 'tiff':
        if fileObj == fileNameOrObj:
            raise NotImplementedError(
                'Save TIFF to a file-like object not implemented')

        from silx.third_party.TiffIO import TiffIO

        tif = TiffIO(fileNameOrObj, mode='wb+')
        tif.writeImage(data, info={'Title': 'OpenGL Plot Snapshot'})

    if fileObj != fileNameOrObj:
        fileObj.close()
