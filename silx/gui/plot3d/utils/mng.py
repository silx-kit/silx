# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""This module provides basic writing Mulitple-image Network Graphics files.

It only supports RGB888 images of the same shape stored as
MNG-VLC (very low complexity) format.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/12/2016"


import logging
import struct
import zlib

import numpy

_logger = logging.getLogger(__name__)


def _png_chunk(name, data):
    """Return a PNG chunk

    :param str name: Chunk type
    :param byte data: Chunk payload
    """
    length = struct.pack('>I', len(data))
    name = [char.encode('ascii') for char in name]
    chunk = struct.pack('cccc', *name) + data
    crc = struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)
    return length + chunk + crc


def convert(images, nb_images=0, fps=25):
    """Convert RGB images to MNG-VLC format.

    See http://www.libpng.org/pub/mng/spec/
    See http://www.libpng.org/pub/png/book/
    See http://www.libpng.org/pub/png/spec/1.2/

    :param images: iterator of RGB888 images
    :type images: iterator of numpy.ndarray of dimension 3
    :param int nb_images: The number of images indicated in the MNG header
    :param int fps: The frame rate indicated in the MNG header
    :return: An iterator of MNG chunks as bytes
    """
    first_image = True

    for image in images:
        if first_image:
            first_image = False

            height, width = image.shape[:2]

            # MNG signature
            yield b'\x8aMNG\r\n\x1a\n'

            # MHDR chunk: File header
            yield _png_chunk('MHDR', struct.pack(
                ">IIIIIII",
                width,
                height,
                fps,  # ticks
                nb_images + 1,  # layer count
                nb_images,  # frame count
                nb_images,  # play time
                1))  # profile: MNG-VLC no alpha: only least significant bit 1

        assert image.shape == (height, width, 3)
        assert image.dtype == numpy.dtype('uint8')

        # IHDR chunk: Image header
        depth = 8  # 8 bit per channel
        color_type = 2  # 'truecolor' = RGB
        interlace = 0  # No
        yield _png_chunk('IHDR', struct.pack(">IIBBBBB",
                                             width,
                                             height,
                                             depth,
                                             color_type,
                                             0, 0, interlace))

        # Add filter 'None' before each scanline
        prepared_data = b'\x00' + b'\x00'.join(
            line.tostring() for line in image)  # TODO optimize that
        compressed_data = zlib.compress(prepared_data, 8)

        # IDAT chunk: Payload
        yield _png_chunk('IDAT', compressed_data)

        # IEND chunk: Image footer
        yield _png_chunk('IEND', b'')

    # MEND chunk: footer
    yield _png_chunk('MEND', b'')
