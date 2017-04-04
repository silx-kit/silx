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
"""This module provides convenient classes for the OpenGL rendering backend.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import logging
from ...._glutils import FramebufferTexture, Texture, VertexBuffer


_logger = logging.getLogger(__name__)


# TODO remove when fully tested
# Monkey-patching

def _VertexBuffer_del(self):
    if self._name is not None:
        _logger.error('Discarding GL resources not yet freed')
        self.discard()

VertexBuffer.__del__ = _VertexBuffer_del


def _FramebufferTexture_del(self):
    if self._name is not None:
        _logger.error('Discarding GL resources not yet freed')
        self.discard()

FramebufferTexture.__del__ = _FramebufferTexture_del


def _Texture_del(self):
    if self._name is not None:
        _logger.error('Discarding GL resources not yet freed')
        self.discard()

Texture.__del__ = _Texture_del


from .GLPlotCurve import *  # noqa
from .GLPlotFrame import *  # noqa
from .GLPlotImage import *  # noqa
from .GLSupport import *  # noqa
from .GLText import *  # noqa
from .GLTexture import *  # noqa
