# coding: utf-8
# /*##########################################################################
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
"""This package provides classes that describes :class:`.Plot` content.

Instances of those classes are returned by :class:`.Plot` methods that give
access to its content such as :meth:`.Plot.getCurve`, :meth:`.Plot.getImage`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2017"

from .core import (Item, LabelsMixIn, DraggableMixIn, ColormapMixIn,  # noqa
                   SymbolMixIn, ColorMixIn, YAxisMixIn, FillMixIn,  # noqa
                   AlphaMixIn, LineMixIn)  # noqa
from .curve import Curve  # noqa
from .histogram import Histogram  # noqa
from .image import ImageBase, ImageData, ImageRgba  # noqa
from .shape import Shape  # noqa
from .scatter import Scatter  # noqa
from .marker import Marker, XMarker, YMarker  # noqa
