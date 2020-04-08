# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
"""This package provides classes that describes :class:`.PlotWidget` content.

Instances of those classes are returned by :class:`.PlotWidget` methods that give
access to its content such as :meth:`.PlotWidget.getCurve`, :meth:`.PlotWidget.getImage`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/06/2017"

from .core import (Item, LabelsMixIn, DraggableMixIn, ColormapMixIn,  # noqa
                   SymbolMixIn, ColorMixIn, YAxisMixIn, FillMixIn,  # noqa
                   AlphaMixIn, LineMixIn, ScatterVisualizationMixIn,  # noqa
                   ComplexMixIn, ItemChangedType, PointsBase)  # noqa
from .complex import ImageComplexData  # noqa
from .curve import Curve, CurveStyle  # noqa
from .histogram import Histogram  # noqa
from .image import ImageBase, ImageData, ImageRgba, ImageStack, MaskImageData  # noqa
from .shape import Shape, BoundingRect, XAxisExtent, YAxisExtent  # noqa
from .scatter import Scatter  # noqa
from .marker import MarkerBase, Marker, XMarker, YMarker  # noqa
from .axis import Axis, XAxis, YAxis, YRightAxis

DATA_ITEMS = (ImageComplexData, Curve, Histogram, ImageBase, Scatter,
              BoundingRect, XAxisExtent, YAxisExtent)
"""Classes of items representing data and to consider to compute data bounds.
"""
