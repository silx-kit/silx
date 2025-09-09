# /*##########################################################################
#
# Copyright (c) 2017-2022 European Synchrotron Radiation Facility
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

from .core import (  # noqa: F401
    Item,
    DataItem,
    LabelsMixIn,
    DraggableMixIn,
    ColormapMixIn,
    LineGapColorMixIn,
    SymbolMixIn,
    ColorMixIn,
    YAxisMixIn,
    FillMixIn,
    AlphaMixIn,
    LineMixIn,
    ScatterVisualizationMixIn,
    ComplexMixIn,
    ItemChangedType,
    PointsBase,
)
from .complex import ImageComplexData  # noqa: F401
from .curve import Curve, CurveStyle  # noqa: F401
from .histogram import Histogram  # noqa: F401
from .image import (  # noqa: F401
    ImageBase,
    ImageData,
    ImageDataBase,
    ImageRgba,
    ImageStack,
    MaskImageData,
)
from .image_aggregated import ImageDataAggregated  # noqa: F401
from .shape import Line, Shape, BoundingRect, XAxisExtent, YAxisExtent  # noqa: F401
from .scatter import Scatter  # noqa: F401
from .marker import MarkerBase, Marker, XMarker, YMarker  # noqa: F401
from .axis import Axis, XAxis, YAxis, YRightAxis  # noqa: F401

DATA_ITEMS = (
    ImageComplexData,
    Curve,
    Histogram,
    ImageBase,
    Scatter,
    BoundingRect,
    XAxisExtent,
    YAxisExtent,
)
"""Classes of items representing data and to consider to compute data bounds.
"""
