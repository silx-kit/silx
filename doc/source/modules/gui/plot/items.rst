
.. currentmodule:: silx.gui.plot

:mod:`items`: Plot primitives
=============================

.. automodule:: silx.gui.plot.items

.. currentmodule:: silx.gui.plot.items

Item
----

All plot primitives inherits from :class:`Item` as a common ground:

.. autoclass:: Item
   :show-inheritance:
   :members:

Curve
-----

.. autoclass:: Curve
   :members: getData, getXData, getYData, getXErrorData, getYErrorData, setData,
             getSymbol, setSymbol, getSymbolSize, setSymbolSize,
             getAlpha, setAlpha,
             getColor, setColor,
             getYAxis, setYAxis,
             isFill, setFill,
             getXLabel, getYLabel,
             getLineWidth, setLineWidth, getLineStyle, setLineStyle,
             isHighlighted, setHighlighted, getHighlightedStyle, setHighlightedStyle,
             getCurrentStyle

.. autoclass:: CurveStyle
   :members: getColor, getLineStyle, getLineWidth, getSymbol, getSymbolSize

Images
------

.. autoclass:: ImageData
   :members: getData, getRgbaImageData,
             getOrigin, setOrigin,
             getScale, setScale,
             isDraggable,
             getAlpha, setAlpha,
             getColormap, setColormap,
             getAlternativeImageData

.. autoclass:: ImageRgba
   :members: getData, getRgbaImageData,
             getOrigin, setOrigin,
             getScale, setScale,
             isDraggable,
             getAlpha, setAlpha

Scatter
-------

.. autoclass:: Scatter
   :members: getValueData,
             getData, getXData, getYData, getXErrorData, getYErrorData, setData,
             getSymbol, setSymbol, getSymbolSize, setSymbolSize,
             getAlpha, setAlpha,
             getColormap, setColormap

Histogram
---------

.. autoclass:: Histogram
   :members: getValueData, getBinEdgesData, getData, setData,
             getAlpha, setAlpha,
             getColor, setColor,
             getYAxis, setYAxis,
             isFill, setFill,
             getLineWidth, setLineWidth, getLineStyle, setLineStyle

Markers
-------

.. autoclass:: Marker
   :members: getText, setText, getXPosition, getYPosition, getPosition, setPosition, getConstraint,
             getSymbol, setSymbol, getSymbolSize, setSymbolSize

.. autoclass:: XMarker
   :members: getText, setText, getXPosition, getYPosition, getPosition, setPosition, getConstraint

.. autoclass:: YMarker
   :members: getText, setText, getXPosition, getYPosition, getPosition, setPosition, getConstraint

Shapes
------

.. autoclass:: Shape
   :members: setOverlay,
             getColor, setColor,
             isFill, setFill,
             getType, getPoints, setPoints

Item changed signal
-------------------

Plot items emit a :attr:`Item.sigItemChanged` signal when their values are updated.
This signal provides a flag in the following enumeration describing the modified value:

.. autoclass:: ItemChangedType
   :members:

Axis
----

.. autoclass:: Axis
   :members:


:mod:`~silx.gui.plot.items.roi`: Regions of Interest
----------------------------------------------------

.. automodule:: silx.gui.plot.items.roi
   :members:
