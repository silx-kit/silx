.. currentmodule:: silx.gui.plot3d

:mod:`items`: SceneWidget items
===============================

The following classes are items that describes the content of a :class:`SceneWidget`:

.. currentmodule:: silx.gui.plot3d.items

- :class:`~silx.gui.plot3d.items.image.ImageData`
- :class:`~silx.gui.plot3d.items.image.ImageRgba`
- :class:`~silx.gui.plot3d.items.scatter.Scatter2D`
- :class:`~silx.gui.plot3d.items.scatter.Scatter3D`
- :class:`~silx.gui.plot3d.items.volume.ScalarField3D`
- :class:`~silx.gui.plot3d.items.clipplane.ClipPlane`
- :class:`~silx.gui.plot3d.items.mesh.Mesh`
- :class:`~silx.gui.plot3d.items.core.GroupItem`

2D images
---------

.. currentmodule:: silx.gui.plot3d.core

.. currentmodule:: silx.gui.plot3d.items.image

:class:`ImageData`
++++++++++++++++++

:class:`ImageData` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: ImageData
   :members: getData, setData,
             getColormap, setColormap,
             getInterpolation, setInterpolation

:class:`ImageRgba`
++++++++++++++++++

:class:`ImageRgba` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: ImageRgba
   :members: getData, setData,
             getInterpolation, setInterpolation

2D/3D scatter data
------------------

.. currentmodule:: silx.gui.plot3d.items.scatter

:class:`Scatter2D`
++++++++++++++++++

:class:`Scatter2D` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: Scatter2D
   :members: getData, setData, getXData, getYData, getValueData,
             supportedVisualizations, isPropertyEnabled,
             getVisualization, setVisualization,
             isHeightMap, setHeightMap,
             getLineWidth, setLineWidth,
             getColormap, setColormap,
             getSupportedSymbols, getSymbol, setSymbol

:class:`Scatter3D`
++++++++++++++++++

:class:`Scatter3D` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: Scatter3D
   :members: getData, setData, getXData, getYData, getZData, getValueData,
             getColormap, setColormap,
             getSupportedSymbols, getSymbol, setSymbol

3D volume
---------

.. currentmodule:: silx.gui.plot3d.items.volume

:class:`ScalarField3D`
++++++++++++++++++++++

:class:`ScalarField3D` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: ScalarField3D
   :members: getData, setData,
             getCutPlanes,
             sigIsosurfaceAdded, sigIsosurfaceRemoved,
             addIsosurface, getIsosurfaces, removeIsosurface, clearIsosurfaces

The following classes allows to configure :class:`ScalarField3D` visualization:

:class:`IsoSurface`
+++++++++++++++++++

:class:`IsoSurface` inherits from :class:`.Item3D` and also provides its API.

.. autoclass:: Isosurface
   :show-inheritance:
   :members:

:class:`CutPlane`
+++++++++++++++++

:class:`CutPlane` inherits from :class:`.Item3D` and also provides its API.

.. autoclass:: CutPlane
   :members: getColormap, setColormap,
             getInterpolation, setInterpolation,
             moveToCenter, isValid,
             getNormal, setNormal,
             getPoint, setPoint,
             getParameters, setParameters,
             getDisplayValuesBelowMin, setDisplayValuesBelowMin

Clipping plane
--------------

.. currentmodule:: silx.gui.plot3d.items.clipplane

:class:`ClipPlane`
++++++++++++++++++

:class:`ClipPlane` inherits from :class:`.Item3D` and also provides its API.

.. autoclass:: ClipPlane
   :show-inheritance:
   :members:

3D mesh
-------

.. currentmodule:: silx.gui.plot3d.items.mesh

:class:`Mesh`
+++++++++++++

:class:`Mesh` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: Mesh
   :show-inheritance:
   :members:

Item base classes
-----------------

The following classes provides the base classes for other items.

.. currentmodule:: silx.gui.plot3d.items.core

:class:`Item3D`
+++++++++++++++

.. autoclass:: Item3D
   :show-inheritance:
   :members:

:class:`DataItem3D`
+++++++++++++++++++

:class:`DataItem3D` inherits from :class:`.Item3D` and also provides its API.

.. autoclass:: DataItem3D
   :show-inheritance:
   :members:

:class:`GroupItem`
++++++++++++++++++

:class:`GroupItem` inherits from :class:`.DataItem3D` and also provides its API.

.. autoclass:: GroupItem
   :show-inheritance:
   :members:
