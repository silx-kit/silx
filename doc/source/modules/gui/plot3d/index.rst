
.. currentmodule:: silx.gui

:mod:`plot3d`: 3D Visualisation widgets
=======================================

.. currentmodule:: silx.gui.plot3d

.. automodule:: silx.gui.plot3d

Widgets gallery
---------------

.. |imgPlot3DWidget| image:: img/Plot3DWidget.png
   :height: 150px
   :align: middle

.. |imgPlot3DWindow| image:: img/Plot3DWindow.png
   :height: 150px
   :align: middle

.. |imgScalarFieldView| image:: img/ScalarFieldView.png
   :height: 150px
   :align: middle

.. |imgSFViewParamTree| image:: img/SFViewParamTree.png
   :height: 150px
   :align: middle

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - |imgScalarFieldView|
     - :class:`ScalarFieldView` is a :class:`Plot3DWindow` dedicated to display 3D scalar field.
       It can display iso-surfaces and an interactive cutting plane.
       Sample code: :doc:`viewer3dvolume_example`.
   * - |imgPlot3DWindow|
     - :class:`Plot3DWindow` is a :class:`QMainWindow` with a :class:`Plot3DWidget` as central widget
       and toolbars.
   * - |imgPlot3DWidget|
     - :class:`Plot3DWidget` is the base Qt widget providing an OpenGL 3D scene.
       Other widgets are using this widget as the OpenGL scene canvas.
   * - |imgSFViewParamTree|
     - :class:`SFViewParamTree` is a :class:`QTreeView` widget that can be attached to a :class:`ScalarFieldView`.
       It displays current parameters of the :class:`ScalarFieldView` and allows to modify it.
       Sample code: :doc:`viewer3dvolume_example`.

Public modules
--------------

The following sub-modules are available:

.. toctree::
   :maxdepth: 2

   plot3dwidget.rst
   plot3dwindow.rst
   scalarfieldview.rst
   sfviewparamtree.rst
   toolbars.rst
   actions.rst


Sample code
-----------

- :doc:`viewer3dvolume_example`: Sample code using :class:`ScalarFieldView`

Internals
---------

.. toctree::
   :maxdepth: 2

   dev.rst

.. toctree::
   :hidden:

   viewer3dvolume_example.rst
