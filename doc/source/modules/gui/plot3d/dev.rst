Package structure
=================

The :mod:`silx.gui.plot3d` package provides 3D visualisation widgets.
This package is structured as follows.

.. currentmodule:: silx.gui.plot3d

Widget-level API
----------------

Widgets are available as modules of the :mod:`silx.gui.plot3d` packages.

The :mod:`.Plot3DWidget` module provides the OpenGL canvas where the scene is rendered.
The :mod:`.Plot3DWindow` module provides a :class:`QMainWindow` with a :class:`Plot3DWindow` as its central widget
and toolbars: :class:`ViewpointToolBar` and :class:`Plot3DToolBar`.
:class:`QAction` that can be associated with a :class:`Plot3DWidget` are defined in the :mod:`.Plot3DActions` module.
Those actions are used by the :class:`Plot3DToolBar` toolbar.

The :mod:`.ScalarFieldView` module defines the :class:`ScalarFieldView` widget that displays iso-surfaces of a 3D scalar data set and the associated classes.
The :mod:`.SFViewParamTree` module defines a :class:`SFViewParamTree.TreeView` widget that can be attached to a :class:`ScalarFieldView` to control the display.

OpenGL scene API
----------------

This API is NOT stable.
Widgets of :mod:`silx.gui.plot3d` are based on the following sub-packages:

- :mod:`.glutils`: Loads PyOpenGL and provides classes to handle OpenGL resources.
- :mod:`.scene`: Provides a hierarchical scene structure handling rendering and interaction.
- :mod:`.utils`: Miscellaneous supporting modules.
- :mod:`._font`: Provides an API to use Qt Font system from the :mod:`.scene` package.

.. toctree::
   :maxdepth: 2

   glutils.rst
   scene.rst
   utils.rst
   _font.rst
