Using silx widgets in Qt Designer
=================================

It is possible to use ``silx.gui`` widgets (and in general widgets based on PyQt) from the `Qt Designer`_.

The following ``silx.gui`` widgets are available in the Qt Designer:

- :class:`silx.gui.plot.Plot1D`
- :class:`silx.gui.plot.Plot2D`
- :class:`silx.gui.plot.PlotWidget`
- :class:`silx.gui.plot.PlotWindow`

Pre-requisite
-------------

The following software must be installed:

- Qt_ with the `Qt Designer`_.
- Python_.
- PyQt_ with the designer plugin.
- The ``silx`` Python package and its dependencies.
  :mod:`silx.gui.plot` widgets require matplotlib_.

Usage
-----

The **PYQTDESIGNERPATH** environment variable defines the search paths for plugins enabling the use of PyQt widgets in the Qt designer.

To access ``silx.gui`` widgets from within Qt designer, under Linux, type from the command line::

    PYQTDESIGNERPATH=<silx_designer_plugin_dir> designer


See `"using Qt Designer" <http://pyqt.sourceforge.net/Docs/PyQt5/designer.html>`_ in the PyQt_ documentation.

.. _Qt: http://www.qt.io/
.. _Python: https://www.python.org/
.. _PyQt: https://riverbankcomputing.com/software/pyqt/intro
.. _Qt Designer: http://doc.qt.io/qt-5/qtdesigner-manual.html
.. _matplotlib: http://matplotlib.org/
