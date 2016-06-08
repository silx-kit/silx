Using silx widgets in Qt Designer
=================================

With PyQt_, it is possible to use ``silx.gui`` widgets (and widgets written with PyQt in general) from the `Qt Designer`_.

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
- The ``silx`` Python library and its dependencies.

Usage
-----

The **PYQTDESIGNERPATH** environment variable sets the directories the PyQt designer plugin is searching for plugins enabling PyQt widgets in the Qt Designer.

To start the Qt Designer with ``silx.gui`` widgets available, run the following from the command line::

    PYQTDESIGNERPATH=<silx_designer_plugin_dir_path> designer

See `Using Qt Designer <http://pyqt.sourceforge.net/Docs/PyQt5/designer.html>`_ in PyQt_ documentation.

.. _Qt: http://www.qt.io/
.. _Python: https://www.python.org/
.. _PyQt: https://riverbankcomputing.com/software/pyqt/intro
.. _Qt Designer: http://doc.qt.io/qt-5/qtdesigner-manual.html
