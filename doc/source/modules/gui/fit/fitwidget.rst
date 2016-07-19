
.. currentmodule:: silx.gui.fit

:mod:`FitWidget`
================

.. automodule:: silx.gui.fit.FitWidget

Usage example
-------------

.. code-block:: python

    import numpy
    from silx.gui.fit import FitWidget
    from silx.math.fit.functions import sum_gauss
    from silx.gui import qt

    x = numpy.arange(2000).astype(numpy.float)
    constant_bg = 3.14

    # gaussian parameters: height, position, fwhm
    p = numpy.array([1000, 100., 30.0,
                     500, 300., 25.,
                     1700, 500., 35.,
                     750, 700., 30.0,
                     1234, 900., 29.5,
                     302, 1100., 30.5,
                     75, 1300., 210.])
    y = sum_gauss(x, *p) + constant_bg

    a = qt.QApplication([])
    a.lastWindowClosed.connect(a.quit)
    w = FitWidget(enableconfig=1, enablestatus=1, enablebuttons=1)
    w.setdata(x=x, y=y)
    w.show()
    a.exec_()

.. |imgFitWidget3| image:: img/fitwidget3.png
   :width: 400px
   :align: middle

Executing this code, then selecting a constant background, clicking
the estimate button, then the fit button, shows the following result:

    |imgFitWidget3|

API
---

.. currentmodule:: silx.gui.fit.FitWidget

.. autoclass:: FitWidget
   :members: __init__, setdata
