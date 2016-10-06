

Fit tools
---------

.. _leastsq-tutorial:

Using :func:`leastsq`
+++++++++++++++++++++

Running an iterative fit with :func:`leastsq` involves the following steps:

    - designing a fit model function that has the signature ``f(x, ...)``,
      where ``x`` is an array of values of the independant variable and all
      remaining parameters are the parameters to be fitted
    - defining the sequence of initial values for all parameters to be fitted.
      You can usually start with ``[1., 1., ...]`` if you don't know a better
      estimate. The algorithm is robust enough to converge to a solution most
      of the time.
    - setting constraints (optional)

Data required to perform a fit is:

    - an array of ``x`` values (abscissa, independant variable)
    - an array of ``y`` data points
    - the ``sigma`` array of uncertainties associated to each data point.
      This is optional, by default each data point gets assigned a weight of 1.

Standard fit
************

Let's demonstrate this process in a short example, using synthetic data.
We generate an array of synthetic data using a polynomial function of degree 4,
and try to use :func:`leastsq` to find back the functions parameters.

.. code-block:: python

   import numpy
   from silx.math.fit import leastsq

   # create some synthetic polynomial data
   x = numpy.arange(1000)
   y = 2.4 * x**4 - 10. * x**3 + 15.2 * x**2 - 24.6 * x + 150.

   # define our fit function: a generic polynomial of degree 4
   def poly4(x, a, b, c, d, e):
       return a * x**4 + b * x**3 + c * x**2 + d * x + e

   # The fit is an iterative process that requires an initial
   # estimation of the parameters. Let's just use 1s.
   initial_parameters = numpy.array([1., 1., 1., 1., 1.])

   # Run fit
   fitresult = leastsq(model=poly4,
                       xdata=x,
                       ydata=y,
                       p0=initial_parameters,
                       full_output=True)

   # leastsq with full_output=True returns 3 objets
   optimal_parameters, covariance, infodict = fitresult
   # the first object is an array with the fitted parameters
   a, b, c, d, e = optimal_parameters

   print("Fit took %d iterations" % infodict["niter"])
   print("Reduced chi-square: %f" % infodict["reduced_chisq"])
   print("Theoretical parameters:\n\t" +
         "a=2.4, b=-10, c=15.2, d=-24.6, e=150")
   print("Optimal parameters for y2 fitting:\n\t" +
         "a=%f, b=%f, c=%f, d=%f, e=%f" % (a, b, c, d, e))

The output of this program is::

   Fit took 35 iterations
   Reduced chi-square: 682592.670690
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters for y fitting:
       a=2.400000, b=-9.999665, c=14.970422, d=31.683448, e=-3216.131136

We can see that this fit result is poor. In particular, parameters ``d`` and ``e``
are very poorly fitted.
This is most likely due to numerical rounding errors. As we are dealing with
very large values in our ``y`` array, we are affected by the limits of how
floating point numbers are represented by computers. The larger a value, the
larger its rounding error.

If you limit the ``x`` range to deal with
smaller ``y`` values, the fit result becomes perfect. In our example, replacing ``x``
with::

    x = numpy.arange(100)

produces the following result::

   Fit took 9 iterations
   Reduced chi-square: 0.000000
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters for y fitting:
       a=2.400000, b=-10.000000, c=15.200000, d=-24.600000, e=150.000000



Constrained fit
***************

But let's revert back to our initial ``x = numpy.arange(1000)``, to experiment
with different approaches to improving the fit.

The :func:`leastsq` functions provides
a way to set constraints on parameters. You can for instance assert that a given
parameter must remain equal to it's initial value, or define an acceptable range
for it to vary, or decide that a parameter must be equal to another parameter
multiplied by a certain factor. This is very useful in cases in which you have
enough knowledge to make reasonable assumptions on some parameters.

In our case, we will set constraints on ``d`` and ``e``. We will quote ``d`` to
stay in the range between -25 and -24, and fix ``e`` to 150.

Replace the call to :func:`leastsq` by following lines:

.. code-block:: python

   # Define constraints
   cons = [[0, 0, 0],          # a: no constraint
           [0, 0, 0],          # b: no constraint
           [0, 0, 0],          # c: no constraint
           [2, -25., -23.],    # -25 < d < -24
           [3, 0, 0]]          # e is fixed to initial value
   fitresult = leastsq(poly4, x, y,
                       # initial values must be consistent with constraints
                       p0=[1., 1., 1., -24., 150.],
                       constraints=cons,
                       full_output=True)

The output of this program is::

   Constrained fit took 100 iterations
   Reduced chi-square: 3.749280
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters for y fitting:
       a=2.400000, b=-9.999999, c=15.199648, d=-24.533014, e=150.000000

The chi-square value is much improved and the results are much better, at the
cost of more iterations.

Weighted fit
************
A third approach to improve our fit is to define uncertainties for the data.
The larger the uncertainty on a data sample, the smaller its weight will be
in the least-square problem.

In our case, we don't have obvious uncertainties associated to our data, altough we could
try to figure out the uncertainties due to numerical rounding errors by closely
looking at how floating point values are stored.

A common approach that requires less work is to use the square-root of the data values
as their uncertainty value. Let's try it:

.. code-block:: python

   sigma = numpy.sqrt(y)

   # Fit y
   fitresult = leastsq(model=poly4,
                       xdata=x,
                       ydata=y,
                       sigma=sigma,
                       p0=initial_parameters,
                       full_output=True)

This results in a great improvement::

   Weighted fit took 6 iterations
   Reduced chi-square: 0.000000
   Theoretical parameters:
       a=2.4, b=-10, c=15.2, d=-24.6, e=150
   Optimal parameters for y fitting:
       a=2.400000, b=-10.000000, c=15.200000, d=-24.600000, e=150.000000

The resulting fit is perfect. The very large ``y`` values with their very large
associated uncertainties have been ignored, for all practical purposes. The fit
converged even faster than with the solution of limiting the ``x`` range to
0 -- 100.

.. _fitmanager-tutorial:

Using :class:`FitManager`
+++++++++++++++++++++++++

A :class:`FitManager` is a tool that provides a way of handling fit functions,
associating estimation functions to estimate the initial parameters, modify
the configuration parameters for the fit (enabling or disabling weights...) or
for the estimation function, and choosing a background model.

Weighted polynomial fit
***********************

The following program accomplishes the same weighted fit of a polynomial as in
the previous tutorial (`Weighted fit`_)

.. code-block:: python

    import numpy
    from silx.math.fit.fitmanager import FitManager

    # Create synthetic data with a sum of gaussian functions
    x = numpy.arange(1000).astype(numpy.float)
    y = 2.4 * x**4 - 10. * x**3 + 15.2 * x**2 - 24.6 * x + 150.

    # define our fit function: a generic polynomial of degree 4
    def poly4(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    # define an estimation function to that returns initial parameters
    # and constraints
    def esti(x, y):
        p0 = numpy.array([1., 1., 1., 1., 1.])
        cons = numpy.zeros(shape=(5, 3))
        return p0, cons

    # Fitting
    fit = FitManager()
    fit.setdata(x=x, y=y)

    fit.addtheory("polynomial",
                  function=poly4,
                  # any list of 5 parameter names would be OK
                  parameters=["A", "B", "C", "D", "E"],
                  estimate=esti)
    fit.settheory('polynomial')
    fit.configure(WeightFlag=True)
    fit.estimate()
    fit.runfit()

    print("\n\nFit took %d iterations" % fit.niter)
    print("Reduced chi-square: %f" % fit.chisq)
    print("Theoretical parameters:\n\t" +
          "a=2.4, b=-10, c=15.2, d=-24.6, e=150")
    a, b, c, d, e = (param['fitresult'] for param in fit.fit_results)
    print("Optimal parameters for y2 fitting:\n\t" +
          "a=%f, b=%f, c=%f, d=%f, e=%f" % (a, b, c, d, e))


The result is the same as in our weighted :func:`leastsq` example,
as expected::

    Fit took 6 iterations
    Reduced chi-square: 0.000000
    Theoretical parameters:
        a=2.4, b=-10, c=15.2, d=-24.6, e=150
    Optimal parameters for y2 fitting:
        a=2.400000, b=-10.000000, c=15.200000, d=-24.600000, e=150.000000

Fitting gaussians
*****************

The :class:`FitManager` object is especially useful for fitting multi-peak
gaussian-shaped spectra. The *silx* module :mod:`silx.math.fit.fittheories`
provides fit functions and their associated estimation functions that are
specifically designed for this purpose.

These fit functions can handle variable number of parameters defining a
variable number of peaks, and the estimation functions use a peak detection
algorithm to determine how many initial parameters must be returned.

For the sake of the example, let's test the multi-peak fitting on synthetic
data, generated using another *silx* module: :mod:`silx.math.fit.functions`.

.. code-block:: python

    import numpy
    from silx.math.fit.functions import sum_gauss
    from silx.math.fit import fittheories
    from silx.math.fit.fitmanager import FitManager

    # Create synthetic data with a sum of gaussian functions
    x = numpy.arange(1000).astype(numpy.float)

    # height, center x, fwhm
    p = [1000, 100., 250,     # 1st peak
         255, 690., 45,       # 2nd peak
         1500, 800.5, 95]     # 3rd peak

    y = sum_gauss(x, *p)

    # Fitting
    fit = FitManager()
    fit.setdata(x=x, y=y)
    fit.loadtheories(fittheories)
    fit.settheory('Gaussians')
    fit.estimate()
    fit.runfit()

    print("Searched parameters = %s" % p)
    print("Obtained parameters : ")
    dummy_list = []
    for param in fit.fit_results:
        print(param['name'], ' = ', param['fitresult'])
        dummy_list.append(param['fitresult'])
    print("chisq = ", fit.chisq)

And the result of this program is::

    Searched parameters = [1000, 100.0, 250, 255, 690.0, 45, 1500, 800.5, 95]
    Obtained parameters :
    ('Height1', ' = ', 1000.0)
    ('Position1', ' = ', 100.0)
    ('FWHM1', ' = ', 250.0)
    ('Height2', ' = ', 255.0)
    ('Position2', ' = ', 690.0)
    ('FWHM2', ' = ', 44.999999999999993)
    ('Height3', ' = ', 1500.0)
    ('Position3', ' = ', 800.5)
    ('FWHM3', ' = ', 95.000000000000014)
    ('chisq = ', 0.0)

In addition to gaussians, we could have fitted several other similar type of
functions: asymetric gaussian functions, lorentzian functions,
Pseudo-Voigt functions or hypermet tailing functions.

The :meth:`loadtheories` method can also be used to load user defined
functions. Instead of a module, a path to a Python source file can be given
as a parameter. This source file must adhere to certain conventions, explained
in the documentation of :mod:`silx.math.fit.fittheories`.

Subtracting a background
************************

:class:`FitManager` provides a few standard background theories, for cases when
a background signal is superimposed on the multi-peak spectrum.

For example, let's add a linear background to our synthetic data, and see how
:class:`FitManager` handles the fitting.

In our previous example, redefine ``y`` as follows:

.. code-block:: python

    p = [1000, 100., 250,
         255, 690., 45,
         1500, 800.5, 95]
    y = sum_gauss(x, *p)
    # add a synthetic linear background
    y += 0.13 * x + 100.

Before the line ``fit.estimate()``, add the following line:

.. code-block:: python

    fit.setbackground('Linear')

The result becomes::

    Searched parameters = [1000, 100.0, 250, 255, 690.0, 45, 1500, 800.5, 95]
    Obtained parameters :
    ('Constant', ' = ', 100.00000000000001)
    ('Slope', ' = ', 0.12999999999999998)
    ('Height1', ' = ', 1000.0)
    ('Position1', ' = ', 100.0)
    ('FWHM1', ' = ', 249.99999999999997)
    ('Height2', ' = ', 255.00000000000003)
    ('Position2', ' = ', 690.0)
    ('FWHM2', ' = ', 44.999999999999993)
    ('Height3', ' = ', 1500.0)
    ('Position3', ' = ', 800.5)
    ('FWHM3', ' = ', 95.0)
    ('chisq = ', 3.1789004676997597e-27)

The available background theories are: *Linear*, *Constant* and *Strip*.

The strip background is a popular background model that can compute and
subtract any background shape as long as its curvature is significantly
lower than the peaks' curvature. In other words, as long as the background
signal is significantly smoother than the actual signal, it can be easily
computed.

The main parameters required by the strip function are the strip width *w*
and the number of iterations. At each iteration, if the contents of channel *i*,
``y(i)``, is above the average of the contents of the channels at *w* channels of
distance, ``y(i-w)`` and ``y(i+w)``,  ``y(i)`` is replaced by the average.
At the end of the process we are left with something that resembles a spectrum
in which the peaks have been "stripped".

The following example illustrates the effect of strip background removal:

.. code-block:: python

    from silx.gui.plot import plot1D, plot2D
    from silx.gui import qt
    import numpy
    from silx.math.fit.filters import strip
    from silx.math.fit.functions import sum_gauss

    x = numpy.arange(5000)
    # (height1, center1, fwhm1, ...) 5 peaks
    params1 = (50, 500, 100,
               20, 2000, 200,
               50, 2250, 100,
               40, 3000, 75,
               23, 4000, 150)
    y0 = sum_gauss(x, *params1)

    # random values between [-1;1]
    noise = 2 * numpy.random.random(5000) - 1
    # make it +- 5%
    noise *= 0.05

    # 2 gaussians with very large fwhm, as background signal
    actual_bg = sum_gauss(x, 15, 3500, 3000, 5, 1000, 1500)

    # Add 5% random noise to gaussians and add background
    y = y0 * (1 + noise) + actual_bg

    # compute strip background model
    strip_bg = strip(y, w=5, niterations=5000)

    # plot results
    app = qt.QApplication([])
    plot1D(x, (y, actual_bg, strip_bg))
    plot1D(x, (y, y - strip_bg))
    app.exec_()

.. |imgStrip1| image:: img/stripbg_plot1.png
   :height: 300px
   :align: middle

.. |imgStrip2| image:: img/stripbg_plot2.png
   :height: 300px
   :align: middle

.. list-table::
   :widths: 1 2

   * - |imgStrip1|
     - Data with background in black (``y``), actual background in red, computed strip
       background in green
   * - |imgStrip2|
     - Data with background in blue, data after subtracting strip background in black


The strip also removes the statistical noise, so the computed strip background
will be slightly lower than the actual background. This can be solved by
performing a smoothing prior to the strip computation.

.. _fitwidget-tutorial:

Using :class:`FitWidget`
++++++++++++++++++++++++

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
    w.setData(x=x, y=y)
    w.show()
    a.exec_()

.. |imgFitWidget3| image:: img/fitwidget3.png
   :width: 400px
   :align: middle

Executing this code, then selecting a constant background, clicking
the estimate button, then the fit button, shows the following result:

    |imgFitWidget3|


The following example shows how to define a custom fit function.

.. code-block:: python

    from silx.math.fit import FitManager
    from silx.gui import qt
    from silx.gui.fit import FitWidget

    def linearfun(x, a, b):
        return a * x + b

    # create synthetic data for the example
    x = list(range(0, 100))
    y = [linearfun(x_, 2.0, 3.0) for x_ in x]

    # we need to create a custom fit manager and add our theory
    myfitmngr = FitManager()
    myfitmngr.setData(x, y)
    myfitmngr.addtheory("my linear function",
                        function=linearfun,
                        parameters=["a", "b"])

    a = qt.QApplication([])

    # our fit widget can now use our custom fit manager
    fw = FitWidget(fitmngr=myfitmngr)
    fw.show()

    a.exec_()
