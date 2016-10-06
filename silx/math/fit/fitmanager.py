# coding: utf-8
# /*#########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
# ##########################################################################*/
"""
This module provides a tool to perform advanced fitting. The actual fit relies
on :func:`silx.math.fit.leastsq`.

This module deals with:

    - handling of the model functions (using a set of default functions or
      loading custom user functions)
    - handling of estimation function, that are used to determine the number
      of parameters to be fitted for functions with unknown number of
      parameters (such as the sum of a variable number of gaussian curves),
      and find reasonable initial parameters for input to the iterative
      fitting algorithm
    - handling of custom  derivative functions that can be passed as a
      parameter to  :func:`silx.math.fit.leastsq`
    - removal of constant and linear background signal prior to performing the
      actual fit

"""
from collections import OrderedDict
import logging
import numpy
from numpy.linalg.linalg import LinAlgError
import os
import sys

from .filters import strip, smooth1d
from .leastsq import leastsq
from .fittheory import FitTheory


__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "29/08/2016"

_logger = logging.getLogger(__name__)


class FitManager(object):
    """
    Fit functions manager

    :param x: Abscissa data. If ``None``, :attr:`xdata` is set to
        ``numpy.array([0.0, 1.0, 2.0, ..., len(y)-1])``
    :type x: Sequence or numpy array or None
    :param y: The dependant data ``y = f(x)``. ``y`` must have the same
        shape as ``x`` if ``x`` is not ``None``.
    :type y: Sequence or numpy array or None
    :param sigmay: The uncertainties in the ``ydata`` array. These can be
        used as weights in the least-squares problem, if ``weight_flag``
        is ``True``.
        If ``None``, the uncertainties are assumed to be 1, unless
        ``weight_flag`` is ``True``, in which case the square-root
        of ``y`` is used.
    :type sigmay: Sequence or numpy array or None
    :param weight_flag: If this parameter is ``True`` and ``sigmay``
        uncertainties are not specified, the square root of ``y`` is used
        as weights in the least-squares problem. If ``False``, the
        uncertainties are set to 1.
    :type weight_flag: boolean
    """
    def __init__(self, x=None, y=None, sigmay=None, weight_flag=False):
        """
        """
        self.fitconfig = {
            'FwhmPoints': 8,   # Fixme: if we decide to drop square filter BG,
                               # we can get rid of this param (will be defined in fittheories for peak detection)
            'WeightFlag': weight_flag,
            'fitbkg': 'No Background',
            'fittheory': None,
            'StripWidth': 2,
            'StripNIterations': 5000,
            'StripThresholdFactor': 1.0,
            'SmoothStrip': False
        }
        """Dictionary of fit configuration parameters.
        These parameters can be modified using the :meth:`configure` method.

        Keys are:

            - 'fitbkg': name of the function used for fitting a low frequency
              background signal
            - 'FwhmPoints': default full width at half maximum value for the
              peaks'.
            - 'Sensitivity': Sensitivity parameter for the peak detection
              algorithm (:func:`silx.math.fit.peak_search`)
        """

        self.theories = OrderedDict()
        """Dictionary of fit theories, defining functions to be fitted
        to individual peaks.

        Keys are descriptive theory names (e.g "Gaussians" or "Step up").
        Values are :class:`silx.math.fit.fittheory.FitTheory` objects with
        the following attributes:

            - *"function"* is the fit function for an individual peak
            - *"parameters"* is a sequence of parameter names
            - *"estimate"* is the parameter estimation function
            - *"configure"* is the function returning the configuration dict
              for the theory in the format described in the :attr:` fitconfig`
              documentation
            - *"derivative"* (optional) is a custom derivative function, whose
              signature is described in the documentation of
              :func:`silx.math.fit.leastsq.leastsq`
              (``model_deriv(xdata, parameters, index)``).
            - *"description"* is a description string
        """

        self.selectedtheory = None
        """Name of currently selected theory. This name matches a key in
        :attr:`theories`."""

        self.bgtheories = OrderedDict((
             ('No Background', FitTheory(
                                 description="No background function",
                                 function=self.bkg_none,
                                 parameters=[],
                                 estimate=None)),
             ('Constant', FitTheory(
                                 description="Constant background",
                                 function=self.bkg_constant,
                                 parameters=['Constant'],
                                 estimate=self.estimate_builtin_bkg)),
             ('Linear', FitTheory(
                                 description="Linear background, parameters 'Constant' and 'Slope'",
                                 function=self.bkg_linear,
                                 parameters=['Constant', 'Slope'],
                                 estimate=self.estimate_builtin_bkg)),
             ('Strip', FitTheory(
                                 description="Background based on strip filter\n" +
                                             "Parameters 'StripWidth', 'StripIterations'",
                                 function=self.bkg_strip,
                                 parameters=['StripWidth', 'StripIterations'],
                                 estimate=self.estimate_builtin_bkg))))
        """Dictionary of background theories.

        Keys are descriptive theory names (e.g "Constant" or "Linear").
        Values are :class:`silx.math.fit.fittheory.FitTheory` objects.

          - *description* is an optional description string, which can be used
            for instance as a tooltip message in a GUI.

          - *function* is a callable function with the signature ``function(x, params) -> y``
            where params is a sequence of parameters.

          - *parameters* is a sequence of parameter names (e.g. could be
            for a linear function ``["constant", "slope"]``).

          - *estimate* is a function to compute initial values for parameters.
            It should have the following signature:
            ``f(x, y) -> (estimated_param, constraints)``

                Parameters:

                - ``x`` is the independant variable, i.e. all the points where
                  the function is calculated
                - ``y`` is the data from which we want to extract the bg

                Return values:

                - ``estimated_param`` is a list of estimated values for each
                  background parameter.
                - ``constraints`` is a 2D sequence of dimension ``(n_parameters, 3)``

                  See explanation about 'constraints' in :attr:`fit_results`
                  documentation.
        """

        self.selectedbg = 'No Background'
        """Name of currently selected background theory. This name matches a
        key in :attr:``."""

        self.fit_results = []
        """This list stores detailed information about all fit parameters.
        It is initialized in :meth:`estimate` and completed with final fit
        values in :meth:`runfit`.

        Each fit parameter is stored as a dictionary with following fields:

            - 'name': Parameter name.
            - 'estimation': Estimated value.
            - 'group': Group number. Group 0 corresponds to the background
              function parameters. Group ``n`` (for ``n>0``) corresponds to
              the fit function parameters for the n-th peak.
            - 'code': Constraint code

                - 0 - FREE
                - 1 - POSITIVE
                - 2 - QUOTED
                - 3 - FIXED
                - 4 - FACTOR
                - 5 - DELTA
                - 6 - SUM

            - 'cons1':

                - Ignored if 'code' is FREE, POSITIVE or FIXED.
                - Min value of the parameter if code is QUOTED
                - Index of fitted parameter to which 'cons2' is related
                  if code is FACTOR, DELTA or SUM.

            - 'cons2':

                - Ignored if 'code' is FREE, POSITIVE or FIXED.
                - Max value of the parameter if QUOTED
                - Factor to apply to related parameter with index 'cons1' if
                  'code' is FACTOR
                - Difference with parameter with index 'cons1' if
                  'code' is DELTA
                - Sum obtained when adding parameter with index 'cons1' if
                  'code' is SUM

            - 'fitresult': Fitted value.
            - 'sigma': Standard deviation for the parameter estimate
            - 'xmin': Lower limit of the ``x`` data range on which the fit
              was performed
            - 'xmax': Upeer limit of the ``x`` data range on which the fit
              was performed
        """

        self.parameter_names = []
        """This list stores all fit parameter names: background function
        parameters and fit function parameters for every peak. It is filled
        in :meth:`estimate`.

        It is the responsibility of the estimate function defined in
        :attr:`theories` to determine how many parameters are needed,
        based on how many peaks are detected and how many parameters are needed
        to fit an individual peak.
        """

        self.setdata(x, y, sigmay)

        # Attributes used to store internal background parameters and data,
        # to avoid costly computations when parameters stay the same
        self._bkg_strip_oldx = numpy.array([])
        self._bkg_strip_oldy = numpy.array([])
        self._bkg_strip_oldpars = [0, 0]
        self._bkg_strip_oldbkg = numpy.array([])

    ##################
    # Public methods #
    ##################
    def addbackground(self, bgname, bgtheory):
        """Add a new background theory to dictionary :attr:`bgtheories`.

        :param bgname: String with the name describing the function
        :param bgtheory:  :class:`FitTheory` object
        :type bgtheory: :class:`silx.math.fit.fittheory.FitTheory`
        """
        self.bgtheories[bgname] = bgtheory

    def addtheory(self, name, theory=None,
                  function=None, parameters=None,
                  estimate=None, configure=None, derivative=None,
                  description=None, config_widget=None,
                  pymca_legacy=False):
        """Add a new theory to dictionary :attr:`theories`.

        You can pass a name and a :class:`FitTheory` object as arguments, or
        alternatively provide all arguments necessary to instantiate a new
        :class:`FitTheory` object.

        See :meth:`loadtheories` for more information on estimation functions,
        configuration functions and custom derivative functions.

        :param name: String with the name describing the function
        :param theory: :class:`FitTheory` object, defining a fit function and
            associated information (estimation function, description…).
            If this parameter is provided, all other parameters, except for
            ``name``, are ignored.
        :type theory: :class:`silx.math.fit.fittheory.FitTheory`
        :param function function: Mandatory argument if ``theory`` is not provided.
            See documentation for :attr:`silx.math.fit.fittheory.FitTheory.function`.
        :param list[str] parameters: Mandatory argument if ``theory`` is not provided.
            See documentation for :attr:`silx.math.fit.fittheory.FitTheory.parameters`.
        :param function estimate: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.estimate`
        :param function configure: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.configure`
        :param function derivative: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.derivative`
        :param str description: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.description`
        :param config_widget: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.config_widget`
        :param bool pymca_legacy: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.pymca_legacy`
        """
        if theory is not None:
            self.theories[name] = theory

        elif function is not None and parameters is not None:
            self.theories[name] = FitTheory(
                description=description,
                function=function,
                parameters=parameters,
                estimate=estimate,
                configure=configure,
                derivative=derivative,
                config_widget=config_widget,
                pymca_legacy=pymca_legacy
            )

        else:
            raise TypeError("You must supply a FitTheory object or define " +
                            "a fit function and its parameters.")

    def configure(self, **kw):
        """Configure the current theory by filling or updating the
        :attr:`fitconfig` dictionary.
        Call the custom configuration function, if any. This allows the user
        to modify the behavior of the custom fit function or the custom
        estimate function.

        This methods accepts only named parameters. All ``**kw`` parameters
        are expected to be fields of :attr:`fitconfig` to be updated, unless
        they have a special meaning for the custom configuration function
        of the currently selected theory..

        This method returns the modified config dictionary returned by the
        custom configuration function.
        """
        # inspect **kw to find known keys, update them in self.fitconfig
        for key in self.fitconfig:
            if key in kw:
                self.fitconfig[key] = kw[key]

        # initialize dict with existing config dict
        result = {}
        result.update(self.fitconfig)

        if "WeightFlag" in kw:
            if kw["WeightFlag"]:
                self.enableweight()
            else:
                self.disableweight()

        if self.selectedtheory is None:
            return result

        # Apply custom configuration function
        custom_config_fun = self.theories[self.selectedtheory].configure
        if custom_config_fun is not None:
            result.update(custom_config_fun(**kw))

            # Update self.fitconfig with custom config
            for key in self.fitconfig:
                if key in result:
                    self.fitconfig[key] = result[key]

        result.update(self.fitconfig)
        return result

    def dataupdate(self):
        """This method can be updated with a user defined function to
        update data (for instance modify range fo :attr:`xdata`,
        :attr:`ydata` and :attr:`sigmay` when user zooms in or out in a GUI
        plot).

        It is called at the beginning of :meth:`estimate` and
        :meth:`runfit`.

        By default, it does nothing.
        """
        pass

    def estimate(self, callback=None):
        """
        Fill :attr:`fit_results` with an estimation of the fit parameters.

        At first, the background parameters are estimated, if a background
        model has been specified.
        Then, a custom estimation function related to the model function is
        called.

        This process determines the number of needed fit parameters and
        provides an initial estimation for them, to serve as an input for the
        actual iterative fitting performed in :meth:`runfit`.

        :param callback: Optional callback function, conforming to the
            signature ``callback(data)`` with ``data`` being a dictionary.
            This callback function is called before and after the estimation
            process, and is given a dictionary containing the values of
            :attr:`state` (``'Estimate in progress'`` or ``'Ready to Fit'``)
            and :attr:`chisq`.
            This is used for instance in :mod:`silx.gui.fit.FitWidget` to
            update a widget displaying a status message.
        :return: Estimated parameters
        """
        self.state = 'Estimate in progress'
        self.chisq = None

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        CONS = {0: 'FREE',
                1: 'POSITIVE',
                2: 'QUOTED',
                3: 'FIXED',
                4: 'FACTOR',
                5: 'DELTA',
                6: 'SUM',
                7: 'IGNORE'}

        # Update data using user defined method
        self.dataupdate()

        xwork = self.xdata
        ywork = self.ydata

        # estimate the background
        bg_params, bg_constraints = self.estimate_bkg(xwork, ywork)

        # estimate the function
        try:
            fun_params, fun_constraints = self.estimate_fun(xwork, ywork)
        except LinAlgError:
            self.state = 'Estimate failed'
            if callback is not None:
                callback(data={'status': self.state})
            raise

        # build the names
        self.parameter_names = []

        for bg_param_name in self.bgtheories[self.selectedbg].parameters:
            self.parameter_names.append(bg_param_name)

        fun_param_names = self.theories[self.selectedtheory].parameters
        param_index, peak_index = 0, 0
        while param_index < len(fun_params):
            peak_index += 1
            for fun_param_name in fun_param_names:
                self.parameter_names.append(fun_param_name + "%d" % peak_index)
                param_index += 1

        self.fit_results = []
        nb_fun_params_per_group = len(fun_param_names)
        group_number = 0
        xmin = min(xwork)
        xmax = max(xwork)
        nb_bg_params = len(bg_params)
        for (pindex, pname) in enumerate(self.parameter_names):
            # First come background parameters
            if pindex < nb_bg_params:
                estimation_value = bg_params[pindex]
                constraint_code = CONS[int(bg_constraints[pindex][0])]
                cons1 = bg_constraints[pindex][1]
                cons2 = bg_constraints[pindex][2]
            # then come peak function parameters
            else:
                fun_param_index = pindex - nb_bg_params

                # increment group_number for each new fitted peak
                if (fun_param_index % nb_fun_params_per_group) == 0:
                    group_number += 1

                estimation_value = fun_params[fun_param_index]
                constraint_code = CONS[int(fun_constraints[fun_param_index][0])]
                # cons1 is the index of another fit parameter. In the global
                # fit_results, we must adjust the index to account for the bg
                # params added to the start of the list.
                cons1 = fun_constraints[fun_param_index][1]
                if constraint_code in ["FACTOR", "DELTA", "SUM"]:
                    cons1 += nb_bg_params
                cons2 = fun_constraints[fun_param_index][2]

            self.fit_results.append({'name': pname,
                                     'estimation': estimation_value,
                                     'group': group_number,
                                     'code': constraint_code,
                                     'cons1': cons1,
                                     'cons2': cons2,
                                     'fitresult': 0.0,
                                     'sigma': 0.0,
                                     'xmin': xmin,
                                     'xmax': xmax})

        self.state = 'Ready to Fit'
        self.chisq = None
        self.niter = 0

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})
        return numpy.append(bg_params, fun_params)

    def fit(self):
        """Convenience method to call :meth:`estimate` followed by :meth:`runfit`.

        :return: Output of :meth:`runfit`"""
        self.estimate()
        return self.runfit()

    def gendata(self, x=None, paramlist=None):
        """Return a data array using the currently selected fit function
        and the fitted parameters.

        :param x: Independent variable where the function is calculated.
            If ``None``, use :attr:`xdata`.
        :param paramlist: List of dictionaries, each dictionary item being a
            fit parameter. The dictionary's format is documented in
            :attr:`fit_results`.
            If ``None`` (default), use parameters from :attr:`fit_results`.
        :return: :meth:`fitfunction` calculated for parameters whose code is
            not set to ``"IGNORE"``.

        This calculates :meth:`fitfunction` on `x` data using fit parameters
        from a list of parameter dictionaries, if field ``code`` is not set
        to ``"IGNORE"``.
        """
        if x is None:
            x = self.xdata
        if paramlist is None:
            paramlist = self.fit_results
        active_params = []
        for param in paramlist:
            if param['code'] not in ['IGNORE', 7]:
                active_params.append(param['fitresult'])

        newdata = self.fitfunction(numpy.array(x), *active_params)
        return newdata

    def get_estimation(self):
        """Return the list of fit parameter names."""
        if self.state not in ["Ready to fit", "Fit in progress", "Ready"]:
            _logger.warning("get_estimation() called before estimate() completed")
        return [param["estimation"] for param in self.fit_results]

    def get_names(self):
        """Return the list of fit parameter estimations."""
        if self.state not in ["Ready to fit", "Fit in progress", "Ready"]:
            msg = "get_names() called before estimate() completed, "
            msg += "names are not populated at this stage"
            _logger.warning(msg)
        return [param["name"] for param in self.fit_results]

    def get_fitted_parameters(self):
        """Return the list of fitted parameters."""
        if self.state not in ["Ready"]:
            msg = "get_fitted_parameters() called before runfit() completed, "
            msg += "results are not available a this stage"
            _logger.warning(msg)
        return [param["fitresult"] for param in self.fit_results]

    def loadtheories(self, theories):
        """Import user defined fit functions defined in an external Python
        source file, and save them in :attr:`theories`.

        An example of such a file can be found in the sources of
        :mod:`silx.math.fit.fittheories`. It must contain a
        dictionary named ``THEORY`` with the following structure::

            THEORY = {
                'theory_name_1':
                    FitTheory(description='Description of theory 1',
                              function=fitfunction1,
                              parameters=('param name 1', 'param name 2', …),
                              estimate=estimation_function1,
                              configure=configuration_function1,
                              derivative=derivative_function1),
                'theory_name_2':
                    FitTheory(…),
            }

        See documentation of :mod:`silx.math.fit.fittheories` and
        :mod:`silx.math.fit.fittheory` for more
        information on designing your fit functions file.

        This method can also load user defined functions in the legacy
        format used in *PyMca*.

        :param theories: Name of python source file, or module containing the
            definition of fit functions.
        :raise: ImportError if theories cannot be imported
        """
        from types import ModuleType
        if isinstance(theories, ModuleType):
            theories_module = theories
        else:
            # if theories is not a module, it must be a string
            string_types = (basestring,) if sys.version_info[0] == 2 else (str,)  # noqa
            if not isinstance(theories, string_types):
                raise ImportError("theory must be a python module, a module" +
                                  "name or a python filename")
            # if theories is a filename
            if os.path.isfile(theories):
                sys.path.append(os.path.dirname(theories))
                f = os.path.basename(os.path.splitext(theories)[0])
                theories_module = __import__(f)
            # if theories is a module name
            else:
                theories_module = __import__(theories)

        if hasattr(theories_module, "INIT"):
            theories.INIT()

        if not hasattr(theories_module, "THEORY"):
            msg = "File %s does not contain a THEORY dictionary" % theories
            raise ImportError(msg)

        elif isinstance(theories_module.THEORY, dict):
            # silx format for theory definition
            for theory_name, fittheory in list(theories_module.THEORY.items()):
                self.addtheory(theory_name, fittheory)
        else:
            self._load_legacy_theories(theories_module)

    def setbackground(self, theory):
        """Choose a background type from within :attr:`bgtheories`.

        This updates :attr:`selectedbg`.

        :param theory: The name of the background to be used.
        :raise: KeyError if ``theory`` is not a key of :attr:`bgtheories``.
        """
        if theory in self.bgtheories:
            self.selectedbg = theory
        else:
            msg = "No theory with name %s in bgtheories.\n" % theory
            msg += "Available theories: %s\n" % self.bgtheories.keys()
            raise KeyError(msg)

    def setdata(self, x, y, sigmay=None, xmin=None, xmax=None):
        """Set data attributes:

            - ``xdata0``, ``ydata0`` and ``sigmay0`` store the initial data
              and uncertainties. These attributes are not modified after
              initialization.
            - ``xdata``, ``ydata`` and ``sigmay`` store the data after
              removing values where ``xdata < xmin`` or ``xdata > xmax``.
              These attributes may be modified at a latter stage by filters.

        :param x: Abscissa data. If ``None``, :attr:`xdata`` is set to
            ``numpy.array([0.0, 1.0, 2.0, ..., len(y)-1])``
        :type x: Sequence or numpy array or None
        :param y: The dependant data ``y = f(x)``. ``y`` must have the same
            shape as ``x`` if ``x`` is not ``None``.
        :type y: Sequence or numpy array or None
        :param sigmay: The uncertainties in the ``ydata`` array. These are
            used as weights in the least-squares problem.
            If ``None``, the uncertainties are assumed to be 1.
        :type sigmay: Sequence or numpy array or None
        :param xmin: Lower value of x values to use for fitting
        :param xmax: Upper value of x values to use for fitting
        """
        if y is None:
            self.xdata0 = numpy.array([], numpy.float)
            self.ydata0 = numpy.array([], numpy.float)
            # self.sigmay0 = numpy.array([], numpy.float)
            self.xdata = numpy.array([], numpy.float)
            self.ydata = numpy.array([], numpy.float)
            # self.sigmay = numpy.array([], numpy.float)

        else:
            self.ydata0 = numpy.array(y)
            self.ydata = numpy.array(y)
            if x is None:
                self.xdata0 = numpy.arange(len(self.ydata0))
                self.xdata = numpy.arange(len(self.ydata0))
            else:
                self.xdata0 = numpy.array(x)
                self.xdata = numpy.array(x)

            # default weight
            if sigmay is None:
                self.sigmay0 = None
                self.sigmay = numpy.sqrt(self.ydata) if self.fitconfig["WeightFlag"] else None
            else:
                self.sigmay0 = numpy.array(sigmay)
                self.sigmay = numpy.array(sigmay) if self.fitconfig["WeightFlag"] else None

            # take the data between limits, using boolean array indexing
            if (xmin is not None or xmax is not None) and len(self.xdata):
                xmin = xmin if xmin is not None else min(self.xdata)
                xmax = xmax if xmax is not None else max(self.xdata)
                bool_array = (self.xdata >= xmin) & (self.xdata <= xmax)
                self.xdata = self.xdata[bool_array]
                self.ydata = self.ydata[bool_array]
                self.sigmay = self.sigmay[bool_array] if sigmay is not None else None

    def enableweight(self):
        """This method can be called to set :attr:`sigmay`. If :attr:`sigmay0` was filled with
        actual uncertainties in :meth:`setdata`, use these values.
        Else, use ``sqrt(self.ydata)``.
        """
        if self.sigmay0 is None:
            self.sigmay = numpy.sqrt(self.ydata) if self.fitconfig["WeightFlag"] else None
        else:
            self.sigmay = self.sigmay0

    def disableweight(self):
        """This method can be called to set :attr:`sigmay` equal to ``None``.
        As a result, :func:`leastsq` will consider that the weights in the
        least square problem are 1 for all samples."""
        self.sigmay = None

    def settheory(self, theory):
        """Pick a theory from :attr:`theories`.

        :param theory: Name of the theory to be used.
        :raise: KeyError if ``theory`` is not a key of :attr:`theories`.
        """
        if theory is None:
            self.selectedtheory = None
        elif theory in self.theories:
            self.selectedtheory = theory
        else:
            msg = "No theory with name %s in theories.\n" % theory
            msg += "Available theories: %s\n" % self.theories.keys()
            raise KeyError(msg)

        # run configure to apply our fitconfig to the selected theory
        # through its custom config function
        self.configure(**self.fitconfig)

    def runfit(self, callback=None):
        """Run the actual fitting and fill :attr:`fit_results` with fit results.

        Before running this method, :attr:`fit_results` must already be
        populated with a list of all parameters and their estimated values.
        For this, run :meth:`estimate` beforehand.

        :param callback: Optional callback function, conforming to the
            signature ``callback(data)`` with ``data`` being a dictionary.
            This callback function is called before and after the estimation
            process, and is given a dictionary containing the values of
            :attr:`state` (``'Fit in progress'`` or ``'Ready'``)
            and :attr:`chisq`.
            This is used for instance in :mod:`silx.gui.fit.FitWidget` to
            update a widget displaying a status message.
        :return: Tuple ``(fitted parameters, uncertainties, infodict)``.
            *infodict* is the dictionary returned by
            :func:`silx.math.fit.leastsq` when called with option
            ``full_output=True``. Uncertainties is a sequence of uncertainty
            values associated with each fitted parameter.
        """
        self.dataupdate()

        self.state = 'Fit in progress'
        self.chisq = None

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        param_val = []
        param_constraints = []
        # Initial values are set to the ones computed in estimate()
        for param in self.fit_results:
            param_val.append(param['estimation'])
            param_constraints.append([param['code'], param['cons1'], param['cons2']])

        ywork = self.ydata

        if self.selectedbg == "Square Filter":
            ywork = self.squarefilter(
                    self.ydata, self.fit_results[0]['estimation'])

        try:
            params, covariance_matrix, infodict = leastsq(
                    self.fitfunction,  # bg + actual model function
                    self.xdata, ywork, param_val,
                    sigma=self.sigmay,
                    constraints=param_constraints,
                    model_deriv=self.theories[self.selectedtheory].derivative,
                    full_output=True, left_derivative=True)
        except LinAlgError:
            self.state = 'Fit failed'
            callback(data={'status': self.state})
            raise

        sigmas = infodict['uncertainties']

        for i, param in enumerate(self.fit_results):
            if param['code'] != 'IGNORE':
                param['fitresult'] = params[i]
                param['sigma'] = sigmas[i]

        self.chisq = infodict["reduced_chisq"]
        self.niter = infodict["niter"]
        self.state = 'Ready'

        if callback is not None:
            callback(data={'chisq': self.chisq,
                           'status': self.state})

        return params, sigmas, infodict

    ###################
    # Private methods #
    ###################
    def fitfunction(self, x, *pars):
        """Function to be fitted.

        This is the sum of the selected background function plus
        a number of peak functions.

        :param x: Independent variable where the function is calculated.
        :param pars: Sequence of all fit parameters. The first few parameters
            are background parameters, then come the peak function parameters.
            The total number of fit parameters in ``pars`` will
            be `nb_bg_pars + nb_peak_pars * nb_peaks`.
        :return: Output of the fit function with ``x`` as input and ``pars``
            as fit parameters.
        """
        bg_pars_list = self.bgtheories[self.selectedbg].parameters
        nb_bg_pars = len(bg_pars_list)

        peak_pars_list = self.theories[self.selectedtheory].parameters
        nb_peak_pars = len(peak_pars_list)

        nb_peaks = int((len(pars) - nb_bg_pars) / nb_peak_pars)

        result = numpy.zeros(numpy.shape(x), numpy.float)

        # Compute one peak function per peak, and sum the output numpy arrays
        selectedfun = self.theories[self.selectedtheory].function
        for i in range(nb_peaks):
            start_par_index = nb_bg_pars + i * nb_peak_pars
            end_par_index = nb_bg_pars + (i + 1) * nb_peak_pars
            result += selectedfun(x, *pars[start_par_index:end_par_index])

        if nb_bg_pars > 0:
            bgfun = self.bgtheories[self.selectedbg].function
            result += bgfun(x, *pars[0:nb_bg_pars])
        # TODO: understand and document this Square Filter
        if self.selectedbg == "Square Filter":
            result = result - pars[1]
            return pars[1] + self.squarefilter(result, pars[0])
        else:
            return result

    def estimate_bkg(self, x, y):
        """Estimate background parameters using the function defined in
        the current fit configuration.

        To change the selected background model, attribute :attr:`selectdbg`
        must be changed using method :meth:`setbackground`.

        The actual background function to be used is
        referenced in :attr:`bgtheories`

        :param x: Sequence of x data
        :param y: sequence of y data
        :return: Tuple of two sequences and one data array
            ``(estimated_param, constraints, bg_data)``:

            - ``estimated_param`` is a list of estimated values for each
              background parameter.
            - ``constraints`` is a 2D sequence of dimension ``(n_parameters, 3)``

                - ``constraints[i][0]``: Constraint code.
                  See explanation about codes in :attr:`fit_results`

                - ``constraints[i][1]``
                  See explanation about 'cons1' in :attr:`fit_results`
                  documentation.

                - ``constraints[i][2]``
                  See explanation about 'cons2' in :attr:`fit_results`
                  documentation.
        """
        background_estimate_function = self.bgtheories[self.selectedbg].estimate
        if background_estimate_function is not None:
            return background_estimate_function(x, y)
        else:
            return [], []

    def estimate_fun(self, x, y):
        """Estimate fit parameters using the function defined in
        the current fit configuration.

        :param x: Sequence of x data
        :param y: sequence of y data
        :param bg: Background signal, to be subtracted from ``y`` before fitting.
        :return: Tuple of two sequences ``(estimated_param, constraints)``:

            - ``estimated_param`` is a list of estimated values for each
              background parameter.
            - ``constraints`` is a 2D sequence of dimension (n_parameters, 3)

                - ``constraints[i][0]``: Constraint code.
                  See explanation about codes in :attr:`fit_results`

                - ``constraints[i][1]``
                  See explanation about 'cons1' in :attr:`fit_results`
                  documentation.

                - ``constraints[i][2]``
                  See explanation about 'cons2' in :attr:`fit_results`
                  documentation.
        :raise: ``TypeError`` if estimation function is not callable

        """
        estimatefunction = self.theories[self.selectedtheory].estimate
        if hasattr(estimatefunction, '__call__'):
            if not self.theories[self.selectedtheory].pymca_legacy:
                return estimatefunction(x, y)
            else:
                # legacy pymca estimate functions have a different signature
                if self.fitconfig["fitbkg"] == "No Background":
                    bg = numpy.zeros_like(y)
                else:
                    if self.fitconfig["SmoothStrip"]:
                        y = smooth1d(y)
                    bg = strip(y,
                               w=self.fitconfig["StripWidth"],
                               niterations=self.fitconfig["StripNIterations"],
                               factor=self.fitconfig["StripThresholdFactor"])
                # fitconfig can be filled by user defined config function
                xscaling = self.fitconfig.get('Xscaling', 1.0)
                yscaling = self.fitconfig.get('Yscaling', 1.0)
                return estimatefunction(x, y, bg, xscaling, yscaling)
        else:
            raise TypeError("Estimation function in attribute " +
                            "theories[%s]" % self.selectedtheory +
                            " must be callable.")

    def bkg_constant(self, x, *pars):
        """Constant background function ``y = constant``

        :param x: Abscissa values
        :type x: numpy.ndarray
        :param pars: Background function parameters: ``(constant, )``
        :return: Array of the same shape as ``x`` filled with constant value
        """
        return pars[0] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_linear(self, x, *pars):
        """Linear background function ``y = constant + slope * x``

        :param x: Abscissa values
        :type x: numpy.ndarray
        :param pars: Background function parameters: ``(constant, slope)``
        :return: Array ``y = constant + slope * x``
        """
        return pars[0] + pars[1] * x

    def bkg_strip(self, x, *pars):
        """
        Internal Background based on a strip filter
        (:meth:`silx.math.fit.filters.strip`)

        Parameters are *(strip_width, n_iterations)*

        A 1D smoothing is applied prior to the stripping, if configuration
        parameter ``SmoothStrip`` in :attr:`fitconfig` is ``True``.

        See http://pymca.sourceforge.net/stripbackground.html
        """
        if self._bkg_strip_oldpars[0] == pars[0]:
            if self._bkg_strip_oldpars[1] == pars[1]:
                if (len(x) == len(self._bkg_strip_oldx)) & \
                   (len(self.ydata) == len(self._bkg_strip_oldy)):
                    # same parameters
                    if numpy.sum(self._bkg_strip_oldx == x) == len(x):
                        if numpy.sum(self._bkg_strip_oldy == self.ydata) == len(self.ydata):
                            return self._bkg_strip_oldbkg
        self._bkg_strip_oldy = self.ydata
        self._bkg_strip_oldx = x
        self._bkg_strip_oldpars = pars
        idx = numpy.nonzero((self.xdata >= x[0]) & (self.xdata <= x[-1]))[0]
        yy = numpy.take(self.ydata, idx)
        if self.fitconfig["SmoothStrip"]:
            yy = smooth1d(yy)

        nrx = numpy.shape(x)[0]
        nry = numpy.shape(yy)[0]
        if nrx == nry:
            self._bkg_strip_oldbkg = strip(yy, pars[0], pars[1])
            return self._bkg_strip_oldbkg

        else:
            self._bkg_strip_oldbkg = strip(numpy.take(yy, numpy.arange(0, nry, 2)),
                                           pars[0], pars[1])
            return self._bkg_strip_oldbkg

    def bkg_squarefilter(self, x, *pars):
        """
        Square filter Background
        """
        # TODO: docstring
        # why is this different than bkg_constant?
        # what is pars[0]?
        # what defines the (xmin, xmax) limits of the square function?
        return pars[1] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_none(self, x, *pars):
        """Null background function.

        :param x: Abscissa values
        :type x: numpy.ndarray
        :param pars: Background function parameters. Ignored, only present
            because other methods expect this signature for all background
            functions
        :return: Array of 0 values of the same shape as ``x``
        """
        return numpy.zeros(x.shape, numpy.float)

    def _load_legacy_theories(self, theories_module):
        """Load theories from a custom module in the old PyMca format.

        See PyMca5.PyMcaMath.fitting.SpecfitFunctions for an example.
        """
        mandatory_attributes = ["THEORY", "PARAMETERS",
                                "FUNCTION", "ESTIMATE"]
        err_msg = "Custom fit function file must define: "
        err_msg += ", ".join(mandatory_attributes)
        for attr in mandatory_attributes:
            if not hasattr(theories_module, attr):
                raise ImportError(err_msg)

        derivative = theories_module.DERIVATIVE if hasattr(theories_module, "DERIVATIVE") else None
        configure = theories_module.CONFIGURE if hasattr(theories_module, "CONFIGURE") else None
        estimate = theories_module.ESTIMATE if hasattr(theories_module, "ESTIMATE") else None
        if isinstance(theories_module.THEORY, (list, tuple)):
            # multiple fit functions
            for i in range(len(theories_module.THEORY)):
                deriv = derivative[i] if derivative is not None else None
                config = configure[i] if configure is not None else None
                estim = estimate[i] if estimate is not None else None
                self.addtheory(theories_module.THEORY[i],
                               FitTheory(
                                   theories_module.FUNCTION[i],
                                   theories_module.PARAMETERS[i],
                                   estim,
                                   config,
                                   deriv,
                                   pymca_legacy=True))
        else:
            # single fit function
            self.addtheory(theories_module.THEORY,
                           FitTheory(
                               theories_module.FUNCTION,
                               theories_module.PARAMETERS,
                               estimate,
                               configure,
                               derivative,
                               pymca_legacy=True))

    def estimate_builtin_bkg(self, x, y):
        """Compute the initial parameters for the background function before
        starting the iterative fit.

        The return parameters and constraints depends on the selected theory:

            - ``'Constant'``: [min(background)], constraint FREE
            - ``'Strip'``: [2.000, 5000, 0.0], constraint FIXED
            - ``'No Background'``: empty array []
            - ``'Square Filter'``:
            - ``'Linear'``: [constant, slope], constraint FREE

        :param x: Array of values for the independant variable
        :param y: Array of data values for the dependant data
        :return: Tuple ``(fitted_param, constraints)`` where:

            - ``fitted_param`` is a list of the estimated background
              parameters. The length of the list depends on the theory used
            - ``constraints`` is a numpy array of shape
              *(len(fitted_param), 3)* containing constraint information for
              each parameter *code, cons1, cons2* (see documentation of
              :attr:`fit_results`)
        """
        # TODO: document square filter

        # extract bg by applying a strip filter
        if self.fitconfig["SmoothStrip"]:
            y = smooth1d(y)
        background = strip(y,
                           w=self.fitconfig["StripWidth"],
                           niterations=self.fitconfig["StripNIterations"],
                           factor=self.fitconfig["StripThresholdFactor"])

        npoints = len(background)
        if self.selectedbg == 'Constant':
            # Constant background
            Sy = min(background)
            fittedpar = [Sy]
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)

        elif self.selectedbg == 'Strip':
            # Strip
            fittedpar = [self.fitconfig["StripWidth"],
                         self.fitconfig["StripNIterations"]]
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
            # code = 3: FIXED
            cons[0][0] = 3
            cons[1][0] = 3

        elif self.selectedbg == 'No Background':
            # None
            fittedpar = []
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)

        elif self.selectedbg == 'Square Filter':
            fwhm = self.fitconfig['FwhmPoints']
            # set an odd number
            if fwhm % 2 == 1:
                fittedpar = [fwhm, 0.0]
            else:
                fittedpar = [fwhm + 1, 0.0]
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)
            # code = 3: FIXED
            cons[0][0] = 3
            cons[1][0] = 3

        elif self.selectedbg == 'Linear':
            n = float(npoints)
            Sy = numpy.sum(background)
            Sx = float(numpy.sum(x))
            Sxx = float(numpy.sum(x * x))
            Sxy = float(numpy.sum(x * background))

            deno = n * Sxx - (Sx * Sx)
            if (deno != 0):
                bg = (Sxx * Sy - Sx * Sxy) / deno
                slop = (n * Sxy - Sx * Sy) / deno
            else:
                bg = 0.0
                slop = 0.0
            fittedpar = [bg / 1.0, slop / 1.0]
            # code = 0: FREE
            cons = numpy.zeros((len(fittedpar), 3), numpy.float)

        else:
            # this can happen if someone modifies self.selectedbg without
            # going through the proper channels (method setbackground)
            msg = "Selected background theory %s " % self.selectedbg
            msg += "not a valid theory. Valid theories: "
            msg += str(list(self.bgtheories.keys()))
            raise AttributeError(msg)

        return fittedpar, cons

    def squarefilter(self, y, width):
        """

        :param y:
        :param width:
        :return:
        """ # TODO: document
        if len(y) == 0:
            if isinstance(y, list):
                return []
            else:
                return numpy.array([])

        # make width an odd number of samples and calculate half width
        width = int(width) + ((int(width) + 1) % 2)
        half_width = int(width / 2)

        len_coef = 2 * half_width + width

        if len(y) < len_coef:
            return y

        coef = numpy.zeros((len_coef,), numpy.float)

        coef[0:half_width] = -0.5 / float(half_width)
        coef[half_width:(half_width + width)] = 1.0 / float(width)
        coef[(half_width + width):len(coef)] = -0.5 / float(half_width)

        result = numpy.zeros(len(y), numpy.float)
        result[(width - 1):-(width - 1)] = numpy.convolve(y, coef, 0)
        result[0:width - 1] = result[width - 1]
        result[-(width - 1):] = result[-(width + 1)]
        return result


def test():
    from .functions import sum_gauss
    from . import fittheories

    # Create synthetic data with a sum of gaussian functions
    x = numpy.arange(1000).astype(numpy.float)

    p = [1000, 100., 250,
         255, 690., 45,
         1500, 800.5, 95]
    y = 2.65 * x + 13 + sum_gauss(x, *p)

    # Fitting
    fit = FitManager()
    # more sensitivity necessary to resolve
    # overlapping peaks at x=690 and x=800.5
    fit.setdata(x=x, y=y)
    fit.loadtheories(fittheories)
    fit.settheory('Gaussians')
    fit.setbackground('Linear')
    fit.estimate()
    fit.runfit()

    print("Searched parameters = ", p)
    print("Obtained parameters : ")
    dummy_list = []
    for param in fit.fit_results:
        print(param['name'], ' = ', param['fitresult'])
        dummy_list.append(param['fitresult'])
    print("chisq = ", fit.chisq)

    # Plot
    constant, slope = dummy_list[:2]
    p1 = dummy_list[2:]
    print(p1)
    y2 = slope * x + constant + sum_gauss(x, *p1)

    try:
        from silx.gui import qt
        from silx.gui.plot.PlotWindow import PlotWindow
        app = qt.QApplication([])
        pw = PlotWindow(control=True)
        pw.addCurve(x, y, "Original")
        pw.addCurve(x, y2, "Fit result")
        pw.legendsDockWidget.show()
        pw.show()
        app.exec_()
    except ImportError:
        _logger.warning("Could not import qt to display fit result as curve")


if __name__ == "__main__":
    test()
