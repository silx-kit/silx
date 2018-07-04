# coding: utf-8
# /*#########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
    - providing different background models

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
from . import bgtheories


__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "16/01/2017"

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
            'WeightFlag': weight_flag,
            'fitbkg': 'No Background',
            'fittheory': None,
            # Next few parameters are defined for compatibility with legacy theories
            # which take the background as argument for their estimation function
            'StripWidth': 2,
            'StripIterations': 5000,
            'StripThresholdFactor': 1.0,
            'SmoothingFlag': False
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

        self.bgtheories = OrderedDict()
        """Dictionary of background theories.

        See :attr:`theories` for documentation on theories.
        """

        # Load default theories (constant, linear, strip)
        self.loadbgtheories(bgtheories)

        self.selectedbg = 'No Background'
        """Name of currently selected background theory. This name must be
        an existing key in :attr:`bgtheories`."""

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
                  description=None, pymca_legacy=False):
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
        :param callable function: Mandatory argument if ``theory`` is not provided.
            See documentation for :attr:`silx.math.fit.fittheory.FitTheory.function`.
        :param List[str] parameters: Mandatory argument if ``theory`` is not provided.
            See documentation for :attr:`silx.math.fit.fittheory.FitTheory.parameters`.
        :param callable estimate: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.estimate`
        :param callable configure: See documentation for
            :attr:`silx.math.fit.fittheory.FitTheory.configure`
        :param callable derivative: See documentation for
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
                pymca_legacy=pymca_legacy
            )

        else:
            raise TypeError("You must supply a FitTheory object or define " +
                            "a fit function and its parameters.")

    def addbgtheory(self, name, theory=None,
                    function=None, parameters=None,
                    estimate=None, configure=None,
                    derivative=None, description=None):
        """Add a new theory to dictionary :attr:`bgtheories`.

        You can pass a name and a :class:`FitTheory` object as arguments, or
        alternatively provide all arguments necessary to instantiate a new
        :class:`FitTheory` object.

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
        """
        if theory is not None:
            self.bgtheories[name] = theory

        elif function is not None and parameters is not None:
            self.bgtheories[name] = FitTheory(
                description=description,
                function=function,
                parameters=parameters,
                estimate=estimate,
                configure=configure,
                derivative=derivative,
                is_background=True
            )

        else:
            raise TypeError("You must supply a FitTheory object or define " +
                            "a background function and its parameters.")

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

        custom_bg_config_fun = self.bgtheories[self.selectedbg].configure
        if custom_bg_config_fun is not None:
            result.update(custom_bg_config_fun(**kw))

        # Update self.fitconfig with custom config
        for key in self.fitconfig:
            if key in result:
                self.fitconfig[key] = result[key]

        result.update(self.fitconfig)
        return result

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

    def gendata(self, x=None, paramlist=None, estimated=False):
        """Return a data array using the currently selected fit function
        and the fitted parameters.

        :param x: Independent variable where the function is calculated.
            If ``None``, use :attr:`xdata`.
        :param paramlist: List of dictionaries, each dictionary item being a
            fit parameter. The dictionary's format is documented in
            :attr:`fit_results`.
            If ``None`` (default), use parameters from :attr:`fit_results`.
        :param estimated: If *True*, use estimated parameters.
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
                if not estimated:
                    active_params.append(param['fitresult'])
                else:
                    active_params.append(param['estimation'])

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

    def loadbgtheories(self, theories):
        """Import user defined background functions defined in an external Python
        module (source file), and save them in :attr:`theories`.

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
                'theory_name_2':
                    FitTheory(…),
            }

        See documentation of :mod:`silx.math.fit.bgtheories` and
        :mod:`silx.math.fit.fittheory` for more
        information on designing your background functions file.

        :param theories: Module or name of python source file containing the
            definition of background functions.
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
                self.addbgtheory(theory_name, fittheory)

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

        # run configure to apply our fitconfig to the selected theory
        # through its custom config function
        self.configure(**self.fitconfig)

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
        # self.dataupdate()

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
        the selected fit model function.

        :param x: Independent variable where the function is calculated.
        :param pars: Sequence of all fit parameters. The first few parameters
            are background parameters, then come the peak function parameters.
        :return: Output of the fit function with ``x`` as input and ``pars``
            as fit parameters.
        """
        result = numpy.zeros(numpy.shape(x), numpy.float)

        if self.selectedbg is not None:
            bg_pars_list = self.bgtheories[self.selectedbg].parameters
            nb_bg_pars = len(bg_pars_list)

            bgfun = self.bgtheories[self.selectedbg].function
            result += bgfun(x, self.ydata, *pars[0:nb_bg_pars])
        else:
            nb_bg_pars = 0

        selectedfun = self.theories[self.selectedtheory].function
        result += selectedfun(x, *pars[nb_bg_pars:])

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
                    if self.fitconfig["SmoothingFlag"]:
                        y = smooth1d(y)
                    bg = strip(y,
                               w=self.fitconfig["StripWidth"],
                               niterations=self.fitconfig["StripIterations"],
                               factor=self.fitconfig["StripThresholdFactor"])
                # fitconfig can be filled by user defined config function
                xscaling = self.fitconfig.get('Xscaling', 1.0)
                yscaling = self.fitconfig.get('Yscaling', 1.0)
                return estimatefunction(x, y, bg, xscaling, yscaling)
        else:
            raise TypeError("Estimation function in attribute " +
                            "theories[%s]" % self.selectedtheory +
                            " must be callable.")

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


def test():
    from .functions import sum_gauss
    from . import fittheories
    from . import bgtheories

    # Create synthetic data with a sum of gaussian functions
    x = numpy.arange(1000).astype(numpy.float)

    p = [1000, 100., 250,
         255, 690., 45,
         1500, 800.5, 95]
    y = 0.5 * x + 13 + sum_gauss(x, *p)

    # Fitting
    fit = FitManager()
    # more sensitivity necessary to resolve
    # overlapping peaks at x=690 and x=800.5
    fit.setdata(x=x, y=y)
    fit.loadtheories(fittheories)
    fit.settheory('Gaussians')
    fit.loadbgtheories(bgtheories)
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
