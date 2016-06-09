# coding: utf-8
#  /*#########################################################################
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
"""Fitting functions manager

"""
from collections import OrderedDict
import logging
import numpy
import os
import sys

# TODO: remove dependency on pymca
import PyMca5
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaMath.fitting.Gefit import LeastSquaresFit
def curve_fit(model, xdata, ydata, p0, sigma=None,
              constraints=None, model_deriv=None, weightflag=0):
    return LeastSquaresFit(model, p0,
                           xdata=xdata,
                           ydata=ydata,
                           # weightflag=1 if sigma is not None else 0,
                           sigmadata=sigma,
                           constrains=constraints,
                           model_deriv=model_deriv,
                           )
#from .fit import curve_fit
from PyMca5.PyMcaCore import EventHandler


__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "03/06/2016"

_logger = logging.getLogger(__name__)


class Specfit():
    """
    Fitting functions manager

    """

    def __init__(self, x=None, y=None, sigmay=None, auto_fwhm=0, fwhm_points=8,
                 auto_scaling=0, yscaling=1.0, sensitivity=2.5,
                 residuals_flag=0, mca_mode=0, event_handler=None):
        """

        :param x: The independent variable where the data is measured.
        :param y: The dependent data --- nominally f(xdata, ...).
        :param sigmay: The uncertainties in the ``y`` array. These are used as
            weights in the least-squares problem.
            If ``None``, the uncertainties are assumed to be 1
        :param auto_fwhm:
        :param fwhm_points:
        :param auto_scaling:
        :param yscaling:
        :param sensitivity:
        :param residuals_flag:
        :param event_handler:
        """
        self.fitconfig = {}
        self.filterlist = []
        self.filterdict = {}
        self.theorydict = OrderedDict()
        self.dataupdate = None      # FIXME: this seems to be unused. Document or remove it

        self.fitconfig['AutoFwhm'] = auto_fwhm
        self.fitconfig['FwhmPoints'] = fwhm_points
        self.fitconfig['AutoScaling'] = auto_scaling
        self.fitconfig['Yscaling'] = yscaling
        self.fitconfig['Sensitivity'] = sensitivity
        self.fitconfig['ResidualsFlag'] = residuals_flag
        self.fitconfig['McaMode'] = mca_mode

        if event_handler is not None:
            self.eh = event_handler
        else:
            self.eh = EventHandler.EventHandler()

        self.bkgdict = OrderedDict(
            [('No Background', [self.bkg_none, [], None]),
             ('Constant', [self.bkg_constant, ['Constant'],
                           self.estimate_builtin_bkg]),
             ('Linear', [self.bkg_linear, ['Constant', 'Slope'],
                         self.estimate_builtin_bkg]),
             ('Internal', [self.bkg_internal,
                           ['Curvature', 'Iterations', 'Constant'],
                           self.estimate_builtin_bkg])])

        self.fitconfig['fitbkg'] = 'No Background'
        self.bkg_internal_oldx = numpy.array([])
        self.bkg_internal_oldy = numpy.array([])
        self.bkg_internal_oldpars = [0, 0]
        self.bkg_internal_oldbkg = numpy.array([])
        self.fitconfig['fittheory'] = None

        self.setdata(x, y, sigmay)

    def setdata(self, x, y, sigmay=None, xmin=None, xmax=None):
        if y is None:
            self.xdata0 = numpy.array([], numpy.float)
            self.ydata0 = numpy.array([], numpy.float)
            self.sigmay0 = numpy.array([], numpy.float)
            self.xdata = numpy.array([], numpy.float)
            self.ydata = numpy.array([], numpy.float)
            self.sigmay = numpy.array([], numpy.float)

        else:
            self.ydata0 = numpy.array(y)
            self.ydata = numpy.array(y)
            if x is None:
                self.xdata0 = numpy.arange(len(self.ydata0))
                self.xdata = numpy.arange(len(self.ydata0))
            else:
                self.xdata0 = numpy.array(x)
                self.xdata = numpy.array(x)

            if sigmay is None:
                dummy = numpy.sqrt(abs(self.ydata0))
                self.sigmay0 = numpy.reshape(
                    dummy + numpy.equal(dummy, 0), self.ydata0.shape)
                self.sigmay = numpy.reshape(
                    dummy + numpy.equal(dummy, 0), self.ydata0.shape)
            else:
                self.sigmay0 = numpy.array(sigmay)
                self.sigmay = numpy.array(sigmay)

            # take the data between limits, using boolean array indexing
            if (xmin is not None or xmax is not None) and len(self.xdata):
                xmin = xmin if xmin is not None else min(self.xdata)
                xmax = xmax if xmax is not None else max(self.xdata)
                bool_array = (self.xdata >= xmin) & (self.xdata <= xmax)
                self.xdata = self.xdata[bool_array]
                self.ydata = self.ydata[bool_array]
                self.sigmay = self.sigmay[bool_array]

    def filter(self, xwork=None, ywork=None, sigmaywork=None):
        # FIXME: could not find any usage of this method. Return status to be removed?
        if xwork is None:
            xwork = self.xdata0
        if ywork is None:
            ywork = self.ydata0
        if sigmaywork is None:
            sigmaywork = self.sigmay0
        filterstatus = 0
        for i in self.filterlist:
            filterstatus += 1
            try:
                xwork, ywork, sigmaywork = self.filterlist[i][0](
                        xwork,
                        ywork,
                        sigmaywork,
                        self.filterlist[i][1],
                        self.filterlist[i][2])
            except:   # FIXME: except what?
                return filterstatus
        self.xdata = xwork
        self.ydata = ywork
        self.sigmay = sigmaywork
        return filterstatus

    def addfilter(self, filterfun, filtername= "Unknown", *vars, **kw):
        addfilterstatus = 0
        kw['filtername'] = filtername
        self.filterlist.append([filterfun, vars, kw])
        return addfilterstatus

    def deletefilter(self, filter_indices=None, filter_name=None):
        """
        Deletes all specified filters from the internal list of filters
        (``self.filterlist``).

        :param filter_indices: List of indices of filters to be removed
        :param filter_name: Name of filter to be removed

        Filters can be specified as a list of indices and/or a filter name.

        Usage examples::

            self.delete(2)                  # del(self.filterlist[2])
            self.delete(filtername='sort')  # deletes any filter named 'sort'

        """
        indices = filter_indices if filter_indices is not None else []

        if min(indices) < 0:
            raise IndexError("Invalid negative filter indices specified")
        if max(indices) >= len(self.filterlist):
            raise IndexError("Found filter indices larger than length " +
                             "of filter list")
        for i in indices:
            del(self.filterlist[i])

        if filter_name is not None:
            len_before_removal = len(self.filterlist)
            self.filterlist = [item for item in self.filterlist if
                               item[2]['filtername'] != filter_name]
            if len(self.filterlist) == len_before_removal:
                _logger.error("No filter named '%s' found" % filter_name)

    def addtheory(self, theory, function, parameters, estimate=None,
                  configure=None, derivative=None):
        """

        :param theory: String with the name describing the function
        :param function: Actual function
        :param parameters: Parameters names for function ``['p1','p2',…]``
        :param estimate: Initial parameters estimation
        :param configure: Optional function to be called to initialize
            parameters prior to fit
        :param derivative: Optional analytical derivative function.
            Its signature should be ``function(parameter_values, parameter_index, x)``
            See Gefit.py module for more information.
        """
        # FIXME adapt derivative to match signature of :param model_deriv: in silx.mat.fit.curve_fit
        self.theorydict[theory] = [function, parameters,
                                   estimate, configure, derivative]

    def addbackground(self, background, function, parameters, estimate=None):
        """

        :param background: String with the name describing the function
        :param function: Actual function
        :param parameters: Parameters names ['p1','p2','p3',...]
        :param estimate:   The initial parameters estimation function if any
        """
        self.bkgdict[background] = [function, parameters, estimate]

    def settheory(self, theory):
        """

        :param theory: Name of the theory to be used.
            It has to be one of the keys of ``self.theorydict``
        """
        if theory in self.theorydict:
            self.fitconfig['fittheory'] = theory
            self.theoryfun = self.theorydict[theory][0]
            self.modelderiv = None
            if len(self.theorydict[theory]) > 5:
                if self.theorydict[theory][5] is not None:
                    self.modelderiv = self.myderiv

    def setbackground(self, theory):
        """

        :param theory: The name of the background to be used.
            It has to be one of the keys of self.bkgdict
        """
        if theory in self.bkgdict:
            self.fitconfig['fitbkg'] = theory
            self.bkgfun = self.bkgdict[theory][0]

    def fitfunction(self, pars, t):
        nb = len(self.bkgdict[self.fitconfig['fitbkg']][1])
        nu = len(self.theorydict[self.fitconfig['fittheory']][1])
        niter = int((len(pars) - nb) / nu)
        u_term = numpy.zeros(numpy.shape(t), numpy.float)
        if niter > 0:
            for i in range(niter):
                u_term = u_term + \
                    self.theoryfun(
                        pars[(nb + i * nu):(nb + (i + 1) * nu)], t)
        if nb > 0:
            result = self.bkgfun(pars[0:nb], t) + u_term
        else:
            result = u_term

        if self.fitconfig['fitbkg'] == "Square Filter":
            result = result - pars[1]
            return pars[1] + self.squarefilter(result, pars[0])
        else:
            return result

    def estimate(self, mcafit=0):
        """
        Fill the parameters entries with an estimation made on the given data.
        """
        self.state = 'Estimate in progress'
        self.chisq = None
        FitStatusChanged = self.eh.create('FitStatusChanged')
        self.eh.event(FitStatusChanged, data={'chisq': self.chisq,
                                              'status': self.state})

        CONS = ['FREE',
                'POSITIVE',
                'QUOTED',
                'FIXED',
                'FACTOR',
                'DELTA',
                'SUM',
                'IGNORE']

        # make sure data are current
        if self.dataupdate is not None:
            if not mcafit:
                self.dataupdate()

        xx = self.xdata
        yy = self.ydata

        # estimate the background
        esti_bkg = self.estimate_bkg(xx, yy)
        bkg_esti_parameters = esti_bkg[0]
        bkg_esti_constrains = esti_bkg[1]
        try:
            zz = numpy.array(esti_bkg[2])
        except:   # FIXME
            zz = numpy.zeros(numpy.shape(yy), numpy.float)
        # added scaling support
        yscaling = 1.0
        if 'AutoScaling' in self.fitconfig:
            if self.fitconfig['AutoScaling']:
                yscaling = self.guess_yscaling(y=yy)
            else:
                if 'Yscaling' in self.fitconfig:
                    yscaling = self.fitconfig['Yscaling']
                else:
                    self.fitconfig['Yscaling'] = yscaling
        else:
            self.fitconfig['AutoScaling'] = 0
            if 'Yscaling' in self.fitconfig:
                yscaling = self.fitconfig['Yscaling']
            else:
                self.fitconfig['Yscaling'] = yscaling

        # estimate the function
        estimation = self.estimate_fun(
            xx, yy, zz, xscaling=1.0, yscaling=yscaling)
        fun_esti_parameters = estimation[0]
        fun_esti_constrains = estimation[1]
        # estimations are made
        # build the names
        self.final_theory = []
        for i in self.bkgdict[self.fitconfig['fitbkg']][1]:
            self.final_theory.append(i)
        i = 0
        j = 1
        while (i < len(fun_esti_parameters)):
            for k in self.theorydict[self.fitconfig['fittheory']][1]:
                self.final_theory.append(k + "%d" % j)
                i = i + 1
            j = j + 1

        self.paramlist = []
        param = self.final_theory
        j = 0
        i = 0
        k = 0
        xmin = min(xx)
        xmax = max(xx)
        # print "xmin = ",xmin,"xmax = ",xmax
        for pname in self.final_theory:
            if i < len(bkg_esti_parameters):
                self.paramlist.append({'name': pname,
                                       'estimation': bkg_esti_parameters[i],
                                       'group': 0,
                                       'code': CONS[int(bkg_esti_constrains[0][i])],
                                       'cons1': bkg_esti_constrains[1][i],
                                       'cons2': bkg_esti_constrains[2][i],
                                       'fitresult': 0.0,
                                       'sigma': 0.0,
                                       'xmin': xmin,
                                       'xmax': xmax})
                i = i + 1
            else:
                if (j % len(self.theorydict[self.fitconfig['fittheory']][1])) == 0:
                    k = k + 1
                if (CONS[int(fun_esti_constrains[0][j])] == "FACTOR") or \
                   (CONS[int(fun_esti_constrains[0][j])] == "DELTA"):
                    fun_esti_constrains[1][j] += len(bkg_esti_parameters)
                self.paramlist.append({'name': pname,
                                       'estimation': fun_esti_parameters[j],
                                       'group': k,
                                       'code': CONS[int(fun_esti_constrains[0][j])],
                                       'cons1': fun_esti_constrains[1][j],
                                       'cons2': fun_esti_constrains[2][j],
                                       'fitresult': 0.0,
                                       'sigma': 0.0,
                                       'xmin': xmin,
                                       'xmax': xmax})
                j += 1

        self.state = 'Ready to Fit'
        self.chisq = None
        self.eh.event(FitStatusChanged, data={'chisq': self.chisq,
                                              'status': self.state})
        return self.paramlist

    def estimate_bkg(self, xx, yy):
        if self.bkgdict[self.fitconfig['fitbkg']][2] is not None:
            return self.bkgdict[self.fitconfig['fitbkg']][2](xx, yy)
        else:
            return [], [[], [], []]

    def estimate_fun(self, xx, yy, zz, xscaling=1.0, yscaling=None):
        if self.theorydict[self.fitconfig['fittheory']][2] is not None:
            return self.theorydict[self.fitconfig['fittheory']][2](xx,
                                                                   yy,
                                                                   zz,
                                                                   xscaling=xscaling,
                                                                   yscaling=yscaling)
        else:
            return [], [[], [], []]

    def importfun(self, file):
        """Import user defined fit functions defined in an external Python
        source file.

        An example of such a file can be found at
        `https://github.com/vasole/pymca/blob/master/PyMca5/PyMcaMath/fitting/SpecfitFunctions.py`_


        """
        sys.path.append(os.path.dirname(file))
        f = os.path.basename(os.path.splitext(file)[0])
        newfun = __import__(f)
        if hasattr(newfun, "INIT"):
            newfun.INIT()

        theory = newfun.THEORY if hasattr(newfun, "THEORY") else \
            "%s" % file
        parameters = newfun.PARAMETERS
        function = newfun.FUNCTION
        estimate = newfun.ESTIMATE if hasattr(newfun, "ESTIMATE") else None
        derivative = newfun.DERIVATIVE if hasattr(newfun, "DERIVATIVE")\
            else None
        configure = newfun.CONFIGURE if hasattr(newfun, "CONFIGURE") else None

        # if theory is a list, we assume all other fit parameters to be lists
        # of the same length
        if isinstance(theory, list):
            for i in range(len(theory)):
                deriv = derivative[i] if derivative is not None else None
                self.addtheory(theory[i],
                               function[i],
                               parameters[i],
                               estimate[i],
                               configure[i],
                               deriv)
        else:
            self.addtheory(
                theory, function, parameters, estimate, configure, derivative)

    def startfit(self, mcafit=0):
        """
        Launch the fit routine
        """
        if self.dataupdate is not None:
            if not mcafit:
                self.dataupdate()
        FitStatusChanged = self.eh.create('FitStatusChanged')
        self.state = 'Fit in progress'
        self.chisq = None
        self.eh.event(FitStatusChanged, data={'chisq': self.chisq,
                                              'status': self.state})

        param_list = self.final_theory
        length = len(param_list)
        param_val = []
        param_constrains = [[], [], []]
        flagconstrains = 0
        for param in self.paramlist:
            param_val.append(param['estimation'])
            param_constrains[0].append(param['code'])
            param_constrains[1].append(param['cons1'])
            param_constrains[2].append(param['cons2'])

        ywork = self.ydata

        if self.fitconfig['fitbkg'] == "Square Filter":
            ywork = self.squarefilter(
                self.ydata, self.paramlist[0]['estimation'])

        constrains = None if param['code'] in ['FREE', 0, 0.0] else \
            param_constrains

        # FIXME: model_deriv signature is currently model_deriv(parameters, index, x)
        #        it needs to be model_deriv(xdata, parameters, index) when switching to silx.math.fit.curve_fit
        found = curve_fit(self.fitfunction, self.xdata, ywork, param_val,
                          constraints=constrains,
                          model_deriv=self.modelderiv)

        for i, param in enumerate(self.paramlist):
            if param['code'] != 'IGNORE':
                param['fitresult'] = found[0][i]
                param['sigma'] = found[2][i]

        self.chisq = found[1]
        self.state = 'Ready'
        self.eh.event(FitStatusChanged, data={'chisq': self.chisq,
                                              'status': self.state})

    def myderiv(self, param0, index, t0):
        """Apply derivative function. If no derivative function is provided
        for the chosen theory, use :meth:`num_deriv`

        """
        nb = len(self.bkgdict[self.fitconfig['fitbkg']][1])
        if index >= nb:
            if len(self.theorydict[self.fitconfig['fittheory']]) > 5:
                if self.theorydict[self.fitconfig['fittheory']][5] is not None:
                    return self.theorydict[self.fitconfig['fittheory']][5](param0, index - nb, t0)
                else:
                    return self.num_deriv(param0, index, t0)
            else:
                return self.num_deriv(param0, index, t0)
        else:
            return self.num_deriv(param0, index, t0)

    def num_deriv(self, param0, index, t0):
        # numerical derivative
        x = numpy.array(t0)
        delta = (param0[index] + numpy.equal(param0[index], 0.0)) * 0.00001
        newpar = param0.__copy__()
        newpar[index] = param0[index] + delta
        f1 = self.fitfunction(newpar, x)
        newpar[index] = param0[index] - delta
        f2 = self.fitfunction(newpar, x)
        return (f1 - f2) / (2.0 * delta)

    def gendata(self, x=None, paramlist=None):
        if x is None:
            x=self.xdata
        if paramlist is None:
            paramlist=self.paramlist
        noigno = []
        for param in paramlist:
            if param['code'] != 'IGNORE':
                noigno.append(param['fitresult'])

        newdata = self.fitfunction(noigno, numpy.array(x))
        return newdata

    def bkg_constant(self, pars, x):
        """
        Constant background
        """
        return pars[0] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_linear(self, pars, x):
        """
        Linear background
        """
        return pars[0] + pars[1] * x

    def bkg_internal(self, pars, x):
        """
        Internal Background
        """
        # fast
        # return self.zz
        # slow: recalculate the background as function of the parameters
        # yy=SpecfitFuns.subac(self.ydata*self.fitconfig['Yscaling'],
        #                     pars[0],pars[1])
        if self.bkg_internal_oldpars[0] == pars[0]:
            if self.bkg_internal_oldpars[1] == pars[1]:
                if (len(x) == len(self.bkg_internal_oldx)) & \
                   (len(self.ydata) == len(self.bkg_internal_oldy)):
                    # same parameters
                    if numpy.sum(self.bkg_internal_oldx == x) == len(x):
                        if numpy.sum(self.bkg_internal_oldy == self.ydata) == len(self.ydata):
                            return self.bkg_internal_oldbkg + pars[2] * numpy.ones(numpy.shape(x), numpy.float)
        self.bkg_internal_oldy = self.ydata
        self.bkg_internal_oldx = x
        self.bkg_internal_oldpars = pars
        idx = numpy.nonzero( (self.xdata >= x[0]) & (self.xdata <= x[-1]))[0]
        yy = numpy.take(self.ydata, idx)
        nrx = numpy.shape(x)[0]
        nry = numpy.shape(yy)[0]
        if nrx == nry:
            self.bkg_internal_oldbkg = SpecfitFuns.subac(yy, pars[0], pars[1])
            return self.bkg_internal_oldbkg + pars[2] * numpy.ones(numpy.shape(x), numpy.float)

        else:
            self.bkg_internal_oldbkg = SpecfitFuns.subac(numpy.take(yy, numpy.arange(0, nry, 2)),
                                                         pars[0], pars[1])
            return self.bkg_internal_oldbkg + pars[2] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_squarefilter(self, pars, x):
        """
        Square filter Background
        """
        return pars[1] * numpy.ones(numpy.shape(x), numpy.float)

    def bkg_none(self, pars, x):
        """
        Internal Background
        """
        return numpy.zeros(x.shape, numpy.float)

    def estimate_builtin_bkg(self, xx, yy):
        self.zz = SpecfitFuns.subac(yy, 1.0001, 1000)
        zz = self.zz
        npoints = len(zz)
        if self.fitconfig['fitbkg'] == 'Constant':
            # Constant background
            S = float(npoints)
            Sy = min(zz)
            fittedpar = [Sy]
            cons = numpy.zeros((3, len(fittedpar)), numpy.float)
        elif self.fitconfig['fitbkg'] == 'Internal':
            # Internal
            fittedpar = [1.000, 10000, 0.0]
            cons = numpy.zeros((3, len(fittedpar)), numpy.float)
            cons[0][0] = 3
            cons[0][1] = 3
            cons[0][2] = 3
        elif self.fitconfig['fitbkg'] == 'No Background':
            # None
            fittedpar = []
            cons = numpy.zeros((3, len(fittedpar)), numpy.float)
        elif self.fitconfig['fitbkg'] == 'Square Filter':
            fwhm = 5
            fwhm = self.fitconfig['FwhmPoints']

            # set an odd number
            if fwhm % 2:
                fittedpar = [fwhm, 0.0]
            else:
                fittedpar = [fwhm + 1, 0.0]
            cons = numpy.zeros((3, len(fittedpar)), numpy.float)
            cons[0][0] = 3
            cons[0][1] = 3
        else:
            S = float(npoints)
            Sy = numpy.sum(zz)
            Sx = float(numpy.sum(xx))
            Sxx = float(numpy.sum(xx * xx))
            Sxy = float(numpy.sum(xx * zz))

            deno = S * Sxx - (Sx * Sx)
            if (deno != 0):
                bg = (Sxx * Sy - Sx * Sxy) / deno
                slop = (S * Sxy - Sx * Sy) / deno
            else:
                bg = 0.0
                slop = 0.0
            fittedpar = [bg / 1.0, slop / 1.0]
            cons = numpy.zeros((3, len(fittedpar)), numpy.float)
        return fittedpar, cons, zz

    def configure(self, **kw):
        """
        Configure the current theory passing a dictionary to the supply method
        """
        for key in self.fitconfig.keys():
            if key in kw:
                self.fitconfig[key] = kw[key]
        result = {}
        result.update(self.fitconfig)
        if self.fitconfig['fittheory'] is not None:
            if self.fitconfig['fittheory'] in self.theorydict.keys():
                if self.theorydict[self.fitconfig['fittheory']][3] is not None:
                    result.update(self.theorydict[
                                  self.fitconfig['fittheory']][3](**kw))
                    # take care of possible user interfaces
                    for key in self.fitconfig.keys():
                        if key in result:
                            self.fitconfig[key] = result[key]
        # make sure fitconfig is configured in case of having the same keys
        for key in self.fitconfig.keys():
            if key in kw:
                self.fitconfig[key] = kw[key]
            if key == "fitbkg":
                self.setbackground(self.fitconfig[key])
            if key == "fittheory":
                if self.fitconfig['fittheory'] is not None:
                    self.settheory(self.fitconfig[key])

        result.update(self.fitconfig)
        return result

    def mcafit(self, x=None, y=None, sigmay=None, yscaling=None,
               sensitivity=None, fwhm_points=None, **kw):
        # TODO: remove this debugging error after investigating usage of this method
        if len(kw):
            raise ValueError("Key not handled:" + str(list(kw.keys())))

        if y is None:
            y = self.ydata0

        if x is None:
            x = numpy.arange(len(y)).astype(numpy.float)

        self.setdata(x, y, sigmay)

        if yscaling is None:
            if self.fitconfig['AutoScaling']:
                yscaling = self.guess_yscaling()
            else:
                yscaling = self.fitconfig['Yscaling']

        if sensitivity is None:
            sensitivity = self.fitconfig['Sensitivity']

        if fwhm_points is None:
            if self.fitconfig['AutoFwhm']:
                fwhm = self.guess_fwhm(y=y)
            else:
                fwhm = self.fitconfig['FwhmPoints']


        fwhm = int(fwhm)

        # needed to make sure same peaks are found
        self.configure(Yscaling=yscaling,
                       # lowercase on purpose
                       autoscaling=0,
                       FwhmPoints=fwhm,
                       Sensitivity=sensitivity)
        ysearch = self.ydata * yscaling
        npoints = len(ysearch)
        peaks = []
        if npoints > (1.5) * fwhm:
            peaksidx = SpecfitFuns.seek(ysearch, 0, npoints,
                                        fwhm,
                                        sensitivity)
            for idx in peaksidx:
                peaks.append(self.xdata[int(idx)])
            _logger.debug("MCA Found peaks = " + str(peaks))
        if len(peaks):
            regions = self.mcaregions(peaks, self.xdata[fwhm] - self.xdata[0])
        else:
            regions = []
        _logger.debug(" regions = " + str(regions))
        mcaresult = []
        xmin0 = self.xdata[0]
        xmax0 = self.xdata[-1]
        for region in regions:
            self.setdata(self.xdata0, self.ydata0, self.sigmay0,
                         xmin=region[0], xmax=region[1])

            self.estimate(mcafit=1)
            if self.state == 'Ready to Fit':
                self.startfit(mcafit=1)
                if self.chisq is not None:
                    if self.fitconfig['ResidualsFlag']:
                        while(self.chisq > 2.5):
                            newpar, newcons = self.mcaresidualssearch()
                            if newpar != []:
                                newg = 1
                                for param in self.paramlist:
                                    newg = max(
                                        newg, int(float(param['group']) + 1))
                                    param['estimation'] = param['fitresult']
                                i = -1
                                for pname in self.theorydict[self.fitconfig['fittheory']][1]:
                                    i = i + 1
                                    name = pname + "%d" % newg
                                    self.paramlist.append({'name': name,
                                                           'estimation': newpar[i],
                                                           'group': newg,
                                                           'code': newcons[0][i],
                                                           'cons1': newcons[1][i],
                                                           'cons2': newcons[2][i],
                                                           'fitresult': 0.0,
                                                           'sigma': 0.0})
                                self.startfit()
                            else:
                                break
            mcaresult.append(self.mcagetresult())
            self.setdata(self.xdata0, self.ydata0, xmin=xmin0, xmax=xmax0)
        return mcaresult

    def mcaregions(self, peaks, fwhm):
        mindelta = 3.0 * fwhm
        plusdelta = 3.0 * fwhm
        regions = []
        xdata0 = min(self.xdata[0], self.xdata[-1])
        xdata1 = max(self.xdata[0], self.xdata[-1])
        for peak in peaks:
            x0 = max(peak - mindelta, xdata0)
            x1 = min(peak + plusdelta, xdata1)
            if regions == []:
                regions.append([x0, x1])
            else:
                if x0 < regions[-1][0]:
                    regions[-1][0] = x0
                elif x0 < (regions[-1][1]):
                    regions[-1][1] = x1
                else:
                    regions.append([x0, x1])
        return regions

    def mcagetresult(self):
        result = {}
        result['xbegin'] = min(self.xdata[0], self.xdata[-1])
        result['xend'] = max(self.xdata[0], self.xdata[-1])
        try:
            result['fitstate'] = self.state
        except:
            result['fitstate'] = 'Unknown'
        result['fitconfig'] = self.fitconfig
        result['config'] = self.configure()
        result['paramlist'] = self.paramlist
        result['chisq'] = self.chisq
        result['mca_areas'] = self.mcagetareas()

        return result

    def mcagetareas(self, x=None, y=None, sigmay=None, paramlist=None):
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        if sigmay is None:
            sigmay = self.sigmay
        if paramlist is None:
            paramlist = self.paramlist
        groups = []
        for param in paramlist:
            if param['code'] != 'IGNORE':
                if int(float(param['group'])) != 0:
                    if param['group'] not in groups:
                        groups.append(param['group'])

        result = []
        for group in groups:
            noigno = []
            pos = 0
            fwhm = 0
            for param in paramlist:
                if param['group'] != group:
                    if param['code'] != 'IGNORE':
                        noigno.append(param['fitresult'])
                else:
                    if 'Position' in param['name']:
                        pos = param['fitresult']
                    if 'Fwhm' in param['name']:
                        fwhm = param['fitresult']
            # now I add everything around +/- 4 sigma
            # around the peak position
            sigma = fwhm / 2.354
            xmin = max(pos - 3.99 * sigma, min(x))
            xmax = min(pos + 3.99 * sigma, max(x))
            idx = numpy.nonzero((x >= xmin) & (x <= xmax))[0]
            x_around = numpy.take(x, idx)
            y_around = numpy.take(y, idx)
            ybkg_around = numpy.take(self.fitfunction(noigno, x), idx)
            neto = y_around - ybkg_around
            deltax = x_around[1:] - x_around[0:-1]
            area = numpy.sum(neto[0:-1] * deltax)
            sigma_area = (numpy.sqrt(numpy.sum(y_around)))
            result.append([pos, area, sigma_area, fwhm])

        return result

    def guess_yscaling(self, y=None):
        if y is None:
            y = self.ydata

        zz = SpecfitFuns.subac(y, 1.0, 10)

        zz = numpy.convolve(y, [1., 1., 1.]) / 3.0
        yy = y[1:-1]
        yfit = zz
        idx = numpy.nonzero(numpy.fabs(yy) > 0.0)[0]
        yy = numpy.take(yy, idx)
        yfit = numpy.take(yfit, idx)

        try:
            chisq = numpy.sum(((yy - yfit) * (yy - yfit)) /
                              (numpy.fabs(yy) * len(yy)))
            scaling = 1. / chisq
        except ZeroDivisionError:
            scaling = 1.0
        return scaling

    def guess_fwhm(self, x=None, y=None):
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        # set at least a default value for the fwhm
        fwhm = 4

        zz = SpecfitFuns.subac(y, 1.000, 1000)
        yfit = y - zz

        # now I should do some sort of peak search ...
        maximum = max(yfit)
        idx = numpy.nonzero(yfit == maximum)[0]
        pos = numpy.take(x, idx)[-1]
        posindex = idx[-1]
        height = yfit[posindex]
        imin = posindex
        while yfit[imin] > 0.5 * height and imin > 0:
            imin -= 1
        imax = posindex
        while yfit[imax] > 0.5 * height and imax < len(yfit) - 1:
            imax += 1
        fwhm = max(imax - imin - 1, fwhm)

        return fwhm

    def mcaresidualssearch(self, x=None, y=None, sigmay=None, paramlist=None):
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        if sigmay is None:
            sigmay = self.sigmay
        if paramlist is None:
            paramlist = self.paramlist

        groups = []
        for param in paramlist:
            if param['code'] != 'IGNORE':
                if int(float(param['group'])) != 0:
                    if param['group'] not in groups:
                        groups.append(param['group'])

        newpar = []
        newcodes = [[], [], []]
        if self.fitconfig['fitbkg'] == 'Square Filter':
            y = self.squarefilter(y, paramlist[0]['estimation'])
            return newpar, newcodes
        areanotdone = 1

        # estimate the fwhm
        fwhm = 10
        fwhmcode = 'POSITIVE'
        fwhmcons1 = 0
        fwhmcons2 = 0
        i = -1
        peaks = []
        for param in paramlist:
            i = i + 1
            pname = param['name']
            if 'Fwhm' in param['name']:
                fwhm = param['fitresult']
                if param['code'] in ['FREE', 'FIXED', 'QUOTED', 'POSITIVE',
                                     0, 1, 2, 3]:
                    fwhmcode = 'FACTOR'
                    fwhmcons1 = i
                    fwhmcons2 = 1.0
            if pname.find('Position') != -1:
                peaks.append(param['fitresult'])

        # calculate the residuals
        yfit = self.gendata(x=x, paramlist=paramlist)

        residuals = (y - yfit) / (sigmay + numpy.equal(sigmay, 0.0))

        # set to zero all the residuals around peaks
        for peak in peaks:
            idx = numpy.less(x, peak - 0.8 * fwhm) + \
                numpy.greater(x, peak + 0.8 * fwhm)
            yfit *= idx
            y *= idx
            residuals *= idx

        # estimate the position
        maxres = max(residuals)
        idx = numpy.nonzero(residuals == maxres)[0]
        pos = numpy.take(x, idx)[-1]

        # estimate the height!
        height = numpy.take(y - yfit, idx)[-1]
        if (height <= 0):
            return newpar, newcodes

        for pname in self.theorydict[self.fitconfig['fittheory']][1]:
            if pname.find('Position') != -1:
                estimation = pos
                code = 'QUOTED'
                cons1 = pos - 0.5 * fwhm
                cons2 = pos + 0.5 * fwhm
            elif pname.find('Area') != -1:
                if areanotdone:
                    areanotdone = 0
                    area = (height * fwhm / (2.0 * numpy.sqrt(2 * numpy.log(2)))) * \
                        numpy.sqrt(2 * numpy.pi)
                    if area <= 0:
                        return [], [[], [], []]
                    estimation = area
                    code = 'POSITIVE'
                    cons1 = 0.0
                    cons2 = 0.0
                else:
                    estimation = 0.0
                    code = 'FIXED'
                    cons1 = 0.0
                    cons2 = 0.0
            elif 'Fwhm' in pname:
                estimation = fwhm
                code = fwhmcode
                cons1 = fwhmcons1
                cons2 = fwhmcons2
            else:
                estimation = 0.0
                code = 'FIXED'
                cons1 = 0.0
                cons2 = 0.0
            newpar.append(estimation)
            newcodes[0].append(code)
            newcodes[1].append(cons1)
            newcodes[2].append(cons2)
        return newpar, newcodes

    def mcaresidualssearch_old(self, x=None, y=None, sigmay=None, paramlist=None):
        if x is None:
            x = self.xdata
        if y is None:
            y = self.ydata
        if sigmay is None:
            sigmay = self.sigmay
        if paramlist is None:
            paramlist = self.paramlist

        areanotdone = 1
        newg = 1
        for param in self.paramlist:
            newg = max(newg, int(float(param['group']) + 1))
        if newg == 1:
            return areanotdone

        # estimate the fwhm
        fwhm = 10
        fwhmcode = 'POSITIVE'
        fwhmcons1 = 0
        fwhmcons2 = 0
        i = -1
        peaks = []
        for param in paramlist:
            i = i + 1
            pname = param['name']
            if 'Fwhm' in pname:
                fwhm = param['fitresult']
                if param['code'] in ['FREE', 'FIXED', 'QUOTED', 'POSITIVE',
                                     0, 1, 2, 3]:
                    fwhmcode = 'FACTOR'
                    fwhmcons1 = i
                    fwhmcons2 = 1.0
            if "Position" in pname:
                peaks.append(param['fitresult'])

        # calculate the residuals
        yfit = self.gendata(x=x, paramlist=paramlist)
        residuals = (y - yfit) / (sigmay + numpy.equal(sigmay, 0.0))

        # set to zero all the residuals around peaks
        for peak in peaks:
            idx = numpy.less(x, peak - 0.8 * fwhm) + \
                numpy.greater(x, peak + 0.8 * fwhm)
            yfit *= idx
            y *= idx
            residuals *= idx

        # estimate the position
        maxres = max(residuals)
        idx = numpy.nonzero(residuals == maxres)[0]
        pos = numpy.take(x, idx)[-1]

        # estimate the height!
        height = numpy.take(y - yfit, idx)[-1]

        for pname in self.theorydict[self.fitconfig['fittheory']][1]:
            self.final_theory.append(pname)
            if 'Fwhm' in pname:
                estimation = fwhm
                code = fwhmcode
                cons1 = fwhmcons1
                cons2 = fwhmcons2
            else:
                estimation = 0.0
                code = 'FIXED'
                cons1 = 0.0
                cons2 = 0.0
            paramlist.append({'name': pname,
                              'estimation': estimation,
                              'group': newg,
                              'code': code,
                              'cons1': cons1,
                              'cons2': cons2,
                              'fitresult': 0.0,
                              'sigma': 0.0})
        return areanotdone

    def squarefilter(self, y, width):
        w = int(width) + ((int(width) + 1) % 2)
        u = int(w / 2)
        coef = numpy.zeros((2 * u + w), numpy.float)
        coef[0:u] = -0.5 / float(u)
        coef[u:(u + w)] = 1.0 / float(w)
        coef[(u + w):len(coef)] = -0.5 / float(u)
        if len(y) == 0:
            if type(y) == type([]):
                return []
            else:
                return numpy.array([])
        else:
            if len(y) < len(coef):
                return y
            else:
                result = numpy.zeros(len(y), numpy.float)
                result[(w - 1):-(w - 1)] = numpy.convolve(y, coef, 0)
                result[0:w - 1] = result[w - 1]
                result[-(w - 1):] = result[-(w + 1)]
                return result


def test():
    from PyMca5.PyMcaMath.fitting import SpecfitFunctions
    a = SpecfitFunctions.SpecfitFunctions()
    x = numpy.arange(1000).astype(numpy.float)
    constant_background = 3.14
    p1 = numpy.array([1500, 100., 50])
    p2 = numpy.array([1000, 700., 30.5])
    p3 = numpy.array([314, 800.5, 15])
    y = constant_background + a.gauss(p1, x) + a.gauss(p2, x) + a.gauss(p3, x)
    fit = Specfit()
    fit.setdata(x=x, y=y)
    #fit.importfun(os.path.join(os.path.dirname(
    #    __file__), "SpecfitFunctions.py"))
    fit.importfun(PyMca5.PyMcaMath.fitting.SpecfitFunctions.__file__)  # FIXME: beurk
    fit.settheory('Gaussians')
    fit.setbackground('Constant')
    if 1:
        fit.estimate()
        fit.startfit()
    else:
        fit.mcafit()
    print("Searched parameters = ", [3.14, 1500, 100., 50.0, 1000, 700., 30.5, 314, 800.5, 15])
    print("Obtained parameters : ")
    dummy_list = []
    for param in fit.paramlist:
        print(param['name'], ' = ', param['fitresult'])
        dummy_list.append(param['fitresult'])
    print("chisq = ", fit.chisq)

    constant_background = dummy_list[0]
    p1 = numpy.array(dummy_list[1:4])
    p2 = numpy.array(dummy_list[4:7])
    p3 = numpy.array(dummy_list[7:10])
    y2 = constant_background + a.gauss(p1, x) + a.gauss(p2, x) + a.gauss(p3, x)

    from silx.gui import qt
    from silx.gui.plot.PlotWindow import PlotWindow
    app = qt.QApplication([])
    pw = PlotWindow(control=True)
    pw.addCurve(x, y, "Original")
    pw.addCurve(x, y2, "Fit result")
    pw.legendsDockWidget.show()
    pw.show()
    app.exec_()

if __name__ == "__main__":
    test()
