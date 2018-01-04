#!/usr/bin/env python 
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------
# Copyright (c) 2013, NeXpy Development Team.
#
# Author: Paul Kienzle, Ray Osborn
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING, distributed with this software.
#-----------------------------------------------------------------------------

"""
This module provides a standard Matplotlib plotting option to the NeXus Python
API
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
from . import NXfield, NeXusError


def centers(axis, dimlen):
    """
    Return the centers of the axis bins.

    This works regardless if the axis contains bin boundaries or centers.
    """
    ax = axis.astype(np.float32)
    if ax.shape[0] == dimlen+1:
        return (ax[:-1] + ax[1:])/2
    else:
        assert ax.shape[0] == dimlen
        return ax


def boundaries(axis, dimlen):
    """
    Return the boundaries of the axis bins.

    This works regardless if the axis contains bin boundaries or centers.
    """
    ax = axis.astype(np.float32)
    if ax.shape[0] == dimlen:
        start = ax[0] - (ax[1] - ax[0])/2
        end = ax[-1] + (ax[-1] - ax[-2])/2
        return np.concatenate((np.atleast_1d(start),
                               (ax[:-1] + ax[1:])/2, 
                               np.atleast_1d(end)))
    else:
        assert ax.shape[0] == dimlen + 1
        return ax


def label(field):
    """
    Return a label for a data field suitable for use on a graph axis.
    """
    if 'long_name' in field.attrs:
        return field.long_name
    elif 'units' in field.attrs:
        return "%s (%s)"%(field.nxname, field.units)
    else:
        return field.nxname


class PylabPlotter(object):

    """
    Matplotlib plotter class for NeXus data.
    """

    def plot(self, data_group, fmt, xmin, xmax, ymin, ymax, zmin, zmax, **opts):
        """
        Plot the data entry.

        Raises NeXusError if the data cannot be plotted.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise NeXusError("Default plotting package (matplotlib) not available.")

        over = opts.pop("over", False)
        image = opts.pop("image", False)
        log = opts.pop("log", False)
        logx = opts.pop("logx", False)
        logy = opts.pop("logy", False)

        signal = data_group.nxsignal
        if signal.ndim > 2 and not image:
            raise NeXusError(
                "Can only plot 1D and 2D data - please select a slice")
        elif signal.ndim > 1 and over:
            raise NeXusError("Cannot overplot 2D data")
        errors = data_group.nxerrors
        title = data_group.nxtitle

        # Provide a new view of the data if there is a dimension of length 1
        data, axes = (signal.nxdata.reshape(data_group.plot_shape), 
                      data_group.plot_axes)

        isinteractive = plt.isinteractive()
        plt.ioff()

        try:
            if over:
                plt.autoscale(enable=False)
            else:
                plt.autoscale(enable=True)
                plt.clf()

            #One-dimensional Plot
            if len(data.shape) == 1:
                if fmt == '': 
                    fmt = 'o'
                if hasattr(signal, 'units'):
                    if not errors and signal.units == 'counts':
                        errors = NXfield(np.sqrt(data))
                if errors:
                    ebars = errors.nxdata
                    plt.errorbar(centers(axes[0], data.shape[0]), data, ebars, 
                                 fmt=fmt, **opts)
                else:
                    plt.plot(centers(axes[0], data.shape[0]), data, fmt, **opts)
                if not over:
                    ax = plt.gca()
                    xlo, xhi = ax.set_xlim(auto=True)        
                    ylo, yhi = ax.set_ylim(auto=True)                
                    if xmin: 
                        xlo = xmin
                    if xmax: 
                        xhi = xmax
                    ax.set_xlim(xlo, xhi)
                    if ymin: 
                        ylo = ymin
                    if ymax: 
                        yhi = ymax
                    ax.set_ylim(ylo, yhi)
                    if logx: 
                        ax.set_xscale('symlog')
                    if log or logy: 
                        ax.set_yscale('symlog')
                    plt.xlabel(label(axes[0]))
                    plt.ylabel(label(signal))
                    plt.title(title)

            #Two dimensional plot
            else:
                from matplotlib.colors import LogNorm, Normalize

                if image:
                    x = boundaries(axes[-2], data.shape[-2])
                    y = boundaries(axes[-3], data.shape[-3])
                    xlabel, ylabel = label(axes[-2]), label(axes[-3])
                else:
                    x = boundaries(axes[-1], data.shape[-1])
                    y = boundaries(axes[-2], data.shape[-2])
                    xlabel, ylabel = label(axes[-1]), label(axes[-2])

                if not zmin: 
                    zmin = np.nanmin(data[data>-np.inf])
                if not zmax: 
                    zmax = np.nanmax(data[data<np.inf])
            
                if not image:
                    if log:
                        zmin = max(zmin, 0.01)
                        zmax = max(zmax, 0.01)
                        opts["norm"] = LogNorm(zmin, zmax)
                    else:
                        opts["norm"] = Normalize(zmin, zmax)

                ax = plt.gca()
                if image:
                    im = ax.imshow(data, **opts)
                    ax.set_aspect('equal')
                else:
                    im = ax.pcolormesh(x, y, data, **opts)
                    im.get_cmap().set_bad('k', 1.0)
                    ax.set_xlim(x[0], x[-1])
                    ax.set_ylim(y[0], y[-1])
                    ax.set_aspect('auto')
                if not image:
                    plt.colorbar(im)
	
                if 'origin' in opts and opts['origin'] == 'lower':
                    image = False
                if xmin: 
                    ax.set_xlim(left=xmin)
                if xmax: 
                    ax.set_xlim(right=xmax)
                if ymin: 
                    if image:
                        ax.set_ylim(top=ymin)
                    else:
                        ax.set_ylim(bottom=ymin)
                if ymax: 
                    if image:
                        ax.set_ylim(bottom=ymax)
                    else:
                        ax.set_ylim(top=ymax)

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)

            if isinteractive:
                plt.pause(0.001)
                plt.show(block=False)
            else:
                plt.show()

        finally:
            if isinteractive:
                plt.ion()

plotview = PylabPlotter()
