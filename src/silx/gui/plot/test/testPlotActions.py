# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Test of actions integrated in the plot window"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "09/11/2018"


import pytest
import weakref

from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.plot.PlotWindow import PlotWindow

import numpy


@pytest.fixture
def colormap1():
    colormap = Colormap(name='gray',
                        vmin=10.0, vmax=20.0,
                        normalization='linear')
    yield colormap


@pytest.fixture
def colormap2():
    colormap = Colormap(name='red',
                        vmin=10.0, vmax=20.0,
                        normalization='linear')
    yield colormap


@pytest.fixture
def plot(qapp):
    plot = PlotWindow()
    plot.setAttribute(qt.Qt.WA_DeleteOnClose)
    yield weakref.proxy(plot)
    plot.close()
    qapp.processEvents()


def test_action_active_colormap(qapp_utils, plot, colormap1, colormap2):
    plot.getColormapAction()._actionTriggered(checked=True)
    colormapDialog = plot.getColormapAction()._dialog

    defaultColormap = plot.getDefaultColormap()
    assert colormapDialog.getColormap() is defaultColormap

    plot.addImage(data=numpy.random.rand(10, 10), legend='img1',
                  origin=(0, 0),
                  colormap=colormap1)
    plot.setActiveImage('img1')
    assert colormapDialog.getColormap() is colormap1

    plot.addImage(data=numpy.random.rand(10, 10), legend='img2',
                  origin=(0, 0), colormap=colormap2)
    plot.addImage(data=numpy.random.rand(10, 10), legend='img3',
                  origin=(0, 0))

    plot.setActiveImage('img3')
    assert colormapDialog.getColormap() is defaultColormap
    plot.getActiveImage().setColormap(colormap2)
    assert colormapDialog.getColormap() is colormap2

    plot.remove('img2')
    plot.remove('img3')
    plot.remove('img1')
    assert colormapDialog.getColormap() is defaultColormap


def test_action_show_hide_colormap_dialog(qapp_utils, plot, colormap1):
    plot.getColormapAction()._actionTriggered(checked=True)
    colormapDialog = plot.getColormapAction()._dialog

    plot.getColormapAction()._actionTriggered(checked=False)
    assert not plot.getColormapAction().isChecked()
    plot.getColormapAction()._actionTriggered(checked=True)
    assert plot.getColormapAction().isChecked()
    plot.addImage(data=numpy.random.rand(10, 10), legend='img1',
                  origin=(0, 0), colormap=colormap1)
    colormap1.setName('red')
    plot.getColormapAction()._actionTriggered()
    colormap1.setName('blue')
    colormapDialog.close()
    assert not plot.getColormapAction().isChecked()
