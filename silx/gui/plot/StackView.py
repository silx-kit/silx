# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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
"""QWidget displaying a 3D volume as a stack of 2D images.

The :class:`StackView` implements this widget, and
:class:`StackViewMainWindow` provides a main window with additional toolbar
and status bar.

Basic usage of :class:`StackView` is through the following methods:

- :meth:`StackView.getColormap`, :meth:`StackView.setColormap` to update the
  default colormap to use and update the currently displayed image.
- :meth:`StackView.setVolume` to update the displayed image.

The :class:`StackView` uses :class:`PlotWindow` and also
exposes a subset of the :class:`silx.gui.plot.Plot` API for further control
(plot title, axes labels, ...).


"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "07/12/2016"

import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D   # ??? or PlotWindow, to redefine ProfileToolBar
# from silx.gui.plot.PlotTools import ProfileToolBar
from silx.gui.plot.Colors import cursorColorForColormap
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser




# TODO: Methods to be referenced from plot
#  - getSupportedColormaps
#  - clear
#  - getGraphTitle() / setGraphTitle(title='')
#  - getGraphXLabel() / setGraphXLabel(label='X')
#  - getGraphYLabel(axis='left') / setGraphYLabel(label='Y', axis='left') (remove axis param?)
#  - setYAxisInverted(flag=True) / isYAxisInverted()
#  - isXAxisLogarithmic()/setXAxisLogarithmic(flag)    ??????????????????
#  - isKeepDataAspectRatio() / setKeepDataAspectRatio(flag=True)   ???????????
#  - resetZoom() ?????


# TODO: Methods to be implemented
#  - setColormap (see ImageView) DONE
#  - getColormap (see ImageView) DONE
#  - setVolume                   IN PROGRESS (perspective to be implemented)
#  - getVolumeShape (or getVolumeLimits?)
#  - setPerspective (or setSlicePlane? or setXYAxes?)

# TODO:
#  - reimplement autoscale feature in setColormap to use min and max
#    in the complete volume, not just vmin/vmax of the 2D slice

class StackView(qt.QWidget):
    """

    """
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self._imageLegend = '__StackView__image' + str(id(self))
        self._volume = None
        self._autoscaleCmap = True

        self._plot = Plot2D(self)

        self._browser = HorizontalSliderWithBrowser(self)
        self._browser.valueChanged[int].connect(self.__updateFrameNumber)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self._plot)
        layout.addWidget(self._browser)

    def __updateFrameNumber(self, index):
        self._plot.addImage(self._volume[index, :, :],     # TODO: depends on perspective
                            legend=self._imageLegend)

    # public API
    def setVolume(self, volume, origin=(0, 0), scale=(1., 1.),
                  copy=True, reset=True):
        """Set the stack of images to display.

        :param volume: A 3D array representing the image or None to empty plot.
        :type volume: numpy.ndarray-like with 3 dimensions or None.
        :param origin: The (x, y) position of the origin of the image.
                       Default: (0, 0).
                       The origin is the lower left corner of the image when
                       the Y axis is not inverted.
        :type origin: Tuple of 2 floats: (origin x, origin y).
        :param scale: The scale factor to apply to the image on X and Y axes.
                      Default: (1, 1).
                      It is the size of a pixel in the coordinates of the axes.
                      Scales must be positive numbers.
        :type scale: Tuple of 2 floats: (scale x, scale y).
        :param bool copy: Whether to copy volume data (default) or not.
        :param bool reset: Whether to reset zoom or not.
        """
        assert len(origin) == 2
        assert len(scale) == 2
        assert scale[0] > 0
        assert scale[1] > 0

        if volume is None:
            self._plot.remove(self._imageLegend, kind='image')
            self._volume = None
            self._browser.setEnabled(False)
            return

        data = numpy.array(volume, order='C', copy=copy)
        assert data.size != 0
        assert len(data.shape) == 3, "data must be 3D"

        self._volume = data

        # This call to setColormap takes redefines the meaning of autoscale
        # for 3D volume: take global min/max rather than frame min/max
        if self._autoscaleCmap:
            self.setColormap(autoscale=True)

        # init plot
        self._plot.addImage(data[0, :, :],         # TODO: depends on perspective
                            legend=self._imageLegend,
                            origin=origin, scale=scale,
                            colormap=self.getColormap(),
                            replace=False)
        self._plot.setActiveImage(self._imageLegend)

        if reset:
            self._plot.resetZoom()

        # enable and init browser
        nframes, height, width = data.shape  # TODO: depends on perspective
        self._browser.setRange(0, nframes - 1)
        self._browser.setEnabled(True)

    def getColormap(self):
        """Get the current colormap description.

        :return: A description of the current colormap.
                 See :meth:`setColormap` for details.
        :rtype: dict
        """
        # default colormap used by addImage
        return self._plot.getDefaultColormap()

    def setColormap(self, colormap=None, normalization=None,
                    autoscale=None, vmin=None, vmax=None, colors=None):
        """Set the colormap and update active image.

        Parameters that are not provided are taken from the current colormap.

        The colormap parameter can also be a dict with the following keys:

        - *name*: string. The colormap to use:
          'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
        - *normalization*: string. The mapping to use for the colormap:
          either 'linear' or 'log'.
        - *autoscale*: bool. Whether to use autoscale (True) or range
          provided by keys
          'vmin' and 'vmax' (False).
        - *vmin*: float. The minimum value of the range to use if 'autoscale'
          is False.
        - *vmax*: float. The maximum value of the range to use if 'autoscale'
          is False.
        - *colors*: optional. Nx3 or Nx4 array of float in [0, 1] or uint8.
                    List of RGB or RGBA colors to use (only if name is None)

        :param colormap: Name of the colormap in
            'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
            Or the description of the colormap as a dict.
        :type colormap: dict or str.
        :param str normalization: Colormap mapping: 'linear' or 'log'.
        :param bool autoscale: Whether to use autoscale (True)
                               or [vmin, vmax] range (False).
        :param float vmin: The minimum value of the range to use if
                           'autoscale' is False.
        :param float vmax: The maximum value of the range to use if
                           'autoscale' is False.
        :param numpy.ndarray colors: Only used if name is None.
            Custom colormap colors as Nx3 or Nx4 RGB or RGBA arrays
        """
        cmapDict = self.getColormap()

        if isinstance(colormap, dict):
            # Support colormap parameter as a dict
            errmsg = "If colormap is provided as a dict, all other parameters"
            errmsg += " must not be specified when calling setColormap"
            assert normalization is None, errmsg
            assert autoscale is None, errmsg
            assert vmin is None, errmsg
            assert vmax is None, errmsg
            assert colors is None, errmsg
            for key, value in colormap.items():
                cmapDict[key] = value

        else:
            if colormap is not None:
                cmapDict['name'] = colormap
            if normalization is not None:
                cmapDict['normalization'] = normalization
            if colors is not None:
                cmapDict['colors'] = colors

            # Default meaning of autoscale is to reset min and max
            # each time a new image is added to the plot.
            # Here, we want to use min and max of global volume,
            # and not change them when browsing slides
            cmapDict['autoscale'] = False         # autoscale in the addImage meaning
            self._autoscaleCmap = autoscale       # our autoscale attribute
            if autoscale and (self._volume is not None):
                cmapDict['vmin'] = self._volume.min()
                cmapDict['vmax'] = self._volume.max()
            else:
                if vmin is not None:
                    cmapDict['vmin'] = vmin
                if vmax is not None:
                    cmapDict['vmax'] = vmax

        cursorColor = cursorColorForColormap(cmapDict['name'])
        self._plot.setInteractiveMode('zoom', color=cursorColor)

        self._plot.setDefaultColormap(cmapDict)

        # Refresh image with new colormap
        activeImage = self._plot.getActiveImage()
        if activeImage is not None:
            data, legend, info, _pixmap = activeImage[0:4]

            self._plot.addImage(data, legend=legend, info=info,
                                colormap=self.getColormap(),
                                replace=False)

# fixme: move demo to silx/examples when complete
if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv[1:])

    mycube = numpy.fromfunction(
        lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2*numpy.sin(k/6.),
        (256, 256, 256)
    )

    sv = StackView()
    sv.setColormap("jet", autoscale=True)
    sv.setVolume(mycube)

    sv.show()

    app.exec_()
