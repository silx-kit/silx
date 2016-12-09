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

__authors__ = ["P. Knobel", "H. Payno"]
__license__ = "MIT"
__date__ = "07/12/2016"

import numpy

from silx.gui import qt
from silx.gui.plot import PlotWindow, PlotActions
from silx.gui.plot.Colors import cursorColorForColormap
from silx.gui.widgets.FrameBrowser import HorizontalSliderWithBrowser
from silx.gui.plot.PlotTools import ProfileToolBar


class StackView(qt.QWidget):
    """Simple stack view 

    :param parent: the Qt parent
    """
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self._volume = None
        """Loaded data volume, as a numpy array"""
        self.__volumeview = None
        """View on :attr:`_volume` with the axes sorted, to have
        the orthogonal dimension first"""
        self.__perspective = 0
        """Orthogonal dimension (depth) in :attr:`_volume`"""

        self.__imageLegend = '__StackView__image' + str(id(self))
        self.__autoscaleCmap = True
        """Flag to disable/enable colormap autoscaling
        based on the 3D volume"""

        self._plot = PlotWindow(parent=self, backend=None,
                                resetzoom=True, autoScale=False,
                                logScale=False, grid=False,
                                curveStyle=False, colormap=True,
                                aspectRatio=True, yInverted=True,
                                copy=True, save=True, print_=True,
                                control=False, position=None,
                                roi=False, mask=False)

        self._plot.profile = Profile3DToolBar(parent=self._plot,
                                              plot=self._plot,
                                              volume=self.__volumeview)
        self._plot.addToolBar(self._plot.profile)
        self._plot.setGraphXLabel('Columns')
        self._plot.setGraphYLabel('Rows')

        self._browser = HorizontalSliderWithBrowser(self)
        self._browser.valueChanged[int].connect(self.__updateFrameNumber)
        self._browser.setEnabled(False)

        layout = qt.QVBoxLayout(self)

        planeSelction = DockPlanes(self._plot)
        planeSelction.sigPlaneSelectionChanged.connect(self.__setPerspective)
        self._plot._introduceNewDockWidget(planeSelction)
        layout.addWidget(self._plot)
        layout.addWidget(self._browser)

    def __setPerspective(self, perspective):
        """Function called when the dimension browse changes

        :param persepective: the new dimension browse
        """
        if perspective == self.__perspective:
            return
        else:
            if perspective > 2 or perspective < 0:
                raise ValueError('Can\'t set persepective')

            self.__perspective = perspective
            self.__createVolumeView()
            self.__updateFrameNumber(self._browser.value())
            self._plot.resetZoom()

    def __createVolumeView(self):
        """Create the new view on the volume depending on the perspective
        (set orthogonal axis browsed on the viewer as first dimension)
        """
        assert not self._volume is None
        assert 0 <= self.__perspective < 3
        if self.__perspective == 0:
            self.__volumeview = self._volume.view()
        if self.__perspective == 1:
            self.__volumeview = numpy.rollaxis(self._volume, 1)
        if self.__perspective == 2:
            self.__volumeview = numpy.rollaxis(self._volume, 2)

        self._browser.setRange(0, self.__volumeview.shape[0] - 1)

        self._plot.profile.updateVolume(self.__volumeview)

    def __updateFrameNumber(self, index):
        """Update the current image displayed

        :param index: index of the image to display
        """
        assert self.__volumeview is not None
        self._plot.addImage(self.__volumeview[index, :, :],
                            legend=self.__imageLegend,
                            resetzoom=False)

    # public API
    def setVolume(self, volume, origin=(0, 0), scale=(1., 1.),
                  copy=True, reset=True):
        """Set the stack of images to display.

        :param volume: A 3D array representing the image or None to clear plot.
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
            self.clear()
            return

        data = numpy.array(volume, order='C', copy=copy)
        assert data.size != 0
        assert len(data.shape) == 3, "data must be 3D"

        self._volume = data
        self.__createVolumeView()    

        # This call to setColormap takes redefines the meaning of autoscale
        # for 3D volume: take global min/max rather than frame min/max
        if self.__autoscaleCmap:
            self.setColormap(autoscale=True)

        # init plot
        self._plot.addImage(self.__volumeview[0, :, :],
                            legend=self.__imageLegend,
                            origin=origin, scale=scale,
                            colormap=self.getColormap())

        self._plot.setActiveImage(self.__imageLegend)

        if reset:
            self._plot.resetZoom()

        # enable and init browser
        self._browser.setEnabled(True)

    def clear(self):
        """Clear the widget:

         - clear the plot
         - clear the loaded data volume
        """
        self._volume = None
        self.__volumeview = None
        self.__perspective = 0
        self._browser.setEnabled(False)
        self._plot.clear()

    def resetZoom(self):
        """Reset the plot limits to the bounds of the data and redraw the plot.
        """
        self._plot.resetZoom()

    def getGraphTitle(self):
        """Return the plot main title as a str."""
        return self._plot.getGraphTitle()

    def setGraphTitle(self, title=""):
        """Set the plot main title.

        :param str title: Main title of the plot (default: '')
        """
        return self._plot.setGraphTitle(title)

    def getGraphXLabel(self):
        """Return the current X axis label as a str."""
        return self._plot.getGraphXLabel()

    def setGraphXLabel(self, label="X"):
        """Set the plot X axis label.

        :param str label: The X axis label (default: 'X')
        """
        self._plot.setGraphXLabel(label)

    def getGraphYLabel(self, axis='left'):
        """Return the current Y axis label as a str.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        return self._plot.getGraphYLabel(axis)

    def setGraphYLabel(self, label="Y", axis='left'):
        """Set the plot Y axis label.

        :param str label: The Y axis label (default: 'Y')
        :param str axis: The Y axis for which to set the label (left or right)
        """
        self._plot.setGraphYLabel(label, axis)

    def setYAxisInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        self._plot.setYAxisInverted(flag)

    def isYAxisInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self._backend.isYAxisInverted()

    def getSupportedColormaps(self):
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:
        ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
        """
        return self._plot.getSupportedColormaps()

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
            cmapDict['autoscale'] = False
            self.__autoscaleCmap = autoscale
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
                                resetzoom=False)

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not."""
        return self._plot.isKeepDataAspectRatio()

    def setKeepDataAspectRatio(self, flag=True):
        """Set whether the plot keeps data aspect ratio or not.

        :param bool flag: True to respect data aspect ratio
        """
        self._plot.setKeepDataAspectRatio(flag)


class DockPlanes(qt.QDockWidget):
    """Dock widget for the plane selection

    :param parent: the Qt parent
    """
    sigPlaneSelectionChanged = qt.Signal(int)

    def __init__(self, parent):
        super(DockPlanes, self).__init__(parent)

        planeGB = qt.QGroupBox(self)
        planeGBLayout = qt.QVBoxLayout()
        planeGB.setLayout(planeGBLayout)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Expanding)

        self._qrbDim0Dim1 = qt.QRadioButton('Dim0-Dim1', planeGB)
        self._qrbDim1Dim2 = qt.QRadioButton('Dim1-Dim2', planeGB)
        self._qrbDim0Dim2 = qt.QRadioButton('Dim0-Dim2', planeGB)
        self._qrbDim1Dim2.setChecked(True)

        self._qrbDim1Dim2.toggled.connect(self.__planeSelectionChanged)
        self._qrbDim0Dim1.toggled.connect(self.__planeSelectionChanged)
        self._qrbDim0Dim2.toggled.connect(self.__planeSelectionChanged)

        planeGBLayout.addWidget(self._qrbDim0Dim1)
        planeGBLayout.addWidget(self._qrbDim1Dim2)
        planeGBLayout.addWidget(self._qrbDim0Dim2)
        planeGBLayout.addItem(spacer)
        planeGBLayout.setContentsMargins(0, 0, 0, 0)
        self.setWidget(planeGB)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle('Planes')

        self.setFeatures(qt.QDockWidget.DockWidgetMovable)

    def __planeSelectionChanged(self):
        """Callback function when the radio button change
        """
        self.sigPlaneSelectionChanged.emit(self.getPlaneSelected())

    def getPlaneSelected(self):
        """Return the plane selected following the convention : 
        - dimension : Dim1Dim2 : 0
        - dimension : Dim0Dim2 : 1
        - dimension : Dim0Dim1 : 2
        """
        if self._qrbDim1Dim2.isChecked():
            return 0
        if self._qrbDim0Dim2.isChecked():
            return 1
        if self._qrbDim0Dim1.isChecked():
            return 2

        raise RuntimeError('No plane selected')


class Profile3DToolBar(ProfileToolBar):
    def __init__(self, parent=None, plot=None, profileWindow=None,
                 title='Profile Selection', volume=None):
        """QToolBar providing profile tools for 2D and 3D.

        :param parent: the Qt parent
        :param plot: :class:`PlotWindow` instance on which to operate.
        :param profileWindow: :class:`ProfileScanWidget` instance where to
                              display the profile curve or None to create one.
        :param str title: See :class:`QToolBar`.
        :param parent: See :class:`QToolBar`.
        :param volume: the 3D volume. Always compute the profile across
            the first axis
        """
        super(Profile3DToolBar, self).__init__(parent, plot, profileWindow, title)
        self._volume = volume
        self.profil3DAction = self.__create3DProfilAction(volume)
        self._setComputeIn3D(False)

    def __create3DProfilAction(self, volume):
        """Initialize the Profile3DAction action 
        """
        self.profile3d = Profile3DAction(plot=self.plot, parent=self.plot)
        self.profile3d.sigChange3DProfile.connect(self._setComputeIn3D)
        self.addAction(self.profile3d)

    def updateVolume(self, volume):
        """Update the volume browse

        ..note ::We will always browse through the first dimension

        :param volume: the new volume epxlored
        """
        self._volume = volume

    def _setComputeIn3D(self, b):
        """Set if we want to compute the profile in 2D or in 3D

        :param b:boolean
        """
        self._computeIn3D = b
        self.updateProfile()

    def updateProfile(self):
        """Redefine the one from :class:`ProfileToolBar`
        """
        if self._computeIn3D is False:
            super(Profile3DToolBar, self).updateProfile()
        else:
            self.plot.remove(self._POLYGON_LEGEND, kind='item')
            self.profileWindow.clear()
            self.profileWindow.setGraphTitle('')
            self.profileWindow.setGraphXLabel('X')
            self.profileWindow.setGraphYLabel('Y')

            # TODO : should we add a different color gradation relative to slices ?
            data = self.plot.getActiveImage()
            super(Profile3DToolBar, self)._createProfile(currentData=self._volume[0, :, :], 
                                                         params=self.plot.getActiveImage()[4],
                                                         volume=self._volume)


class Profile3DAction(PlotActions.PlotAction):
    """Base class for QAction that operates on a PlotWidget.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param icon: QIcon or str name of icon to use
    :param str text: The name of this action to be used for menu label
    :param str tooltip: The text of the tooltip
    :param triggered: The callback to connect to the action's triggered
                      signal or None for no callback.
    :param bool checkable: True for checkable action, False otherwise (default)
    :param parent: See :class:`QAction`.
    """
    sigChange3DProfile = qt.Signal(bool)
    def __init__(self, plot, parent=None):
        super(Profile3DAction, self).__init__(
                plot=plot,
                icon='cube',
                text='3D profile',
                tooltip='If activated, will compute the profile on the dimension browsed',
                triggered=self.__compute3DProfile,
                checkable=True,
                parent=parent)

    def __compute3DProfile(self):
        """Callback when the QAction is activated
        """
        self.sigChange3DProfile.emit(self.isChecked())


# fixme: move demo to silx/examples when complete
if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv[1:])

    mycube = numpy.fromfunction(
        lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2*numpy.sin(k/6.),
        (100, 256, 256)
    )

    sv = StackView()
    sv.setColormap("jet", autoscale=True)
    sv.setVolume(mycube)

    sv.show()

    app.exec_()


