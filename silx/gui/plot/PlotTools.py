# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Set of widgets to associate with a :class:'PlotWidget'.
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "17/10/2016"


import logging
import numbers
import traceback
import weakref

import numpy

from .. import icons
from .. import qt
from .Colors import cursorColorForColormap
from .PlotActions import PlotAction
from .Profile import _alignedFullProfile, _alignedPartialProfile, createProfile

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


# PositionInfo ################################################################

class PositionInfo(qt.QWidget):
    """QWidget displaying coords converted from data coords of the mouse.

    Provide this widget with a list of couple:

    - A name to display before the data
    - A function that takes (x, y) as arguments and returns something that
      gets converted to a string.
      If the result is a float it is converted with '%.7g' format.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow and add a QToolBar where to place the
    PositionInfo widget.

    >>> from silx.gui.plot import PlotWindow
    >>> from silx.gui import qt

    >>> plot = PlotWindow()  # Create a PlotWindow to add the widget to
    >>> toolBar = qt.QToolBar()  # Create a toolbar to place the widget in
    >>> plot.addToolBar(qt.Qt.BottomToolBarArea, toolBar)  # Add it to plot

    Then, create the PositionInfo widget and add it to the toolbar.
    The PositionInfo widget is created with a list of converters, here
    to display polar coordinates of the mouse position.

    >>> import numpy
    >>> from silx.gui.plot.PlotTools import PositionInfo

    >>> position = PositionInfo(plot=plot, converters=[
    ...     ('Radius', lambda x, y: numpy.sqrt(x*x + y*y)),
    ...     ('Angle', lambda x, y: numpy.degrees(numpy.arctan2(y, x)))])

    >>> toolBar.addWidget(position)  # Add the widget to the toolbar
    <...>

    >>> plot.show()  # To display the PlotWindow with the position widget

    :param plot: The PlotWidget this widget is displaying data coords from.
    :param converters: List of name to display and conversion function from
                       (x, y) in data coords to displayed value.
                       If None, the default, it displays X and Y.
    :type converters: Iterable of 2-tuple (str, function)
    :param parent: Parent widget
    """

    def __init__(self, parent=None, plot=None, converters=None):
        assert plot is not None
        self._plotRef = weakref.ref(plot)

        super(PositionInfo, self).__init__(parent)

        if converters is None:
            converters = (('X', lambda x, y: x), ('Y', lambda x, y: y))

        self.autoSnapToActiveCurve = False
        """Toggle snapping use position to active curve.

        - True to snap used coordinates to the active curve if the active curve
          is displayed with symbols and mouse is close enough.
          If the mouse is not close to a point of the curve, values are
          displayed in red.
        - False (the default) to always use mouse coordinates.

        """

        self._fields = []  # To store (QLineEdit, name, function (x, y)->v)

        # Create a new layout with new widgets
        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)

        # Create all QLabel and store them with the corresponding converter
        for name, func in converters:
            layout.addWidget(qt.QLabel('<b>' + name + ':</b>'))

            contentWidget = qt.QLabel()
            contentWidget.setText('------')
            contentWidget.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
            contentWidget.setFixedWidth(
                contentWidget.fontMetrics().width('##############'))
            layout.addWidget(contentWidget)
            self._fields.append((contentWidget, name, func))

        layout.addStretch(1)
        self.setLayout(layout)

        # Connect to Plot events
        plot.sigPlotSignal.connect(self._plotEvent)

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        return self._plotRef()

    def getConverters(self):
        """Return the list of converters as 2-tuple (name, function)."""
        return [(name, func) for _label, name, func in self._fields]

    def _plotEvent(self, event):
        """Handle events from the Plot.

        :param dict event: Plot event
        """
        if event['event'] == 'mouseMoved':
            x, y = event['x'], event['y']  # Position in data
            styleSheet = "color: rgb(0, 0, 0);"  # Default style

            if self.autoSnapToActiveCurve and self.plot.getGraphCursor():
                # Check if near active curve with symbols.

                styleSheet = "color: rgb(255, 0, 0);"  # Style far from curve

                activeCurve = self.plot.getActiveCurve()
                if activeCurve:
                    xData, yData, _legend, _info, params = activeCurve[0:5]
                    if params['symbol']:  # Only handled if symbols on curve
                        closestIndex = numpy.argmin(
                            pow(xData - x, 2) + pow(yData - y, 2))

                        xClosest = xData[closestIndex]
                        yClosest = yData[closestIndex]

                        closestInPixels = self.plot.dataToPixel(
                            xClosest, yClosest, axis=params['yaxis'])
                        if closestInPixels is not None:
                            xClosest, yClosest = closestInPixels
                            xPixel, yPixel = event['xpixel'], event['ypixel']

                            if (abs(xClosest - xPixel) < 5 and
                                    abs(yClosest - yPixel) < 5):
                                # Update label style sheet
                                styleSheet = "color: rgb(0, 0, 0);"

                                # if close enough, wrap to data point coords
                                x, y = xClosest, yClosest

            for label, name, func in self._fields:
                label.setStyleSheet(styleSheet)

                try:
                    value = func(x, y)
                except:
                    label.setText('Error')
                    _logger.error(
                        "Error while converting coordinates (%f, %f)"
                        "with converter '%s'" % (x, y, name))
                    _logger.error(traceback.format_exc())
                else:
                    if isinstance(value, numbers.Real):
                        value = '%.7g' % value  # Use this for floats and int
                    else:
                        value = str(value)  # Fallback for other types
                    label.setText(value)


# LimitsToolBar ##############################################################

class LimitsToolBar(qt.QToolBar):
    """QToolBar displaying and controlling the limits of a :class:`PlotWidget`.

    :param parent: See :class:`QToolBar`.
    :param plot: :class:`PlotWidget` instance on which to operate.
    :param str title: See :class:`QToolBar`.
    """

    class _FloatEdit(qt.QLineEdit):
        """Field to edit a float value."""
        def __init__(self, value=None, *args, **kwargs):
            qt.QLineEdit.__init__(self, *args, **kwargs)
            self.setValidator(qt.QDoubleValidator())
            self.setFixedWidth(100)
            self.setAlignment(qt.Qt.AlignLeft)
            if value is not None:
                self.setValue(value)

        def value(self):
            return float(self.text())

        def setValue(self, value):
            self.setText('%g' % value)

    def __init__(self, parent=None, plot=None, title='Limits'):
        super(LimitsToolBar, self).__init__(title, parent)
        assert plot is not None
        self._plot = plot
        self._plot.sigPlotSignal.connect(self._plotWidgetSlot)

        self._initWidgets()

    @property
    def plot(self):
        """The :class:`PlotWidget` the toolbar is attached to."""
        return self._plot

    def _initWidgets(self):
        """Create and init Toolbar widgets."""
        xMin, xMax = self.plot.getGraphXLimits()
        yMin, yMax = self.plot.getGraphYLimits()

        self.addWidget(qt.QLabel('Limits: '))
        self.addWidget(qt.QLabel(' X: '))
        self._xMinFloatEdit = self._FloatEdit(xMin)
        self._xMinFloatEdit.editingFinished[()].connect(
            self._xFloatEditChanged)
        self.addWidget(self._xMinFloatEdit)

        self._xMaxFloatEdit = self._FloatEdit(xMax)
        self._xMaxFloatEdit.editingFinished[()].connect(
            self._xFloatEditChanged)
        self.addWidget(self._xMaxFloatEdit)

        self.addWidget(qt.QLabel(' Y: '))
        self._yMinFloatEdit = self._FloatEdit(yMin)
        self._yMinFloatEdit.editingFinished[()].connect(
            self._yFloatEditChanged)
        self.addWidget(self._yMinFloatEdit)

        self._yMaxFloatEdit = self._FloatEdit(yMax)
        self._yMaxFloatEdit.editingFinished[()].connect(
            self._yFloatEditChanged)
        self.addWidget(self._yMaxFloatEdit)

    def _plotWidgetSlot(self, event):
        """Listen to :class:`PlotWidget` events."""
        if event['event'] not in ('limitsChanged',):
            return

        xMin, xMax = self.plot.getGraphXLimits()
        yMin, yMax = self.plot.getGraphYLimits()

        self._xMinFloatEdit.setValue(xMin)
        self._xMaxFloatEdit.setValue(xMax)
        self._yMinFloatEdit.setValue(yMin)
        self._yMaxFloatEdit.setValue(yMax)

    def _xFloatEditChanged(self):
        """Handle X limits changed from the GUI."""
        xMin, xMax = self._xMinFloatEdit.value(), self._xMaxFloatEdit.value()
        if xMax < xMin:
            xMin, xMax = xMax, xMin

        self.plot.setGraphXLimits(xMin, xMax)

    def _yFloatEditChanged(self):
        """Handle Y limits changed from the GUI."""
        yMin, yMax = self._yMinFloatEdit.value(), self._yMaxFloatEdit.value()
        if yMax < yMin:
            yMin, yMax = yMax, yMin

        self.plot.setGraphYLimits(yMin, yMax)


# ProfileToolBar ##############################################################

class ProfileToolBar(qt.QToolBar):
    """QToolBar providing profile tools operating on a :class:`PlotWindow`.

    Attributes:

    - plot: Associated :class:`PlotWindow`.
    - profileWindow: Associated :class:`PlotWindow` displaying the profile.
    - actionGroup: :class:`QActionGroup` of available actions.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow and add a :class:`ProfileToolBar`.

    >>> from silx.gui.plot import PlotWindow
    >>> from silx.gui.plot.PlotTools import ProfileToolBar
    >>> from silx.gui import qt

    >>> plot = PlotWindow()  # Create a PlotWindow
    >>> toolBar = ProfileToolBar(plot=plot)  # Create a profile toolbar
    >>> plot.addToolBar(toolBar)  # Add it to plot
    >>> plot.show()  # To display the PlotWindow with the profile toolbar

    :param plot: :class:`PlotWindow` instance on which to operate.
    :param profileWindow: :class:`ProfileScanWidget` instance where to
                          display the profile curve or None to create one.
    :param str title: See :class:`QToolBar`.
    :param parent: See :class:`QToolBar`.
    """
    # TODO Make it a QActionGroup instead of a QToolBar

    _POLYGON_LEGEND = '__ProfileToolBar_ROI_Polygon'

    def __init__(self, parent=None, plot=None, profileWindow=None,
                 title='Profile Selection'):
        super(ProfileToolBar, self).__init__(title, parent)
        assert plot is not None
        self.plot = plot

        self._overlayColor = None
        self._defaultOverlayColor = 'red'  # update when active image change

        self._roiInfo = None  # Store start and end points and type of ROI

        if profileWindow is None:
            # Import here to avoid cyclic import
            from .PlotWindow import Plot1D      # noqa
            self.profileWindow = Plot1D()
            self._ownProfileWindow = True
        else:
            self.profileWindow = profileWindow
            self._ownProfileWindow = False

        # Actions
        self.browseAction = qt.QAction(
            icons.getQIcon('normal'),
            'Browsing Mode', None)
        self.browseAction.setToolTip(
            'Enables zooming interaction mode')
        self.browseAction.setCheckable(True)
        self.browseAction.triggered[bool].connect(self._browseActionTriggered)

        self.hLineAction = qt.QAction(
            icons.getQIcon('shape-horizontal'),
            'Horizontal Profile Mode', None)
        self.hLineAction.setToolTip(
            'Enables horizontal profile selection mode')
        self.hLineAction.setCheckable(True)
        self.hLineAction.toggled[bool].connect(self._hLineActionToggled)

        self.vLineAction = qt.QAction(
            icons.getQIcon('shape-vertical'),
            'Vertical Profile Mode', None)
        self.vLineAction.setToolTip(
            'Enables vertical profile selection mode')
        self.vLineAction.setCheckable(True)
        self.vLineAction.toggled[bool].connect(self._vLineActionToggled)

        self.lineAction = qt.QAction(
            icons.getQIcon('shape-diagonal'),
            'Free Line Profile Mode', None)
        self.lineAction.setToolTip(
            'Enables line profile selection mode')
        self.lineAction.setCheckable(True)
        self.lineAction.toggled[bool].connect(self._lineActionToggled)

        self.clearAction = qt.QAction(
            icons.getQIcon('image'),
            'Clear Profile', None)
        self.clearAction.setToolTip(
            'Clear the profile Region of interest')
        self.clearAction.setCheckable(False)
        self.clearAction.triggered.connect(self.clearProfile)

        # ActionGroup
        self.actionGroup = qt.QActionGroup(self)
        self.actionGroup.addAction(self.browseAction)
        self.actionGroup.addAction(self.hLineAction)
        self.actionGroup.addAction(self.vLineAction)
        self.actionGroup.addAction(self.lineAction)

        self.browseAction.setChecked(True)

        # Add actions to ToolBar
        self.addAction(self.browseAction)
        self.addAction(self.hLineAction)
        self.addAction(self.vLineAction)
        self.addAction(self.lineAction)
        self.addAction(self.clearAction)

        # Add width spin box to toolbar
        self.addWidget(qt.QLabel('W:'))
        self.lineWidthSpinBox = qt.QSpinBox(self)
        self.lineWidthSpinBox.setRange(0, 1000)
        self.lineWidthSpinBox.setValue(1)
        self.lineWidthSpinBox.valueChanged[int].connect(
            self._lineWidthSpinBoxValueChangedSlot)
        self.addWidget(self.lineWidthSpinBox)

        self.plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)

        # Enable toolbar only if there is an active image
        self.setEnabled(self.plot.getActiveImage(just_legend=True) is not None)
        self.plot.sigActiveImageChanged.connect(
            self._activeImageChanged)

    def _activeImageChanged(self, previous, legend):
        """Handle active image change: toggle enabled toolbar, update curve"""
        self.setEnabled(legend is not None)
        if legend is not None:
            # Update default profile color
            activeImage = self.plot.getActiveImage()
            if activeImage is not None:
                self._defaultOverlayColor = cursorColorForColormap(
                    activeImage[4]['colormap']['name'])

            self.updateProfile()

    def _lineWidthSpinBoxValueChangedSlot(self, value):
        """Listen to ROI width widget to refresh ROI and profile"""
        self.updateProfile()

    def _interactiveModeChanged(self, source):
        """Handle plot interactive mode changed:

        If changed from elsewhere, disable drawing tool
        """
        if source is not self:
            self.browseAction.setChecked(True)

    def _hLineActionToggled(self, checked):
        """Handle horizontal line profile action toggle"""
        if checked:
            self.plot.setInteractiveMode('draw', shape='hline',
                                         color=None, source=self)
            self.plot.sigPlotSignal.connect(self._plotWindowSlot)
        else:
            self.plot.sigPlotSignal.disconnect(self._plotWindowSlot)

    def _vLineActionToggled(self, checked):
        """Handle vertical line profile action toggle"""
        if checked:
            self.plot.setInteractiveMode('draw', shape='vline',
                                         color=None, source=self)
            self.plot.sigPlotSignal.connect(self._plotWindowSlot)
        else:
            self.plot.sigPlotSignal.disconnect(self._plotWindowSlot)

    def _lineActionToggled(self, checked):
        """Handle line profile action toggle"""
        if checked:
            self.plot.setInteractiveMode('draw', shape='line',
                                         color=None, source=self)
            self.plot.sigPlotSignal.connect(self._plotWindowSlot)
        else:
            self.plot.sigPlotSignal.disconnect(self._plotWindowSlot)

    def _browseActionTriggered(self, checked):
        """Handle browse action mode triggered by user."""
        if checked:
            self.plot.setInteractiveMode('zoom', source=self)

    def _plotWindowSlot(self, event):
        """Listen to Plot to handle drawing events to refresh ROI and profile.
        """
        if event['event'] not in ('drawingProgress', 'drawingFinished'):
            return

        checkedAction = self.actionGroup.checkedAction()
        if checkedAction == self.hLineAction:
            lineProjectionMode = 'X'
        elif checkedAction == self.vLineAction:
            lineProjectionMode = 'Y'
        elif checkedAction == self.lineAction:
            lineProjectionMode = 'D'
        else:
            return

        roiStart, roiEnd = event['points'][0], event['points'][1]

        self._roiInfo = roiStart, roiEnd, lineProjectionMode
        self.updateProfile()

    @property
    def overlayColor(self):
        """The color to use for the ROI.

        If set to None (the default), the overlay color is adapted to the
        active image colormap and changes if the active image colormap changes.
        """
        return self._overlayColor or self._defaultOverlayColor

    @overlayColor.setter
    def overlayColor(self, color):
        self._overlayColor = color
        self.updateProfile()

    def clearProfile(self):
        """Remove profile curve and profile area."""
        self._roiInfo = None
        self.updateProfile()

    def updateProfile(self):
        """Update the displayed profile and profile ROI.

        This uses the current active image of the plot and the current ROI.
        """
        imageData = self.plot.getActiveImage()
        if imageData is None:
            return
        
        # Clean previous profile area, and previous curve
        self.plot.remove(self._POLYGON_LEGEND, kind='item')
        self.profileWindow.clear()
        self.profileWindow.setGraphTitle('')
        self.profileWindow.setGraphXLabel('X')
        self.profileWindow.setGraphYLabel('Y')

        self._createProfile(currentData=imageData[0], params=imageData[4])

    # TODO henri : create a function wich will be a get of this
    def _createProfile(self, currentData, params):
        """Create the profile line for the the given image.

        :param numpy.ndarray currentData: the image or the stack of images
            on which we compute the profile
        :param params: parameters of the plot, such as origin, scale
            and colormap
        """
        assert('colormap' in params and 'z' in params)
        if self._roiInfo is None:
            return

        profile, area, profileName, xLabel = createProfile( roiInfo=self._roiInfo, 
                                                            currentData=currentData, 
                                                            params=params,
                                                            lineWidth=self.lineWidthSpinBox.value())
        colorMap = params['colormap']

        dataIs3D = len(currentData.shape) > 2
        if dataIs3D:
            self.profileWindow.addImage(profile,
                                        legend=profileName,
                                        xlabel=xLabel,
                                        colormap=colorMap)
        else:
            coords = numpy.arange(len(profile[0]), dtype=numpy.float32)
            self.profileWindow.addCurve(coords, profile[0],
                                        legend=profileName,
                                        xlabel=xLabel,
                                        color=self.overlayColor)

        self.plot.addItem(area[0], area[1],
                          legend=self._POLYGON_LEGEND,
                          color=self.overlayColor,
                          shape='polygon', fill=True,
                          replace=False, z=params['z'] + 1)

        self._showProfileWindow()

    def _showProfileWindow(self):
        """If profile window was created in this widget,
        it tries to avoid overlapping this widget when shown"""
        if self._ownProfileWindow and not self.profileWindow.isVisible():
            winGeom = self.window().frameGeometry()
            qapp = qt.QApplication.instance()
            screenGeom = qapp.desktop().availableGeometry(self)

            spaceOnLeftSide = winGeom.left()
            spaceOnRightSide = screenGeom.width() - winGeom.right()

            profileWindowWidth = self.profileWindow.frameGeometry().width()
            if (profileWindowWidth < spaceOnRightSide or
                    spaceOnRightSide > spaceOnLeftSide):
                # Place profile on the right
                self.profileWindow.move(winGeom.right(), winGeom.top())
            else:
                # Not enough place on the right, place profile on the left
                self.profileWindow.move(
                    max(0, winGeom.left() - profileWindowWidth), winGeom.top())

        self.profileWindow.show()


class Profile3DAction(PlotAction):
    """PlotAction that emits a signal when checked, to notify

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
                tooltip='If activated, compute the profile on the stack of images',
                triggered=self.__compute3DProfile,
                checkable=True,
                parent=parent)

    def __compute3DProfile(self):
        """Callback when the QAction is activated
        """
        self.sigChange3DProfile.emit(self.isChecked())


class Profile3DToolBar(ProfileToolBar):
    def __init__(self, parent=None, plot=None, profileWindow=None,
                 title='Profile Selection'):
        """QToolBar providing profile tools for an image or a stack of images.

        :param parent: the parent QWidget
        :param plot: :class:`PlotWindow` instance on which to operate.
        :param profileWindow: :class:`ProfileScanWidget` instance where to
                              display the profile curve or None to create one.
        :param str title: See :class:`QToolBar`.
        :param parent: See :class:`QToolBar`.
        """
        super(Profile3DToolBar, self).__init__(parent, plot, profileWindow, title)
        if profileWindow is None:
            self._profileWindow1D = self.profileWindow
            from .PlotWindow import Plot2D      # noqa
            self._profileWindow2D = Plot2D()
        self.__create3DProfileAction()
        self._setComputeIn3D(False)

    def __create3DProfileAction(self):
        """Initialize the Profile3DAction action
        """
        self.profile3d = Profile3DAction(plot=self.plot, parent=self.plot)
        self.profile3d.sigChange3DProfile.connect(self._setComputeIn3D)
        self.addAction(self.profile3d)

    def _setComputeIn3D(self, flag):
        """Set flag to *True* to compute the profile in 3D, else
        the profile is computed in 2D on the active image.

        :param bool flag: Flag used when toggling 2D/3D profile mode
        """
        self._computeIn3D = flag
        self._usePlot2DProfile(flag)
        self.updateProfile()

    def _usePlot2DProfile(self, flag):
        """When 3D action is toggled, switch to 3D mode:
        use a Plot2D to display the profile.

        :param bool flag: Flag used when toggling 2D/3D profile mode
        """
        if not self._ownProfileWindow:
            # profile window handled by user
            return

        profileIsVisible = self.profileWindow.isVisible()
        if flag:
            self.profileWindow = self._profileWindow2D
            if profileIsVisible:
                self._profileWindow1D.hide()
                self._profileWindow2D.show()
        else:
            self.profileWindow = self._profileWindow1D
            if profileIsVisible:
                self._profileWindow2D.hide()
                self._profileWindow1D.show()

    def updateProfile(self):
        """Method overloaded from :class:`ProfileToolBar`,
        to pass the stack of images instead of just the active image.

        In 2D profile mode, use the regular parent method.
        """
        if not self._computeIn3D:
            super(Profile3DToolBar, self).updateProfile()
        else:
            stackData = self.plot.getStack(copy=False,
                                           returnNumpyArray=True)
            self.plot.remove(self._POLYGON_LEGEND, kind='item')
            self.profileWindow.clear()
            self.profileWindow.setGraphTitle('')
            self.profileWindow.setGraphXLabel('X')
            self.profileWindow.setGraphYLabel('Y')

            self._createProfile(currentData=stackData[0],
                                params=stackData[1])
