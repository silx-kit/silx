# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module provides tool bar helper.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/06/2023"


import logging
import weakref
import numpy

from silx.gui import qt


_logger = logging.getLogger(__name__)


class CompareImagesStatusBar(qt.QStatusBar):
    """StatusBar containing specific information contained in a
    :class:`CompareImages` widget

    Use :meth:`setCompareWidget` to connect this toolbar to a specific
    :class:`CompareImages` widget.

    :param Union[qt.QWidget,None] parent: Parent of this widget.
    """

    def __init__(self, parent=None):
        qt.QStatusBar.__init__(self, parent)
        self.setSizeGripEnabled(False)
        self.layout().setSpacing(0)
        self.__compareWidget = None
        self._label1 = qt.QLabel(self)
        self._label1.setFrameShape(qt.QFrame.WinPanel)
        self._label1.setFrameShadow(qt.QFrame.Sunken)
        self._label2 = qt.QLabel(self)
        self._label2.setFrameShape(qt.QFrame.WinPanel)
        self._label2.setFrameShadow(qt.QFrame.Sunken)
        self._transform = qt.QLabel(self)
        self._transform.setFrameShape(qt.QFrame.WinPanel)
        self._transform.setFrameShadow(qt.QFrame.Sunken)
        self.addWidget(self._label1)
        self.addWidget(self._label2)
        self.addWidget(self._transform)
        self._pos = None
        self._updateStatusBar()

    def setCompareWidget(self, widget):
        """
        Connect this tool bar to a specific :class:`CompareImages` widget.

        :param Union[None,CompareImages] widget: The widget to connect with.
        """
        compareWidget = self.getCompareWidget()
        if compareWidget is not None:
            compareWidget.getPlot().sigPlotSignal.disconnect(self.__plotSignalReceived)
            compareWidget.sigConfigurationChanged.disconnect(self.__dataChanged)
        compareWidget = widget
        if compareWidget is None:
            self.__compareWidget = None
        else:
            self.__compareWidget = weakref.ref(compareWidget)
        if compareWidget is not None:
            compareWidget.getPlot().sigPlotSignal.connect(self.__plotSignalReceived)
            compareWidget.sigConfigurationChanged.connect(self.__dataChanged)

    def getCompareWidget(self):
        """Returns the connected widget.

        :rtype: CompareImages
        """
        if self.__compareWidget is None:
            return None
        else:
            return self.__compareWidget()

    def __plotSignalReceived(self, event):
        """Called when old style signals at emmited from the plot."""
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            self.__mouseMoved(x, y)

    def __mouseMoved(self, x, y):
        """Called when mouse move over the plot."""
        self._pos = x, y
        self._updateStatusBar()

    def __dataChanged(self):
        """Called when internal data from the connected widget changes."""
        self._updateStatusBar()

    def _formatData(self, data):
        """Format pixel of an image.

        It supports intensity, RGB, and RGBA.

        :param Union[int,float,numpy.ndarray,str]: Value of a pixel
        :rtype: str
        """
        if data is None:
            return "No data"
        if isinstance(data, (int, numpy.integer)):
            return "%d" % data
        if isinstance(data, (float, numpy.floating)):
            return "%f" % data
        if isinstance(data, numpy.ndarray):
            # RGBA value
            if data.shape == (3,):
                return "R:%d G:%d B:%d" % (data[0], data[1], data[2])
            elif data.shape == (4,):
                return "R:%d G:%d B:%d A:%d" % (data[0], data[1], data[2], data[3])
        _logger.debug("Unsupported data format %s. Cast it to string.", type(data))
        return str(data)

    def _updateStatusBar(self):
        """Update the content of the status bar"""
        widget = self.getCompareWidget()
        if widget is None:
            self._label1.setText("ImageA: NA")
            self._label2.setText("ImageB: NA")
            self._transform.setVisible(False)
        else:
            transform = widget.getTransformation()
            self._transform.setVisible(transform is not None)
            if transform is not None:
                has_notable_translation = not numpy.isclose(
                    transform.tx, 0.0, atol=0.01
                ) or not numpy.isclose(transform.ty, 0.0, atol=0.01)
                has_notable_scale = not numpy.isclose(
                    transform.sx, 1.0, atol=0.01
                ) or not numpy.isclose(transform.sy, 1.0, atol=0.01)
                has_notable_rotation = not numpy.isclose(transform.rot, 0.0, atol=0.01)

                strings = []
                if has_notable_translation:
                    strings.append("Translation")
                if has_notable_scale:
                    strings.append("Scale")
                if has_notable_rotation:
                    strings.append("Rotation")
                if strings == []:
                    has_translation = not numpy.isclose(
                        transform.tx, 0.0
                    ) or not numpy.isclose(transform.ty, 0.0)
                    has_scale = not numpy.isclose(
                        transform.sx, 1.0
                    ) or not numpy.isclose(transform.sy, 1.0)
                    has_rotation = not numpy.isclose(transform.rot, 0.0)
                    if has_translation or has_scale or has_rotation:
                        text = "No big changes"
                    else:
                        text = "No changes"
                else:
                    text = "+".join(strings)
                self._transform.setText("Align: " + text)

                strings = []
                if not numpy.isclose(transform.ty, 0.0):
                    strings.append("Translation x: %0.3fpx" % transform.tx)
                if not numpy.isclose(transform.ty, 0.0):
                    strings.append("Translation y: %0.3fpx" % transform.ty)
                if not numpy.isclose(transform.sx, 1.0):
                    strings.append("Scale x: %0.3f" % transform.sx)
                if not numpy.isclose(transform.sy, 1.0):
                    strings.append("Scale y: %0.3f" % transform.sy)
                if not numpy.isclose(transform.rot, 0.0):
                    strings.append(
                        "Rotation: %0.3fdeg" % (transform.rot * 180 / numpy.pi)
                    )
                if strings == []:
                    text = "No transformation"
                else:
                    text = "\n".join(strings)
                self._transform.setToolTip(text)

            if self._pos is None:
                self._label1.setText("ImageA: NA")
                self._label2.setText("ImageB: NA")
            else:
                data1, data2 = widget.getRawPixelData(self._pos[0], self._pos[1])
                if isinstance(data1, str):
                    self._label1.setToolTip(data1)
                    text1 = "NA"
                else:
                    self._label1.setToolTip("")
                    text1 = self._formatData(data1)
                if isinstance(data2, str):
                    self._label2.setToolTip(data2)
                    text2 = "NA"
                else:
                    self._label2.setToolTip("")
                    text2 = self._formatData(data2)
                self._label1.setText("ImageA: %s" % text1)
                self._label2.setText("ImageB: %s" % text2)
