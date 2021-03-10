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
"""
This module contains an :class:`ImageFileDialog`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "05/03/2019"

import logging
from silx.gui.plot import actions
from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget
from .AbstractDataFileDialog import AbstractDataFileDialog
import silx.io


_logger = logging.getLogger(__name__)


class _ImageSelection(qt.QWidget):
    """Provide a widget allowing to select an image from an hypercube by
    selecting a slice."""

    selectionChanged = qt.Signal()
    """Emitted when the selection change."""

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.__shape = None
        self.__axis = []
        layout = qt.QVBoxLayout()
        self.setLayout(layout)

    def hasVisibleSelectors(self):
        return self.__visibleSliders > 0

    def isUsed(self):
        if self.__shape is None:
            return False
        return len(self.__shape) > 2

    def getSelectedData(self, data):
        slicing = self.slicing()
        image = data[slicing]
        return image

    def setData(self, data):
        if data is None:
            self.__visibleSliders = 0
            return

        shape = data.shape
        if self.__shape is not None:
            # clean up
            for widget in self.__axis:
                self.layout().removeWidget(widget)
                widget.deleteLater()
            self.__axis = []

        self.__shape = shape
        self.__visibleSliders = 0

        if shape is not None:
            # create expected axes
            for index in range(len(shape) - 2):
                axis = qt.QSlider(self)
                axis.setMinimum(0)
                axis.setMaximum(shape[index] - 1)
                axis.setOrientation(qt.Qt.Horizontal)
                if shape[index] == 1:
                    axis.setVisible(False)
                else:
                    self.__visibleSliders += 1

                axis.valueChanged.connect(self.__axisValueChanged)
                self.layout().addWidget(axis)
                self.__axis.append(axis)

        self.selectionChanged.emit()

    def __axisValueChanged(self):
        self.selectionChanged.emit()

    def slicing(self):
        slicing = []
        for axes in self.__axis:
            slicing.append(axes.value())
        return tuple(slicing)

    def setSlicing(self, slicing):
        for i, value in enumerate(slicing):
            if i > len(self.__axis):
                break
            self.__axis[i].setValue(value)

    def selectSlicing(self, slicing):
        """Select a slicing.

        The provided value could be unconsistent and therefore is not supposed
        to be retrivable with a getter.

        :param Union[None,Tuple[int]] slicing:
        """
        if slicing is None:
            # Create a default slicing
            needed = self.__visibleSliders
            slicing = (0,) * needed
        if len(slicing) < self.__visibleSliders:
            slicing = slicing + (0,) * (self.__visibleSliders - len(slicing))
        self.setSlicing(slicing)


class _ImagePreview(qt.QWidget):
    """Provide a preview of the selected image"""

    def __init__(self, parent=None):
        super(_ImagePreview, self).__init__(parent)

        self.__data = None
        self.__plot = PlotWidget(self)
        self.__plot.setAxesDisplayed(False)
        self.__plot.setKeepDataAspectRatio(True)
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__plot)
        self.setLayout(layout)

    def resizeEvent(self, event):
        self.__updateConstraints()
        return qt.QWidget.resizeEvent(self, event)

    def sizeHint(self):
        return qt.QSize(200, 200)

    def plot(self):
        return self.__plot

    def setData(self, data, fromDataSelector=False):
        if data is None:
            self.clear()
            return

        resetzoom = not fromDataSelector
        previousImage = self.data()
        if previousImage is not None and data.shape != previousImage.shape:
            resetzoom = True

        self.__plot.addImage(legend="data", data=data, resetzoom=resetzoom)
        self.__data = data
        self.__updateConstraints()

    def __updateConstraints(self):
        """
        Update the constraints depending on the size of the widget
        """
        image = self.data()
        if image is None:
            return
        size = self.size()
        if size.width() == 0 or size.height() == 0:
            return

        heightData, widthData = image.shape

        widthContraint = heightData * size.width() / size.height()
        if widthContraint > widthData:
            heightContraint = heightData
        else:
            heightContraint = heightData * size.height() / size.width()
            widthContraint = widthData

        midWidth, midHeight = widthData * 0.5, heightData * 0.5
        heightContraint, widthContraint = heightContraint * 0.5, widthContraint * 0.5

        axis = self.__plot.getXAxis()
        axis.setLimitsConstraints(midWidth - widthContraint, midWidth + widthContraint)
        axis = self.__plot.getYAxis()
        axis.setLimitsConstraints(midHeight - heightContraint, midHeight + heightContraint)

    def __imageItem(self):
        image = self.__plot.getImage("data")
        return image

    def data(self):
        if self.__data is not None:
            if hasattr(self.__data, "name"):
                # in case of HDF5
                if self.__data.name is None:
                    # The dataset was closed
                    self.__data = None
        return self.__data

    def colormap(self):
        image = self.__imageItem()
        if image is not None:
            return image.getColormap()
        return self.__plot.getDefaultColormap()

    def setColormap(self, colormap):
        self.__plot.setDefaultColormap(colormap)

    def clear(self):
        self.__data = None
        image = self.__imageItem()
        if image is not None:
            self.__plot.removeImage(legend="data")


class ImageFileDialog(AbstractDataFileDialog):
    """The `ImageFileDialog` class provides a dialog that allow users to select
    an image from a file.

    The `ImageFileDialog` class enables a user to traverse the file system in
    order to select one file. Then to traverse the file to select a frame or
    a slice of a dataset.

    .. image:: img/imagefiledialog_h5.png

    It supports fast access to image files using `FabIO`. Which is not the case
    of the default silx API. Image files still also can be available using the
    NeXus layout, by editing the file type combo box.

    .. image:: img/imagefiledialog_edf.png

    The selected data is an numpy array with 2 dimension.

    Using an `ImageFileDialog` can be done like that.

    .. code-block:: python

        dialog = ImageFileDialog()
        result = dialog.exec_()
        if result:
            print("Selection:")
            print(dialog.selectedFile())
            print(dialog.selectedUrl())
            print(dialog.selectedImage())
        else:
            print("Nothing selected")
    """

    def selectedImage(self):
        """Returns the selected image data as numpy

        :rtype: numpy.ndarray
        """
        url = self.selectedUrl()
        return silx.io.get_data(url)

    def _createPreviewWidget(self, parent):
        previewWidget = _ImagePreview(parent)
        previewWidget.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        return previewWidget

    def _createSelectorWidget(self, parent):
        return _ImageSelection(parent)

    def _createPreviewToolbar(self, parent, dataPreviewWidget, dataSelectorWidget):
        plot = dataPreviewWidget.plot()
        toolbar = qt.QToolBar(parent)
        toolbar.setIconSize(qt.QSize(16, 16))
        toolbar.setStyleSheet("QToolBar { border: 0px }")
        toolbar.addAction(actions.mode.ZoomModeAction(plot, parent))
        toolbar.addAction(actions.mode.PanModeAction(plot, parent))
        toolbar.addSeparator()
        toolbar.addAction(actions.control.ResetZoomAction(plot, parent))
        toolbar.addSeparator()
        toolbar.addAction(actions.control.ColormapAction(plot, parent))
        return toolbar

    def _isDataSupportable(self, data):
        """Check if the selected data can be supported at one point.

        If true, the data selector will be checked and it will update the data
        preview. Else the selecting is disabled.

        :rtype: bool
        """
        if not hasattr(data, "dtype"):
            # It is not an HDF5 dataset nor a fabio image wrapper
            return False

        if data is None or data.shape is None:
            return False

        if data.dtype.kind not in set(["f", "u", "i", "b"]):
            return False

        dim = len(data.shape)
        return dim >= 2

    def _isFabioFilesSupported(self):
        return True

    def _isDataSupported(self, data):
        """Check if the data can be returned by the dialog.

        If true, this data can be returned by the dialog and the open button
        while be enabled. If false the button will be disabled.

        :rtype: bool
        """
        dim = len(data.shape)
        return dim == 2

    def _displayedDataInfo(self, dataBeforeSelection, dataAfterSelection):
        """Returns the text displayed under the data preview.

        This zone is used to display error in case or problem of data selection
        or problems with IO.

        :param numpy.ndarray dataAfterSelection: Data as it is after the
            selection widget (basically the data from the preview widget)
        :param numpy.ndarray dataAfterSelection: Data as it is before the
            selection widget (basically the data from the browsing widget)
        :rtype: bool
        """
        destination = self.__formatShape(dataAfterSelection.shape)
        source = self.__formatShape(dataBeforeSelection.shape)
        return u"%s \u2192 %s" % (source, destination)

    def __formatShape(self, shape):
        result = []
        for s in shape:
            if isinstance(s, slice):
                v = u"\u2026"
            else:
                v = str(s)
            result.append(v)
        return u" \u00D7 ".join(result)
