# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""Widget providing a set of tools to draw masks on a PlotWidget.

This widget is meant to work with a modified :class:`silx.gui.plot.PlotWidget`

- :class:`ScatterMask`: Handle scatter mask update and history
- :class:`ScatterMaskToolsWidget`: GUI for :class:`ScatterMask`
- :class:`ScatterMaskToolsDockWidget`: DockWidget to integrate in :class:`PlotWindow`
"""

from __future__ import division

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "15/02/2019"


import math
import logging
import os
import numpy
import sys

from .. import qt
from ...math.combo import min_max
from ...image import shapes

from .items import ItemChangedType, Scatter
from ._BaseMaskToolsWidget import BaseMask, BaseMaskToolsWidget, BaseMaskToolsDockWidget
from ..colors import cursorColorForColormap, rgba


_logger = logging.getLogger(__name__)


class ScatterMask(BaseMask):
    """A 1D mask for scatter data.
    """
    def __init__(self, scatter=None):
        """

        :param scatter: :class:`silx.gui.plot.items.Scatter` instance
        """
        BaseMask.__init__(self, scatter)

    def _getXY(self):
        x = self._dataItem.getXData(copy=False)
        y = self._dataItem.getYData(copy=False)
        return x, y

    def getDataValues(self):
        """Return scatter data values as a 1D array.

        :rtype: 1D numpy.ndarray
        """
        return self._dataItem.getValueData(copy=False)

    def save(self, filename, kind):
        if kind == 'npy':
            try:
                numpy.save(filename, self.getMask(copy=False))
            except IOError:
                raise RuntimeError("Mask file can't be written")
        elif kind in ["csv", "txt"]:
            try:
                numpy.savetxt(filename, self.getMask(copy=False))
            except IOError:
                raise RuntimeError("Mask file can't be written")

    def updatePoints(self, level, indices, mask=True):
        """Mask/Unmask points with given indices.

        :param int level: Mask level to update.
        :param indices: Sequence or 1D array of indices of points to be
            updated
        :param bool mask: True to mask (default), False to unmask.
        """
        if mask:
            self._mask[indices] = level
        else:
            # unmask only where mask level is the specified value
            indices_stencil = numpy.zeros_like(self._mask, dtype=numpy.bool)
            indices_stencil[indices] = True
            self._mask[numpy.logical_and(self._mask == level, indices_stencil)] = 0
        self._notify()

    # update shapes
    def updatePolygon(self, level, vertices, mask=True):
        """Mask/Unmask a polygon of the given mask level.

        :param int level: Mask level to update.
        :param vertices: Nx2 array of polygon corners as (y, x) or (row, col)
        :param bool mask: True to mask (default), False to unmask.
        """
        polygon = shapes.Polygon(vertices)
        x, y = self._getXY()

        # TODO: this could be optimized if necessary
        indices_in_polygon = [idx for idx in range(len(x)) if
                              polygon.is_inside(y[idx], x[idx])]

        self.updatePoints(level, indices_in_polygon, mask)

    def updateRectangle(self, level, y, x, height, width, mask=True):
        """Mask/Unmask data inside a rectangle

        :param int level: Mask level to update.
        :param float y: Y coordinate of bottom left corner of the rectangle
        :param float x: X coordinate of bottom left corner of the rectangle
        :param float height:
        :param float width:
        :param bool mask: True to mask (default), False to unmask.
        """
        vertices = [(y, x),
                    (y + height, x),
                    (y + height, x + width),
                    (y, x + width)]
        self.updatePolygon(level, vertices, mask)

    def updateDisk(self, level, cy, cx, radius, mask=True):
        """Mask/Unmask a disk of the given mask level.

        :param int level: Mask level to update.
        :param float cy: Disk center (y).
        :param float cx: Disk center (x).
        :param float radius: Radius of the disk in mask array unit
        :param bool mask: True to mask (default), False to unmask.
        """
        x, y = self._getXY()
        stencil = (y - cy)**2 + (x - cx)**2 < radius**2
        self.updateStencil(level, stencil, mask)

    def updateEllipse(self, level, crow, ccol, radius_r, radius_c, mask=True):
        """Mask/Unmask an ellipse of the given mask level.

        :param int level: Mask level to update.
        :param int crow: Row of the center of the ellipse
        :param int ccol: Column of the center of the ellipse
        :param float radius_r: Radius of the ellipse in the row
        :param float radius_c: Radius of the ellipse in the column
        :param bool mask: True to mask (default), False to unmask.
        """
        def is_inside(px, py):
            return (px - ccol)**2 / radius_c**2 + (py - crow)**2 / radius_r**2 <= 1.0
        x, y = self._getXY()
        indices_inside = [idx for idx in range(len(x)) if is_inside(x[idx], y[idx])]
        self.updatePoints(level, indices_inside, mask)

    def updateLine(self, level, y0, x0, y1, x1, width, mask=True):
        """Mask/Unmask points inside a rectangle defined by a line (two
        end points) and a width.

        :param int level: Mask level to update.
        :param float y0: Row of the starting point.
        :param float x0: Column of the starting point.
        :param float row1: Row of the end point.
        :param float col1: Column of the end point.
        :param float width: Width of the line.
        :param bool mask: True to mask (default), False to unmask.
        """
        # theta is the angle between the horizontal and the line
        theta = math.atan((y1 - y0) / (x1 - x0)) if x1 - x0 else 0
        w_over_2_sin_theta = width / 2. * math.sin(theta)
        w_over_2_cos_theta = width / 2. * math.cos(theta)

        vertices = [(y0 - w_over_2_cos_theta, x0 + w_over_2_sin_theta),
                    (y0 + w_over_2_cos_theta, x0 - w_over_2_sin_theta),
                    (y1 + w_over_2_cos_theta, x1 - w_over_2_sin_theta),
                    (y1 - w_over_2_cos_theta, x1 + w_over_2_sin_theta)]

        self.updatePolygon(level, vertices, mask)


class ScatterMaskToolsWidget(BaseMaskToolsWidget):
    """Widget with tools for masking data points on a scatter in a
    :class:`PlotWidget`."""

    def __init__(self, parent=None, plot=None):
        super(ScatterMaskToolsWidget, self).__init__(parent, plot,
                                                     mask=ScatterMask())
        self._z = 2  # Mask layer in plot
        self._data_scatter = None
        """plot Scatter item for data"""

        self._data_extent = None
        """Maximum extent of the data i.e., max(xMax-xMin, yMax-yMin)"""

        self._mask_scatter = None
        """plot Scatter item for representing the mask"""

    def setSelectionMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask:
            The array to use for the mask or None to reset the mask.
        :type mask: numpy.ndarray of uint8, C-contiguous.
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        :return: None if failed, shape of mask as 1-tuple if successful.
                 The mask can be cropped or padded to fit active scatter,
                 the returned shape is that of the scatter data.
        """
        if self._data_scatter is None:
            # this can happen if the mask tools widget has never been shown
            self._data_scatter = self.plot._getActiveItem(kind="scatter")
            if self._data_scatter is None:
                return None
            self._adjustColorAndBrushSize(self._data_scatter)

        if mask is None:
            self.resetSelectionMask()
            return self._data_scatter.getXData(copy=False).shape

        mask = numpy.array(mask, copy=False, dtype=numpy.uint8)

        if self._data_scatter.getXData(copy=False).shape == (0,) \
                or mask.shape == self._data_scatter.getXData(copy=False).shape:
            self._mask.setMask(mask, copy=copy)
            self._mask.commit()
            return mask.shape
        else:
            raise ValueError("Mask does not have the same shape as the data")

    # Handle mask refresh on the plot

    def _updatePlotMask(self):
        """Update mask image in plot"""
        mask = self.getSelectionMask(copy=False)
        if mask is not None:
            self.plot.addScatter(self._data_scatter.getXData(),
                                 self._data_scatter.getYData(),
                                 mask,
                                 legend=self._maskName,
                                 colormap=self._colormap,
                                 z=self._z)
            self._mask_scatter = self.plot._getItem(kind="scatter",
                                                    legend=self._maskName)
            self._mask_scatter.setSymbolSize(
                self._data_scatter.getSymbolSize() + 2.0)
            self._mask_scatter.sigItemChanged.connect(self.__maskScatterChanged)
        elif self.plot._getItem(kind="scatter",
                                legend=self._maskName) is not None:
            self.plot.remove(self._maskName, kind='scatter')

    def __maskScatterChanged(self, event):
        """Handles update of mask scatter"""
        if (event is ItemChangedType.VISUALIZATION_MODE and
                self._mask_scatter is not None):
            self._mask_scatter.setVisualization(Scatter.Visualization.POINTS)

    # track widget visibility and plot active image changes

    def showEvent(self, event):
        try:
            self.plot.sigActiveScatterChanged.disconnect(
                self._activeScatterChangedAfterCare)
        except (RuntimeError, TypeError):
            pass
        self._activeScatterChanged(None, None)   # Init mask + enable/disable widget
        self.plot.sigActiveScatterChanged.connect(self._activeScatterChanged)

    def hideEvent(self, event):
        try:
            # if the method is not connected this raises a TypeError and there is no way
            # to know the connected slots
            self.plot.sigActiveScatterChanged.disconnect(self._activeScatterChanged)
        except (RuntimeError, TypeError):
            _logger.info(sys.exc_info()[1])
        if not self.browseAction.isChecked():
            self.browseAction.trigger()  # Disable drawing tool

        if self.getSelectionMask(copy=False) is not None:
            self.plot.sigActiveScatterChanged.connect(
                self._activeScatterChangedAfterCare)

    def _adjustColorAndBrushSize(self, activeScatter):
        colormap = activeScatter.getColormap()
        self._defaultOverlayColor = rgba(cursorColorForColormap(colormap['name']))
        self._setMaskColors(self.levelSpinBox.value(),
                            self.transparencySlider.value() /
                            self.transparencySlider.maximum())
        self._z = activeScatter.getZValue() + 1
        self._data_scatter = activeScatter

        # Adjust brush size to data range
        xData = self._data_scatter.getXData(copy=False)
        yData = self._data_scatter.getYData(copy=False)
        # Adjust brush size to data range
        if xData.size > 0 and yData.size > 0:
            xMin, xMax = min_max(xData)
            yMin, yMax = min_max(yData)
            self._data_extent = max(xMax - xMin, yMax - yMin)
        else:
            self._data_extent = None

    def _activeScatterChangedAfterCare(self, previous, next):
        """Check synchro of active scatter and mask when mask widget is hidden.

        If active image has no more the same size as the mask, the mask is
        removed, otherwise it is adjusted to z.
        """
        # check that content changed was the active scatter
        activeScatter = self.plot._getActiveItem(kind="scatter")

        if activeScatter is None or activeScatter.getName() == self._maskName:
            # No active scatter or active scatter is the mask...
            self.plot.sigActiveScatterChanged.disconnect(
                self._activeScatterChangedAfterCare)
            self._data_extent = None
            self._data_scatter = None

        else:
            self._adjustColorAndBrushSize(activeScatter)

            if self._data_scatter.getXData(copy=False).shape != self._mask.getMask(copy=False).shape:
                # scatter has not the same size, remove mask and stop listening
                if self.plot._getItem(kind="scatter", legend=self._maskName):
                    self.plot.remove(self._maskName, kind='scatter')

                self.plot.sigActiveScatterChanged.disconnect(
                    self._activeScatterChangedAfterCare)
                self._data_extent = None
                self._data_scatter = None

            else:
                # Refresh in case z changed
                self._mask.setDataItem(self._data_scatter)
                self._updatePlotMask()

    def _activeScatterChanged(self, previous, next):
        """Update widget and mask according to active scatter changes"""
        activeScatter = self.plot._getActiveItem(kind="scatter")

        if activeScatter is None or activeScatter.getName() == self._maskName:
            # No active scatter or active scatter is the mask...
            self.setEnabled(False)

            self._data_scatter = None
            self._data_extent = None
            self._mask.reset()
            self._mask.commit()

        else:  # There is an active scatter
            self.setEnabled(True)
            self._adjustColorAndBrushSize(activeScatter)

            self._mask.setDataItem(self._data_scatter)
            if self._data_scatter.getXData(copy=False).shape != self._mask.getMask(copy=False).shape:
                self._mask.reset(self._data_scatter.getXData(copy=False).shape)
                self._mask.commit()
            else:
                # Refresh in case z changed
                self._updatePlotMask()

        self._updateInteractiveMode()

    # Handle whole mask operations

    def load(self, filename):
        """Load a mask from an image file.

        :param str filename: File name from which to load the mask
        :raise Exception: An exception in case of failure
        :raise RuntimeWarning: In case the mask was applied but with some
            import changes to notice
        """
        _, extension = os.path.splitext(filename)
        extension = extension.lower()[1:]
        if extension == "npy":
            try:
                mask = numpy.load(filename)
            except IOError:
                _logger.error("Can't load filename '%s'", filename)
                _logger.debug("Backtrace", exc_info=True)
                raise RuntimeError('File "%s" is not a numpy file.',
                                   filename)
        elif extension in ["txt", "csv"]:
            try:
                mask = numpy.loadtxt(filename)
            except IOError:
                _logger.error("Can't load filename '%s'", filename)
                _logger.debug("Backtrace", exc_info=True)
                raise RuntimeError('File "%s" is not a numpy txt file.',
                                   filename)
        else:
            msg = "Extension '%s' is not supported."
            raise RuntimeError(msg % extension)

        self.setSelectionMask(mask, copy=False)

    def _loadMask(self):
        """Open load mask dialog"""
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Load Mask")
        dialog.setModal(1)
        filters = [
            'NumPy binary file (*.npy)',
            'CSV text file (*.csv)',
        ]
        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        dialog.setDirectory(self.maskFileDir)
        if not dialog.exec_():
            dialog.close()
            return

        filename = dialog.selectedFiles()[0]
        dialog.close()

        self.maskFileDir = os.path.dirname(filename)
        try:
            self.load(filename)
        # except RuntimeWarning as e:
        #     message = e.args[0]
        #     msg = qt.QMessageBox(self)
        #     msg.setIcon(qt.QMessageBox.Warning)
        #     msg.setText("Mask loaded but an operation was applied.\n" + message)
        #     msg.exec_()
        except Exception as e:
            message = e.args[0]
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot load mask from file. " + message)
            msg.exec_()

    def _saveMask(self):
        """Open Save mask dialog"""
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Save Mask")
        dialog.setModal(1)
        filters = [
            'NumPy binary file (*.npy)',
            'CSV text file (*.csv)',
        ]
        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.AnyFile)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setDirectory(self.maskFileDir)
        if not dialog.exec_():
            dialog.close()
            return

        # convert filter name to extension name with the .
        extension = dialog.selectedNameFilter().split()[-1][2:-1]
        filename = dialog.selectedFiles()[0]
        dialog.close()

        if not filename.lower().endswith(extension):
            filename += extension

        if os.path.exists(filename):
            try:
                os.remove(filename)
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot save.\n"
                            "Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec_()
                return

        self.maskFileDir = os.path.dirname(filename)
        try:
            self.save(filename, extension[1:])
        except Exception as e:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot save file %s\n%s" % (filename, e.args[0]))
            msg.exec_()

    def resetSelectionMask(self):
        """Reset the mask"""
        self._mask.reset(
                shape=self._data_scatter.getXData(copy=False).shape)
        self._mask.commit()

    def _getPencilWidth(self):
        """Returns the width of the pencil to use in data coordinates`

        :rtype: float
        """
        width = super(ScatterMaskToolsWidget, self)._getPencilWidth()
        if self._data_extent is not None:
            width *= 0.01 * self._data_extent
        return width

    def _plotDrawEvent(self, event):
        """Handle draw events from the plot"""
        if (self._drawingMode is None or
                event['event'] not in ('drawingProgress', 'drawingFinished')):
            return

        if not len(self._data_scatter.getXData(copy=False)):
            return

        level = self.levelSpinBox.value()

        if self._drawingMode == 'rectangle':
            if event['event'] == 'drawingFinished':
                doMask = self._isMasking()

                self._mask.updateRectangle(
                    level,
                    y=event['y'],
                    x=event['x'],
                    height=abs(event['height']),
                    width=abs(event['width']),
                    mask=doMask)
                self._mask.commit()

        elif self._drawingMode == 'ellipse':
            if event['event'] == 'drawingFinished':
                doMask = self._isMasking()
                center = event['points'][0]
                size = event['points'][1]
                self._mask.updateEllipse(level, center[1], center[0],
                                         size[1], size[0], doMask)
                self._mask.commit()

        elif self._drawingMode == 'polygon':
            if event['event'] == 'drawingFinished':
                doMask = self._isMasking()
                vertices = event['points']
                vertices = vertices[:, (1, 0)]  # (y, x)
                self._mask.updatePolygon(level, vertices, doMask)
                self._mask.commit()

        elif self._drawingMode == 'pencil':
            doMask = self._isMasking()
            # convert from plot to array coords
            x, y = event['points'][-1]

            brushSize = self._getPencilWidth()

            if self._lastPencilPos != (y, x):
                if self._lastPencilPos is not None:
                    # Draw the line
                    self._mask.updateLine(
                        level,
                        self._lastPencilPos[0], self._lastPencilPos[1],
                        y, x,
                        brushSize,
                        doMask)

                # Draw the very first, or last point
                self._mask.updateDisk(level, y, x, brushSize / 2., doMask)

            if event['event'] == 'drawingFinished':
                self._mask.commit()
                self._lastPencilPos = None
            else:
                self._lastPencilPos = y, x
        else:
            _logger.error("Drawing mode %s unsupported", self._drawingMode)

    def _loadRangeFromColormapTriggered(self):
        """Set range from active scatter colormap range"""
        if self._data_scatter is not None:
            # Update thresholds according to colormap
            colormap = self._data_scatter.getColormap()
            if colormap['autoscale']:
                min_ = numpy.nanmin(self._data_scatter.getValueData(copy=False))
                max_ = numpy.nanmax(self._data_scatter.getValueData(copy=False))
            else:
                min_, max_ = colormap['vmin'], colormap['vmax']
            self.minLineEdit.setText(str(min_))
            self.maxLineEdit.setText(str(max_))


class ScatterMaskToolsDockWidget(BaseMaskToolsDockWidget):
    """:class:`ScatterMaskToolsWidget` embedded in a QDockWidget.

    For integration in a :class:`PlotWindow`.

    :param parent: See :class:`QDockWidget`
    :param plot: The PlotWidget this widget is operating on
    :paran str name: The title of this widget
    """
    def __init__(self, parent=None, plot=None, name='Mask'):
        widget = ScatterMaskToolsWidget(plot=plot)
        super(ScatterMaskToolsDockWidget, self).__init__(parent, name, widget)
