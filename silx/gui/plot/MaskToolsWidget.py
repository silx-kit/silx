# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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

This widget is meant to work with :class:`silx.gui.plot.PlotWidget`.

- :class:`ImageMask`: Handle mask bitmap update and history
- :class:`MaskToolsWidget`: GUI for :class:`Mask`
- :class:`MaskToolsDockWidget`: DockWidget to integrate in :class:`PlotWindow`
"""
from __future__ import division


__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "29/08/2018"


import os
import sys
import numpy
import logging
import collections

from silx.image import shapes

from ._BaseMaskToolsWidget import BaseMask, BaseMaskToolsWidget, BaseMaskToolsDockWidget
from . import items
from ..colors import cursorColorForColormap, rgba
from .. import qt

from silx.third_party.EdfFile import EdfFile
from silx.third_party.TiffIO import TiffIO

try:
    import fabio
except ImportError:
    fabio = None


_logger = logging.getLogger(__name__)


class ImageMask(BaseMask):
    """A 2D mask field with update operations.

    Coords follows (row, column) convention and are in mask array coords.

    This is meant for internal use by :class:`MaskToolsWidget`.
    """
    def __init__(self, image=None):
        """

        :param image: :class:`silx.gui.plot.items.ImageBase` instance
        """
        BaseMask.__init__(self, image)
        self.reset(shape=(0, 0))  # Init the mask with a 2D shape

    def getDataValues(self):
        """Return image data as a 2D or 3D array (if it is a RGBA image).

        :rtype: 2D or 3D numpy.ndarray
        """
        return self._dataItem.getData(copy=False)

    def save(self, filename, kind):
        """Save current mask in a file

        :param str filename: The file where to save to mask
        :param str kind: The kind of file to save in 'edf', 'tif', 'npy',
            or 'msk' (if FabIO is installed)
        :raise Exception: Raised if the file writing fail
        """
        if kind == 'edf':
            edfFile = EdfFile(filename, access="w+")
            edfFile.WriteImage({}, self.getMask(copy=False), Append=0)

        elif kind == 'tif':
            tiffFile = TiffIO(filename, mode='w')
            tiffFile.writeImage(self.getMask(copy=False), software='silx')

        elif kind == 'npy':
            try:
                numpy.save(filename, self.getMask(copy=False))
            except IOError:
                raise RuntimeError("Mask file can't be written")

        elif kind == 'msk':
            if fabio is None:
                raise ImportError("Fit2d mask files can't be written: Fabio module is not available")
            try:
                data = self.getMask(copy=False)
                image = fabio.fabioimage.FabioImage(data=data)
                image = image.convert(fabio.fit2dmaskimage.Fit2dMaskImage)
                image.save(filename)
            except Exception:
                _logger.debug("Backtrace", exc_info=True)
                raise RuntimeError("Mask file can't be written")

        else:
            raise ValueError("Format '%s' is not supported" % kind)

    # Drawing operations
    def updateRectangle(self, level, row, col, height, width, mask=True):
        """Mask/Unmask a rectangle of the given mask level.

        :param int level: Mask level to update.
        :param int row: Starting row of the rectangle
        :param int col: Starting column of the rectangle
        :param int height:
        :param int width:
        :param bool mask: True to mask (default), False to unmask.
        """
        assert 0 < level < 256
        selection = self._mask[max(0, row):row + height + 1,
                               max(0, col):col + width + 1]
        if mask:
            selection[:, :] = level
        else:
            selection[selection == level] = 0
        self._notify()

    def updatePolygon(self, level, vertices, mask=True):
        """Mask/Unmask a polygon of the given mask level.

        :param int level: Mask level to update.
        :param vertices: Nx2 array of polygon corners as (row, col)
        :param bool mask: True to mask (default), False to unmask.
        """
        fill = shapes.polygon_fill_mask(vertices, self._mask.shape)
        if mask:
            self._mask[fill != 0] = level
        else:
            self._mask[numpy.logical_and(fill != 0,
                                         self._mask == level)] = 0
        self._notify()

    def updatePoints(self, level, rows, cols, mask=True):
        """Mask/Unmask points with given coordinates.

        :param int level: Mask level to update.
        :param rows: Rows of selected points
        :type rows: 1D numpy.ndarray
        :param cols: Columns of selected points
        :type cols: 1D numpy.ndarray
        :param bool mask: True to mask (default), False to unmask.
        """
        valid = numpy.logical_and(
            numpy.logical_and(rows >= 0, cols >= 0),
            numpy.logical_and(rows < self._mask.shape[0],
                              cols < self._mask.shape[1]))
        rows, cols = rows[valid], cols[valid]

        if mask:
            self._mask[rows, cols] = level
        else:
            inMask = self._mask[rows, cols] == level
            self._mask[rows[inMask], cols[inMask]] = 0
        self._notify()

    def updateDisk(self, level, crow, ccol, radius, mask=True):
        """Mask/Unmask a disk of the given mask level.

        :param int level: Mask level to update.
        :param int crow: Disk center row.
        :param int ccol: Disk center column.
        :param float radius: Radius of the disk in mask array unit
        :param bool mask: True to mask (default), False to unmask.
        """
        rows, cols = shapes.circle_fill(crow, ccol, radius)
        self.updatePoints(level, rows, cols, mask)

    def updateLine(self, level, row0, col0, row1, col1, width, mask=True):
        """Mask/Unmask a line of the given mask level.

        :param int level: Mask level to update.
        :param int row0: Row of the starting point.
        :param int col0: Column of the starting point.
        :param int row1: Row of the end point.
        :param int col1: Column of the end point.
        :param int width: Width of the line in mask array unit.
        :param bool mask: True to mask (default), False to unmask.
        """
        rows, cols = shapes.draw_line(row0, col0, row1, col1, width)
        self.updatePoints(level, rows, cols, mask)


class MaskToolsWidget(BaseMaskToolsWidget):
    """Widget with tools for drawing mask on an image in a PlotWidget."""

    _maxLevelNumber = 255

    def __init__(self, parent=None, plot=None):
        super(MaskToolsWidget, self).__init__(parent, plot,
                                              mask=ImageMask())
        self._origin = (0., 0.)  # Mask origin in plot
        self._scale = (1., 1.)  # Mask scale in plot
        self._z = 1  # Mask layer in plot
        self._data = numpy.zeros((0, 0), dtype=numpy.uint8)  # Store image

    def setSelectionMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask:
            The array to use for the mask or None to reset the mask.
        :type mask: numpy.ndarray of uint8 of dimension 2, C-contiguous.
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        :return: None if failed, shape of mask as 2-tuple if successful.
                 The mask can be cropped or padded to fit active image,
                 the returned shape is that of the active image.
        """
        if mask is None:
            self.resetSelectionMask()
            return self._data.shape[:2]

        mask = numpy.array(mask, copy=False, dtype=numpy.uint8)
        if len(mask.shape) != 2:
            _logger.error('Not an image, shape: %d', len(mask.shape))
            return None

        # if mask has not changed, do nothing
        if numpy.array_equal(mask, self.getSelectionMask()):
            return mask.shape

        # ensure all mask attributes are synchronized with the active image
        # and connect listener
        activeImage = self.plot.getActiveImage()
        if activeImage is not None and activeImage.getLegend() != self._maskName:
            self._activeImageChanged()
            self.plot.sigActiveImageChanged.connect(self._activeImageChanged)

        if self._data.shape[0:2] == (0, 0) or mask.shape == self._data.shape[0:2]:
            self._mask.setMask(mask, copy=copy)
            self._mask.commit()
            return mask.shape
        else:
            _logger.warning('Mask has not the same size as current image.'
                            ' Mask will be cropped or padded to fit image'
                            ' dimensions. %s != %s',
                            str(mask.shape), str(self._data.shape))
            resizedMask = numpy.zeros(self._data.shape[0:2],
                                      dtype=numpy.uint8)
            height = min(self._data.shape[0], mask.shape[0])
            width = min(self._data.shape[1], mask.shape[1])
            resizedMask[:height, :width] = mask[:height, :width]
            self._mask.setMask(resizedMask, copy=False)
            self._mask.commit()
            return resizedMask.shape

    # Handle mask refresh on the plot
    def _updatePlotMask(self):
        """Update mask image in plot"""
        mask = self.getSelectionMask(copy=False)
        if mask is not None:
            # get the mask from the plot
            maskItem = self.plot.getImage(self._maskName)
            mustBeAdded = maskItem is None
            if mustBeAdded:
                maskItem = items.MaskImageData()
                maskItem._setLegend(self._maskName)
            # update the items
            maskItem.setData(mask, copy=False)
            maskItem.setColormap(self._colormap)
            maskItem.setOrigin(self._origin)
            maskItem.setScale(self._scale)
            maskItem.setZValue(self._z)

            if mustBeAdded:
                self.plot._add(maskItem)

        elif self.plot.getImage(self._maskName):
            self.plot.remove(self._maskName, kind='image')

    def showEvent(self, event):
        try:
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChangedAfterCare)
        except (RuntimeError, TypeError):
            pass
        self._activeImageChanged()  # Init mask + enable/disable widget
        self.plot.sigActiveImageChanged.connect(self._activeImageChanged)

    def hideEvent(self, event):
        try:
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChanged)
        except (RuntimeError, TypeError):
            pass
        if self.isMaskInteractionActivated():
            # Disable drawing tool
            self.browseAction.trigger()

        if self.getSelectionMask(copy=False) is not None:
            self.plot.sigActiveImageChanged.connect(
                self._activeImageChangedAfterCare)

    def _setOverlayColorForImage(self, image):
        """Set the color of overlay adapted to image

        :param image: :class:`.items.ImageBase` object to set color for.
        """
        if isinstance(image, items.ColormapMixIn):
            colormap = image.getColormap()
            self._defaultOverlayColor = rgba(
                cursorColorForColormap(colormap['name']))
        else:
            self._defaultOverlayColor = rgba('black')

    def _activeImageChangedAfterCare(self, *args):
        """Check synchro of active image and mask when mask widget is hidden.

        If active image has no more the same size as the mask, the mask is
        removed, otherwise it is adjusted to origin, scale and z.
        """
        activeImage = self.plot.getActiveImage()
        if activeImage is None or activeImage.getLegend() == self._maskName:
            # No active image or active image is the mask...
            self._data = numpy.zeros((0, 0), dtype=numpy.uint8)
            self._mask.setDataItem(None)
            self._mask.reset()

            if self.plot.getImage(self._maskName):
                self.plot.remove(self._maskName, kind='image')

            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChangedAfterCare)
        else:
            self._setOverlayColorForImage(activeImage)
            self._setMaskColors(self.levelSpinBox.value(),
                                self.transparencySlider.value() /
                                self.transparencySlider.maximum())

            self._origin = activeImage.getOrigin()
            self._scale = activeImage.getScale()
            self._z = activeImage.getZValue() + 1
            self._data = activeImage.getData(copy=False)
            if self._data.shape[:2] != self._mask.getMask(copy=False).shape:
                # Image has not the same size, remove mask and stop listening
                if self.plot.getImage(self._maskName):
                    self.plot.remove(self._maskName, kind='image')

                self.plot.sigActiveImageChanged.disconnect(
                    self._activeImageChangedAfterCare)
            else:
                # Refresh in case origin, scale, z changed
                self._mask.setDataItem(activeImage)
                self._updatePlotMask()

    def _activeImageChanged(self, *args):
        """Update widget and mask according to active image changes"""
        activeImage = self.plot.getActiveImage()
        if (activeImage is None or activeImage.getLegend() == self._maskName or
                activeImage.getData(copy=False).size == 0):
            # No active image or active image is the mask or image has no data...
            self.setEnabled(False)

            self._data = numpy.zeros((0, 0), dtype=numpy.uint8)
            self._mask.reset()
            self._mask.commit()

        else:  # There is an active image
            self.setEnabled(True)

            self._setOverlayColorForImage(activeImage)

            self._setMaskColors(self.levelSpinBox.value(),
                                self.transparencySlider.value() /
                                self.transparencySlider.maximum())

            self._origin = activeImage.getOrigin()
            self._scale = activeImage.getScale()
            self._z = activeImage.getZValue() + 1
            self._data = activeImage.getData(copy=False)
            self._mask.setDataItem(activeImage)
            if self._data.shape[:2] != self._mask.getMask(copy=False).shape:
                self._mask.reset(self._data.shape[:2])
                self._mask.commit()
            else:
                # Refresh in case origin, scale, z changed
                self._updatePlotMask()

            # Threshold tools only available for data with colormap
            self.thresholdGroup.setEnabled(self._data.ndim == 2)

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
                raise RuntimeError('File "%s" is not a numpy file.', filename)
        elif extension in ["tif", "tiff"]:
            try:
                image = TiffIO(filename, mode="r")
                mask = image.getImage(0)
            except Exception as e:
                _logger.error("Can't load filename %s", filename)
                _logger.debug("Backtrace", exc_info=True)
                raise e
        elif extension == "edf":
            try:
                mask = EdfFile(filename, access='r').GetData(0)
            except Exception as e:
                _logger.error("Can't load filename %s", filename)
                _logger.debug("Backtrace", exc_info=True)
                raise e
        elif extension == "msk":
            if fabio is None:
                raise ImportError("Fit2d mask files can't be read: Fabio module is not available")
            try:
                mask = fabio.open(filename).data
            except Exception as e:
                _logger.error("Can't load fit2d mask file")
                _logger.debug("Backtrace", exc_info=True)
                raise e
        else:
            msg = "Extension '%s' is not supported."
            raise RuntimeError(msg % extension)

        effectiveMaskShape = self.setSelectionMask(mask, copy=False)
        if effectiveMaskShape is None:
            return
        if mask.shape != effectiveMaskShape:
            msg = 'Mask was resized from %s to %s'
            msg = msg % (str(mask.shape), str(effectiveMaskShape))
            raise RuntimeWarning(msg)

    def _loadMask(self):
        """Open load mask dialog"""
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Load Mask")
        dialog.setModal(1)

        extensions = collections.OrderedDict()
        extensions["EDF files"] = "*.edf"
        extensions["TIFF files"] = "*.tif *.tiff"
        extensions["NumPy binary files"] = "*.npy"
        # Fit2D mask is displayed anyway fabio is here or not
        # to show to the user that the option exists
        extensions["Fit2D mask files"] = "*.msk"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

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
        except RuntimeWarning as e:
            message = e.args[0]
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText("Mask loaded but an operation was applied.\n" + message)
            msg.exec_()
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
            'EDF (*.edf)',
            'TIFF (*.tif)',
            'NumPy binary file (*.npy)',
            # Fit2D mask is displayed anyway fabio is here or not
            # to show to the user that the option exists
            'Fit2D mask (*.msk)',
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
        self._mask.reset(shape=self._data.shape[:2])
        self._mask.commit()

    def _plotDrawEvent(self, event):
        """Handle draw events from the plot"""
        if (self._drawingMode is None or
                event['event'] not in ('drawingProgress', 'drawingFinished')):
            return

        if not len(self._data):
            return

        level = self.levelSpinBox.value()

        if (self._drawingMode == 'rectangle' and
                event['event'] == 'drawingFinished'):
            # Convert from plot to array coords
            doMask = self._isMasking()
            ox, oy = self._origin
            sx, sy = self._scale

            height = int(abs(event['height'] / sy))
            width = int(abs(event['width'] / sx))

            row = int((event['y'] - oy) / sy)
            if sy < 0:
                row -= height

            col = int((event['x'] - ox) / sx)
            if sx < 0:
                col -= width

            self._mask.updateRectangle(
                level,
                row=row,
                col=col,
                height=height,
                width=width,
                mask=doMask)
            self._mask.commit()

        elif (self._drawingMode == 'polygon' and
                event['event'] == 'drawingFinished'):
            doMask = self._isMasking()
            # Convert from plot to array coords
            vertices = (event['points'] - self._origin) / self._scale
            vertices = vertices.astype(numpy.int)[:, (1, 0)]  # (row, col)
            self._mask.updatePolygon(level, vertices, doMask)
            self._mask.commit()

        elif self._drawingMode == 'pencil':
            doMask = self._isMasking()
            # convert from plot to array coords
            col, row = (event['points'][-1] - self._origin) / self._scale
            col, row = int(col), int(row)
            brushSize = self._getPencilWidth()

            if self._lastPencilPos != (row, col):
                if self._lastPencilPos is not None:
                    # Draw the line
                    self._mask.updateLine(
                        level,
                        self._lastPencilPos[0], self._lastPencilPos[1],
                        row, col,
                        brushSize,
                        doMask)

                # Draw the very first, or last point
                self._mask.updateDisk(level, row, col, brushSize / 2., doMask)

            if event['event'] == 'drawingFinished':
                self._mask.commit()
                self._lastPencilPos = None
            else:
                self._lastPencilPos = row, col

    def _loadRangeFromColormapTriggered(self):
        """Set range from active image colormap range"""
        activeImage = self.plot.getActiveImage()
        if (isinstance(activeImage, items.ColormapMixIn) and
                activeImage.getLegend() != self._maskName):
            # Update thresholds according to colormap
            colormap = activeImage.getColormap()
            if colormap['autoscale']:
                min_ = numpy.nanmin(activeImage.getData(copy=False))
                max_ = numpy.nanmax(activeImage.getData(copy=False))
            else:
                min_, max_ = colormap['vmin'], colormap['vmax']
            self.minLineEdit.setText(str(min_))
            self.maxLineEdit.setText(str(max_))


class MaskToolsDockWidget(BaseMaskToolsDockWidget):
    """:class:`MaskToolsWidget` embedded in a QDockWidget.

    For integration in a :class:`PlotWindow`.

    :param parent: See :class:`QDockWidget`
    :param plot: The PlotWidget this widget is operating on
    :paran str name: The title of this widget
    """
    def __init__(self, parent=None, plot=None, name='Mask'):
        widget = MaskToolsWidget(plot=plot)
        super(MaskToolsDockWidget, self).__init__(parent, name, widget)
