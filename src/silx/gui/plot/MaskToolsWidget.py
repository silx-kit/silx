# /*##########################################################################
#
# Copyright (c) 2017-2022 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "08/12/2020"

import os
import sys
import numpy
import logging
import collections
import h5py

from silx.image import shapes
from silx.io.utils import NEXUS_HDF5_EXT, is_dataset
from silx.gui.dialog.DatasetDialog import DatasetDialog

from ._BaseMaskToolsWidget import BaseMask, BaseMaskToolsWidget, BaseMaskToolsDockWidget
from . import items
from ..colors import cursorColorForColormap, rgba
from .. import qt
from ..utils import LockReentrant

from silx.third_party.EdfFile import EdfFile
from silx.third_party.TiffIO import TiffIO

import fabio

_logger = logging.getLogger(__name__)

_HDF5_EXT_STR = ' '.join(['*' + ext for ext in NEXUS_HDF5_EXT])


def _selectDataset(filename, mode=DatasetDialog.SaveMode):
    """Open a dialog to prompt the user to select a dataset in
    a hdf5 file.

    :param str filename: name of an existing HDF5 file
    :param mode: DatasetDialog.SaveMode or DatasetDialog.LoadMode
    :rtype: str
    :return: Name of selected dataset
    """
    dialog = DatasetDialog()
    dialog.addFile(filename)
    dialog.setWindowTitle("Select a 2D dataset")
    dialog.setMode(mode)
    if not dialog.exec():
        return None
    return dialog.getSelectedDataUrl().data_path()


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
        :param str kind: The kind of file to save in 'edf', 'tif', 'npy', 'h5'
            or 'msk' (if FabIO is installed)
        :raise Exception: Raised if the file writing fail
        """
        if kind == 'edf':
            edfFile = EdfFile(filename, access="w+")
            header = {"program_name": "silx-mask", "masked_value": "nonzero"}
            edfFile.WriteImage(header, self.getMask(copy=False), Append=0)

        elif kind == 'tif':
            tiffFile = TiffIO(filename, mode='w')
            tiffFile.writeImage(self.getMask(copy=False), software='silx')

        elif kind == 'npy':
            try:
                numpy.save(filename, self.getMask(copy=False))
            except IOError:
                raise RuntimeError("Mask file can't be written")

        elif ("." + kind) in NEXUS_HDF5_EXT:
            self._saveToHdf5(filename, self.getMask(copy=False))

        elif kind == 'msk':
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

    @staticmethod
    def _saveToHdf5(filename, mask):
        """Save a mask array to a HDF5 file.

        :param str filename: name of an existing HDF5 file
        :param numpy.ndarray mask: Mask array.
        :returns: True if operation succeeded, False otherwise.
        """
        if not os.path.exists(filename):
            # create new file
            with h5py.File(filename, "w") as _h5f:
                pass
        dataPath = _selectDataset(filename)
        if dataPath is None:
            return False
        with h5py.File(filename, "a") as h5f:
            existing_ds = h5f.get(dataPath)
            if existing_ds is not None:
                reply = qt.QMessageBox.question(
                        None,
                        "Confirm overwrite",
                        "Do you want to overwrite an existing dataset?",
                        qt.QMessageBox.Yes | qt.QMessageBox.No)
                if reply != qt.QMessageBox.Yes:
                    return False
                del h5f[dataPath]
            try:
                h5f.create_dataset(dataPath, data=mask)
            except Exception:
                return False
        return True

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
        if row + height <= 0 or col + width <= 0:
            return  # Rectangle outside image, avoid negative indices
        selection = self._mask[max(0, row):row + height + 1,
                               max(0, col):col + width + 1]
        if mask:
            selection[:,:] = level
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

    def updateEllipse(self, level, crow, ccol, radius_r, radius_c, mask=True):
        """Mask/Unmask an ellipse of the given mask level.

        :param int level: Mask level to update.
        :param int crow: Row of the center of the ellipse
        :param int ccol: Column of the center of the ellipse
        :param float radius_r: Radius of the ellipse in the row
        :param float radius_c: Radius of the ellipse in the column
        :param bool mask: True to mask (default), False to unmask.
        """
        rows, cols = shapes.ellipse_fill(crow, ccol, radius_r, radius_c)
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

        self.__itemMaskUpdatedLock = LockReentrant()
        self.__itemMaskUpdated = False

    def __maskStateChanged(self) -> None:
        """Handle mask commit to update item mask"""
        item = self._mask.getDataItem()
        if item is not None:
            with self.__itemMaskUpdatedLock:
                item.setMaskData(self._mask.getMask(copy=True), copy=False)

    def setItemMaskUpdated(self, enabled: bool) -> None:
        """Toggle item mask and mask tool synchronisation.

        :param bool enabled: True to synchronise. Default: False
        """
        enabled = bool(enabled)
        if enabled != self.__itemMaskUpdated:
            if self.__itemMaskUpdated:
                self._mask.sigStateChanged.disconnect(self.__maskStateChanged)
            self.__itemMaskUpdated = enabled
            if self.__itemMaskUpdated:
                # Synchronize item and tool mask
                self._setMaskedImage(self._mask.getDataItem())
                self._mask.sigStateChanged.connect(self.__maskStateChanged)

    def isItemMaskUpdated(self) -> bool:
        """Returns whether or not item and mask tool masks are synchronised.

        :rtype: bool
        """
        return self.__itemMaskUpdated

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

        # Handle mask with single level
        if self.multipleMasks() == 'single':
            mask = numpy.array(mask != 0, dtype=numpy.uint8)

        # if mask has not changed, do nothing
        if numpy.array_equal(mask, self.getSelectionMask()):
            return mask.shape

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
            resizedMask[:height,:width] = mask[:height,:width]
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
                maskItem.setName(self._maskName)
            # update the items
            maskItem.setData(mask, copy=False)
            maskItem.setColormap(self._colormap)
            maskItem.setOrigin(self._origin)
            maskItem.setScale(self._scale)
            maskItem.setZValue(self._z)

            if mustBeAdded:
                self.plot.addItem(maskItem)

        elif self.plot.getImage(self._maskName):
            self.plot.remove(self._maskName, kind='image')

    def showEvent(self, event):
        try:
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChangedAfterCare)
        except (RuntimeError, TypeError):
            pass

        # Sync with current active image
        self._setMaskedImage(self.plot.getActiveImage())
        self.plot.sigActiveImageChanged.connect(self._activeImageChanged)

    def hideEvent(self, event):
        try:
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChanged)
        except (RuntimeError, TypeError):
            pass

        image = self.getMaskedItem()
        if image is not None:
            try:
                image.sigItemChanged.disconnect(self.__imageChanged)
            except (RuntimeError, TypeError):
                pass  # TODO should not happen

        if self.isMaskInteractionActivated():
            # Disable drawing tool
            self.plot.resetInteractiveMode()

        if self.isItemMaskUpdated():  # No "after-care"
            self._data = numpy.zeros((0, 0), dtype=numpy.uint8)
            self._mask.setDataItem(None)
            self._mask.reset()

            if self.plot.getImage(self._maskName):
                self.plot.remove(self._maskName, kind='image')

        elif self.getSelectionMask(copy=False) is not None:
            self.plot.sigActiveImageChanged.connect(
                self._activeImageChangedAfterCare)

    def _activeImageChanged(self, previous, current):
        """Reacts upon active image change.

        Only handle change of active image items here.
        """
        if previous != current:
            image = self.plot.getActiveImage()
            if image is not None and image.getName() == self._maskName:
                image = None  # Active image is the mask
            self._setMaskedImage(image)

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
        if activeImage is None or activeImage.getName() == self._maskName:
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

    def _setMaskedImage(self, image):
        """Change the image that is used a reference to author the mask"""
        previous = self.getMaskedItem()
        if previous is not None and self.isVisible():
            # Disconnect from previous image
            try:
                previous.sigItemChanged.disconnect(self.__imageChanged)
            except (RuntimeError, TypeError):
                pass  # TODO fixme should not happen

        # Set the image
        self._mask.setDataItem(image)

        if image is None:  # No image, disable mask
            self.setEnabled(False)

            self._data = numpy.zeros((0, 0), dtype=numpy.uint8)
            self._mask.reset()
            self._mask.commit()

            self._updateInteractiveMode()

        else:  # Update and connect to image's sigItemChanged
            if self.isItemMaskUpdated():
                if image.getMaskData(copy=False) is None:
                    # Image item has no mask: use current mask from the tool
                    image.setMaskData(
                        self.getSelectionMask(copy=False), copy=True)
                else:  # Image item has a mask: set it in tool
                    self.setSelectionMask(
                        image.getMaskData(copy=False), copy=True)
                    self._mask.resetHistory()
            self.__imageUpdated()
            if self.isVisible():
                image.sigItemChanged.connect(self.__imageChanged)

    def __imageChanged(self, event):
        """Reacts upon image item changes"""
        image = self._mask.getDataItem()
        if image is None:
            _logger.error("Mask is not attached to an image")
            return

        if event in (items.ItemChangedType.COLORMAP,
                     items.ItemChangedType.DATA,
                     items.ItemChangedType.POSITION,
                     items.ItemChangedType.SCALE,
                     items.ItemChangedType.VISIBLE,
                     items.ItemChangedType.ZVALUE):
            self.__imageUpdated()

        elif (event == items.ItemChangedType.MASK and
                self.isItemMaskUpdated() and
                not self.__itemMaskUpdatedLock.locked()):
            # Update mask from the image item unless mask tool is updating it
            self.setSelectionMask(image.getMaskData(copy=False), copy=True)

    def __imageUpdated(self):
        """Synchronize mask with current state of the image"""
        image = self._mask.getDataItem()
        if image is None:
            _logger.error("No active image while expecting one")
            return

        self._setOverlayColorForImage(image)

        self._setMaskColors(self.levelSpinBox.value(),
                            self.transparencySlider.value() /
                            self.transparencySlider.maximum())

        self._origin = image.getOrigin()
        self._scale = image.getScale()
        self._z = image.getZValue() + 1
        self._data = image.getData(copy=False)
        self._mask.setDataItem(image)
        if self._data.shape[:2] != self._mask.getMask(copy=False).shape:
            self._mask.reset(self._data.shape[:2])
            self._mask.commit()
        else:
            # Refresh in case origin, scale, z changed
            self._updatePlotMask()

        # Visible and with data
        self.setEnabled(image.isVisible() and self._data.size != 0)

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
            try:
                mask = fabio.open(filename).data
            except Exception as e:
                _logger.error("Can't load fit2d mask file")
                _logger.debug("Backtrace", exc_info=True)
                raise e
        elif ("." + extension) in NEXUS_HDF5_EXT:
            mask = self._loadFromHdf5(filename)
            if mask is None:
                raise IOError("Could not load mask from HDF5 dataset")
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
        extensions["HDF5 files"] = _HDF5_EXT_STR
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
        if not dialog.exec():
            dialog.close()
            return

        filename = dialog.selectedFiles()[0]
        dialog.close()

        # Update the directory according to the user selection
        self.maskFileDir = os.path.dirname(filename)

        try:
            self.load(filename)
        except RuntimeWarning as e:
            message = e.args[0]
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText("Mask loaded but an operation was applied.\n" + message)
            msg.exec()
        except Exception as e:
            message = e.args[0]
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot load mask from file. " + message)
            msg.exec()

    @staticmethod
    def _loadFromHdf5(filename):
        """Load a mask array from a HDF5 file.

        :param str filename: name of an existing HDF5 file
        :returns: A mask as a numpy array, or None if the interactive dialog
            was cancelled
        """
        dataPath = _selectDataset(filename, mode=DatasetDialog.LoadMode)
        if dataPath is None:
            return None

        with h5py.File(filename, "r") as h5f:
            dataset = h5f.get(dataPath)
            if not is_dataset(dataset):
                raise IOError("%s is not a dataset" % dataPath)
            mask = dataset[()]
        return mask

    def _saveMask(self):
        """Open Save mask dialog"""
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Save Mask")
        dialog.setOption(qt.QFileDialog.DontUseNativeDialog)
        dialog.setModal(1)
        hdf5Filter = 'HDF5 (%s)' % _HDF5_EXT_STR
        filters = [
            'EDF (*.edf)',
            'TIFF (*.tif)',
            'NumPy binary file (*.npy)',
            hdf5Filter,
            # Fit2D mask is displayed anyway fabio is here or not
            # to show to the user that the option exists
            'Fit2D mask (*.msk)',
        ]
        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.AnyFile)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setDirectory(self.maskFileDir)

        def onFilterSelection(filt_):
            # disable overwrite confirmation for HDF5,
            # because we append the data to existing files
            if filt_ == hdf5Filter:
                dialog.setOption(qt.QFileDialog.DontConfirmOverwrite)
            else:
                dialog.setOption(qt.QFileDialog.DontConfirmOverwrite, False)

        dialog.filterSelected.connect(onFilterSelection)
        if not dialog.exec():
            dialog.close()
            return

        nameFilter = dialog.selectedNameFilter()
        filename = dialog.selectedFiles()[0]
        dialog.close()

        if "HDF5" in nameFilter:
            has_allowed_ext = False
            for ext in NEXUS_HDF5_EXT:
                if (len(filename) > len(ext) and
                        filename[-len(ext):].lower() == ext.lower()):
                    has_allowed_ext = True
                    extension = ext
            if not has_allowed_ext:
                extension = ".h5"
                filename += ".h5"
        else:
            # convert filter name to extension name with the .
            extension = nameFilter.split()[-1][2:-1]
            if not filename.lower().endswith(extension):
                filename += extension

        if os.path.exists(filename) and "HDF5" not in nameFilter:
            try:
                os.remove(filename)
            except IOError as e:
                msg = qt.QMessageBox(self)
                msg.setWindowTitle("Removing existing file")
                msg.setIcon(qt.QMessageBox.Critical)

                if hasattr(e, "strerror"):
                    strerror = e.strerror
                else:
                    strerror = sys.exc_info()[1]
                msg.setText("Cannot save.\n"
                            "Input Output Error: %s" % strerror)
                msg.exec()
                return

        # Update the directory according to the user selection
        self.maskFileDir = os.path.dirname(filename)

        try:
            self.save(filename, extension[1:])
        except Exception as e:
            msg = qt.QMessageBox(self)
            msg.setWindowTitle("Saving mask file")
            msg.setIcon(qt.QMessageBox.Critical)

            if hasattr(e, "strerror"):
                strerror = e.strerror
            else:
                strerror = sys.exc_info()[1]
            msg.setText("Cannot save file %s\n%s" % (filename, strerror))
            msg.exec()

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

        if self._drawingMode == 'rectangle':
            if event['event'] == 'drawingFinished':
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

        elif self._drawingMode == 'ellipse':
            if event['event'] == 'drawingFinished':
                doMask = self._isMasking()
                # Convert from plot to array coords
                center = (event['points'][0] - self._origin) / self._scale
                size = event['points'][1] / self._scale
                center = center.astype(numpy.int64)  # (row, col)
                self._mask.updateEllipse(level, center[1], center[0], size[1], size[0], doMask)
                self._mask.commit()

        elif self._drawingMode == 'polygon':
            if event['event'] == 'drawingFinished':
                doMask = self._isMasking()
                # Convert from plot to array coords
                vertices = (event['points'] - self._origin) / self._scale
                vertices = vertices.astype(numpy.int64)[:, (1, 0)]  # (row, col)
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
        else:
            _logger.error("Drawing mode %s unsupported", self._drawingMode)

    def _loadRangeFromColormapTriggered(self):
        """Set range from active image colormap range"""
        activeImage = self.plot.getActiveImage()
        if (isinstance(activeImage, items.ColormapMixIn) and
                activeImage.getName() != self._maskName):
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
