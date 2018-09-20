# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""This module provides Plot3DAction related to input/output.

It provides QAction to copy, save (snapshot and video), print a Plot3DWidget.
"""

from __future__ import absolute_import, division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/09/2017"


import logging
import os

import numpy

from silx.gui import qt, printer
from silx.gui.icons import getQIcon
from .Plot3DAction import Plot3DAction
from ..utils import mng
from ...utils.image import convertQImageToArray


_logger = logging.getLogger(__name__)


class CopyAction(Plot3DAction):
    """QAction to provide copy of a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, plot3d=None):
        super(CopyAction, self).__init__(parent, plot3d)

        self.setIcon(getQIcon('edit-copy'))
        self.setText('Copy')
        self.setToolTip('Copy a snapshot of the 3D scene to the clipboard')
        self.setCheckable(False)
        self.setShortcut(qt.QKeySequence.Copy)
        self.setShortcutContext(qt.Qt.WidgetShortcut)
        self.triggered[bool].connect(self._triggered)

    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error('Cannot copy widget, no associated Plot3DWidget')
        else:
            image = plot3d.grabGL()
            qt.QApplication.clipboard().setImage(image)


class SaveAction(Plot3DAction):
    """QAction to provide save snapshot of a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, plot3d=None):
        super(SaveAction, self).__init__(parent, plot3d)

        self.setIcon(getQIcon('document-save'))
        self.setText('Save...')
        self.setToolTip('Save a snapshot of the 3D scene')
        self.setCheckable(False)
        self.setShortcut(qt.QKeySequence.Save)
        self.setShortcutContext(qt.Qt.WidgetShortcut)
        self.triggered[bool].connect(self._triggered)

    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error('Cannot save widget, no associated Plot3DWidget')
        else:
            dialog = qt.QFileDialog(self.parent())
            dialog.setWindowTitle('Save snapshot as')
            dialog.setModal(True)
            dialog.setNameFilters(('Plot3D Snapshot PNG (*.png)',
                                   'Plot3D Snapshot JPEG (*.jpg)'))

            dialog.setFileMode(qt.QFileDialog.AnyFile)
            dialog.setAcceptMode(qt.QFileDialog.AcceptSave)

            if not dialog.exec_():
                return

            nameFilter = dialog.selectedNameFilter()
            filename = dialog.selectedFiles()[0]
            dialog.close()

            # Forces the filename extension to match the chosen filter
            extension = nameFilter.split()[-1][2:-1]
            if (len(filename) <= len(extension) or
                    filename[-len(extension):].lower() != extension.lower()):
                filename += extension

            image = plot3d.grabGL()
            if not image.save(filename):
                _logger.error('Failed to save image as %s', filename)
                qt.QMessageBox.critical(
                    self.parent(),
                    'Save snapshot as',
                    'Failed to save snapshot')


class PrintAction(Plot3DAction):
    """QAction to provide printing of a Plot3DWidget

    :param parent: See :class:`QAction`
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    def __init__(self, parent, plot3d=None):
        super(PrintAction, self).__init__(parent, plot3d)

        self.setIcon(getQIcon('document-print'))
        self.setText('Print...')
        self.setToolTip('Print a snapshot of the 3D scene')
        self.setCheckable(False)
        self.setShortcut(qt.QKeySequence.Print)
        self.setShortcutContext(qt.Qt.WidgetShortcut)
        self.triggered[bool].connect(self._triggered)

    def getPrinter(self):
        """Return the QPrinter instance used for printing.

        :rtype: QPrinter
        """
        return printer.getDefaultPrinter()

    def _triggered(self, checked=False):
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.error('Cannot print widget, no associated Plot3DWidget')
        else:
            printer = self.getPrinter()
            dialog = qt.QPrintDialog(printer, plot3d)
            dialog.setWindowTitle('Print Plot3D snapshot')
            if not dialog.exec_():
                return

            image = plot3d.grabGL()

            # Draw pixmap with painter
            painter = qt.QPainter()
            if not painter.begin(printer):
                return

            if (printer.pageRect().width() < image.width() or
                    printer.pageRect().height() < image.height()):
                # Downscale to page
                xScale = printer.pageRect().width() / image.width()
                yScale = printer.pageRect().height() / image.height()
                scale = min(xScale, yScale)
            else:
                scale = 1.

            rect = qt.QRectF(0,
                             0,
                             scale * image.width(),
                             scale * image.height())
            painter.drawImage(rect, image)
            painter.end()


class VideoAction(Plot3DAction):
    """This action triggers the recording of a video of the scene.

    The scene is rotated 360 degrees around a vertical axis.

    :param parent: Action parent see :class:`QAction`.
    :param ~silx.gui.plot3d.Plot3DWidget.Plot3DWidget plot3d:
        Plot3DWidget the action is associated with
    """

    PNG_SERIE_FILTER = 'Serie of PNG files (*.png)'
    MNG_FILTER = 'Multiple-image Network Graphics file (*.mng)'

    def __init__(self, parent, plot3d=None):
        super(VideoAction, self).__init__(parent, plot3d)
        self.setText('Record video..')
        self.setIcon(getQIcon('camera'))
        self.setToolTip(
            'Record a video of a 360 degrees rotation of the 3D scene.')
        self.setCheckable(False)
        self.triggered[bool].connect(self._triggered)

    def _triggered(self, checked=False):
        """Action triggered callback"""
        plot3d = self.getPlot3DWidget()
        if plot3d is None:
            _logger.warning(
                'Ignoring action triggered without Plot3DWidget set')
            return

        dialog = qt.QFileDialog(parent=plot3d)
        dialog.setWindowTitle('Save video as...')
        dialog.setModal(True)
        dialog.setNameFilters([self.PNG_SERIE_FILTER,
                               self.MNG_FILTER])
        dialog.setFileMode(dialog.AnyFile)
        dialog.setAcceptMode(dialog.AcceptSave)

        if not dialog.exec_():
            return

        nameFilter = dialog.selectedNameFilter()
        filename = dialog.selectedFiles()[0]

        # Forces the filename extension to match the chosen filter
        extension = nameFilter.split()[-1][2:-1]
        if (len(filename) <= len(extension) or
                filename[-len(extension):].lower() != extension.lower()):
            filename += extension

        nbFrames = int(4. * 25)  # 4 seconds, 25 fps

        if nameFilter == self.PNG_SERIE_FILTER:
            self._saveAsPNGSerie(filename, nbFrames)
        elif nameFilter == self.MNG_FILTER:
            self._saveAsMNG(filename, nbFrames)
        else:
            _logger.error('Unsupported file filter: %s', nameFilter)

    def _saveAsPNGSerie(self, filename, nbFrames):
        """Save video as serie of PNG files.

        It adds a counter to the provided filename before the extension.

        :param str filename: filename to use as template
        :param int nbFrames: Number of frames to generate
        """
        plot3d = self.getPlot3DWidget()
        assert plot3d is not None

        # Define filename template
        nbDigits = int(numpy.log10(nbFrames)) + 1
        indexFormat = '%%0%dd' % nbDigits
        extensionIndex = filename.rfind('.')
        filenameFormat = \
            filename[:extensionIndex] + indexFormat + filename[extensionIndex:]

        try:
            for index, image in enumerate(self._video360(nbFrames)):
                image.save(filenameFormat % index)
        except GeneratorExit:
            pass

    def _saveAsMNG(self, filename, nbFrames):
        """Save video as MNG file.

        :param str filename: filename to use
        :param int nbFrames: Number of frames to generate
        """
        plot3d = self.getPlot3DWidget()
        assert plot3d is not None

        frames = (convertQImageToArray(im) for im in self._video360(nbFrames))
        try:
            with open(filename, 'wb') as file_:
                for chunk in mng.convert(frames, nb_images=nbFrames):
                    file_.write(chunk)
        except GeneratorExit:
            os.remove(filename)  # Saving aborted, delete file

    def _video360(self, nbFrames):
        """Run the video and provides the images

        :param int nbFrames: The number of frames to generate for
        :return: Iterator of QImage of the video sequence
        """
        plot3d = self.getPlot3DWidget()
        assert plot3d is not None

        angleStep = 360. / nbFrames

        # Create progress bar dialog
        dialog = qt.QDialog(plot3d)
        dialog.setWindowTitle('Record Video')
        layout = qt.QVBoxLayout(dialog)
        progress = qt.QProgressBar()
        progress.setRange(0, nbFrames)
        layout.addWidget(progress)

        btnBox = qt.QDialogButtonBox(qt.QDialogButtonBox.Abort)
        btnBox.rejected.connect(dialog.reject)
        layout.addWidget(btnBox)

        dialog.setModal(True)
        dialog.show()

        qapp = qt.QApplication.instance()

        for frame in range(nbFrames):
            progress.setValue(frame)
            image = plot3d.grabGL()
            yield image
            plot3d.viewport.orbitCamera('left', angleStep)
            qapp.processEvents()
            if not dialog.isVisible():
                break  # It as been rejected by the abort button
        else:
            dialog.accept()

        if dialog.result() == qt.QDialog.Rejected:
            raise GeneratorExit('Aborted')
