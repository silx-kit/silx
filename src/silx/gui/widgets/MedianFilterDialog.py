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
""" MedianFilterDialog
Classes
-------

Widgets:

 - :class:`MedianFilterDialog`
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "14/02/2017"


import logging

from silx.gui import qt


_logger = logging.getLogger(__name__)

class MedianFilterDialog(qt.QDialog):
    """QDialog window featuring a :class:`BackgroundWidget`"""
    sigFilterOptChanged = qt.Signal(int, bool)

    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)

        self.setWindowTitle("Median filter options")
        self.mainLayout = qt.QHBoxLayout(self)
        self.setLayout(self.mainLayout)

        # filter width GUI
        self.mainLayout.addWidget(qt.QLabel('filter width:', parent = self))
        self._filterWidth = qt.QSpinBox(parent=self)
        self._filterWidth.setMinimum(1)
        self._filterWidth.setValue(1)
        self._filterWidth.setSingleStep(2);
        widthTooltip = """radius width of the pixel including in the filter
                        for each pixel"""
        self._filterWidth.setToolTip(widthTooltip)
        self._filterWidth.valueChanged.connect(self._filterOptionChanged)
        self.mainLayout.addWidget(self._filterWidth)

        # filter option GUI
        self._filterOption = qt.QCheckBox('conditional', parent=self)
        conditionalTooltip = """if check, implement a conditional filter"""
        self._filterOption.stateChanged.connect(self._filterOptionChanged)
        self.mainLayout.addWidget(self._filterOption)

    def _filterOptionChanged(self):
        """Call back used when the filter values are changed"""
        if self._filterWidth.value()%2 == 0:
            _logger.warning('median filter only accept odd values')
        else:
            self.sigFilterOptChanged.emit(self._filterWidth.value(), self._filterOption.isChecked())