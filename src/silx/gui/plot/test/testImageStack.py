# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""Basic tests for ImageStack"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "15/01/2020"


import unittest
import tempfile
import numpy
import h5py

from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.io.url import DataUrl
from silx.gui.plot.ImageStack import ImageStack
from silx.gui.utils.testutils import SignalListener
from collections import OrderedDict
import os
import time
import shutil


class TestImageStack(TestCaseQt):
    """Simple test of the Image stack"""

    def setUp(self):
        TestCaseQt.setUp(self)
        self.urls = OrderedDict()
        self._raw_data = {}
        self._folder = tempfile.mkdtemp()
        self._n_urls = 10
        file_name = os.path.join(self._folder, 'test_inage_stack_file.h5')
        with h5py.File(file_name, 'w') as h5f:
            for i in range(self._n_urls):
                width = numpy.random.randint(10, 40)
                height = numpy.random.randint(10, 40)
                raw_data = numpy.random.random((width, height))
                self._raw_data[i] = raw_data
                h5f[str(i)] = raw_data
                self.urls[i] = DataUrl(file_path=file_name,
                                       data_path=str(i),
                                       scheme='silx')
        self.widget = ImageStack()

        self.urlLoadedListener = SignalListener()
        self.widget.sigLoaded.connect(self.urlLoadedListener)

        self.currentUrlChangedListener = SignalListener()
        self.widget.sigCurrentUrlChanged.connect(self.currentUrlChangedListener)

    def tearDown(self):
        shutil.rmtree(self._folder)
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose, True)
        self.widget.close()
        TestCaseQt.setUp(self)

    def testControls(self):
        """Test that selection using the url table and the slider are working
        """
        self.widget.show()
        self.assertEqual(self.widget.getCurrentUrl(), None)
        self.assertEqual(self.widget.getCurrentUrlIndex(), None)
        self.widget.setUrls(list(self.urls.values()))

        # wait for image to be loaded
        self._waitUntilUrlLoaded()

        self.assertEqual(self.widget.getCurrentUrl(), self.urls[0])

        # make sure all image are loaded
        self.assertEqual(self.urlLoadedListener.callCount(), self._n_urls)
        numpy.testing.assert_array_equal(
            self.widget.getPlotWidget().getActiveImage(just_legend=False).getData(),
            self._raw_data[0])
        self.assertEqual(self.widget._slider.value(), 0)

        self.widget._urlsTable.setUrl(self.urls[4])
        numpy.testing.assert_array_equal(
            self.widget.getPlotWidget().getActiveImage(just_legend=False).getData(),
            self._raw_data[4])
        self.assertEqual(self.widget._slider.value(), 4)
        self.assertEqual(self.widget.getCurrentUrl(), self.urls[4])
        self.assertEqual(self.widget.getCurrentUrlIndex(), 4)

        self.widget._slider.setUrlIndex(6)
        numpy.testing.assert_array_equal(
            self.widget.getPlotWidget().getActiveImage(just_legend=False).getData(),
            self._raw_data[6])
        self.assertEqual(self.widget._urlsTable.currentItem().text(),
                         self.urls[6].path())

    def testCurrentUrlSignals(self):
        """Test emission of 'currentUrlChangedListener'"""
        # check initialization
        self.assertEqual(self.currentUrlChangedListener.callCount(), 0)
        self.widget.setUrls(list(self.urls.values()))
        self.qapp.processEvents()
        time.sleep(0.5)
        self.qapp.processEvents()
        # once loaded the two signals should have been sended
        self.assertEqual(self.currentUrlChangedListener.callCount(), 1)
        # if the slider is stuck to the same position no signal should be
        # emitted
        self.qapp.processEvents()
        time.sleep(0.5)
        self.qapp.processEvents()
        self.assertEqual(self.widget._slider.value(), 0)
        self.assertEqual(self.currentUrlChangedListener.callCount(), 1)
        # if slider position is changed, one of each signal should have been
        # emitted
        self.widget._urlsTable.setUrl(self.urls[4])
        self.qapp.processEvents()
        time.sleep(1.5)
        self.qapp.processEvents()
        self.assertEqual(self.currentUrlChangedListener.callCount(), 2)

    def testUtils(self):
        """Test that some utils functions are working"""
        self.widget.show()
        self.widget.setUrls(list(self.urls.values()))
        self.assertEqual(len(self.widget.getUrls()), len(self.urls))

        # wait for image to be loaded
        self._waitUntilUrlLoaded()

        urls_values = list(self.urls.values())
        self.assertEqual(urls_values[0], self.urls[0])
        self.assertEqual(urls_values[7], self.urls[7])

        self.assertEqual(self.widget._getNextUrl(urls_values[2]).path(),
                         urls_values[3].path())
        self.assertEqual(self.widget._getPreviousUrl(urls_values[0]), None)
        self.assertEqual(self.widget._getPreviousUrl(urls_values[6]).path(),
                         urls_values[5].path())

        self.assertEqual(self.widget._getNNextUrls(2, urls_values[0]),
                         urls_values[1:3])
        self.assertEqual(self.widget._getNNextUrls(5, urls_values[7]),
                         urls_values[8:])
        self.assertEqual(self.widget._getNPreviousUrls(3, urls_values[2]),
                         urls_values[:2])
        self.assertEqual(self.widget._getNPreviousUrls(5, urls_values[8]),
                         urls_values[3:8])

    def _waitUntilUrlLoaded(self, timeout=2.0):
        """Wait until all image urls are loaded"""
        loop_duration = 0.2
        remaining_duration = timeout
        while(len(self.widget._loadingThreads) > 0 and remaining_duration > 0):
            remaining_duration -= loop_duration
            time.sleep(loop_duration)
            self.qapp.processEvents()

        if remaining_duration <= 0.0:
            remaining_urls = []
            for thread_ in self.widget._loadingThreads:
                remaining_urls.append(thread_.url.path())
            mess = 'All images are not loaded after the time out. ' \
                   'Remaining urls are: ' + str(remaining_urls)
            raise TimeoutError(mess)
        return True
