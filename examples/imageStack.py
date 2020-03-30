#!/usr/bin/env python
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
"""
Simple example for using the ImageStack.

In this example we want to display images from different source: .h5, .edf
and .npy files.

To do so we simple reimplement the thread managing the loading of data.
"""

import numpy
import h5py
import tempfile
import logging
import shutil
import os
import time
from silx.io.url import DataUrl
from silx.io.utils import get_data
from silx.gui import qt
from silx.gui.plot.ImageStack import ImageStack, UrlLoader
import fabio


logging.basicConfig()
_logger = logging.getLogger("hdf5widget")
"""Module logger"""


def create_random_image():
    """Create a simple image with random values"""
    width = numpy.random.randint(100, 400)
    height = numpy.random.randint(100, 400)
    return numpy.random.random((width, height))


def create_h5py_urls(n_url, file_name):
    """ creates n urls based on h5py"""
    res = []
    with h5py.File(file_name, 'w') as h5f:
        for i in range(n_url):
            h5f[str(i)] = create_random_image()
            res.append(DataUrl(file_path=file_name,
                               data_path=str(i),
                               scheme='silx'))
    return res


def create_numpy_url(file_name):
    """ create a simple DataUrl with a .npy file """
    numpy.save(file=file_name, arr=create_random_image())
    return [DataUrl(file_path=file_name,
                    scheme='numpy'), ]


def create_edf_url(file_name):
    """ create a simple DataUrl with a .edf file"""
    dsc = fabio.edfimage.EdfImage(data=create_random_image(), header={})
    dsc.write(file_name)
    return [DataUrl(file_path=file_name,
                    data_slice=(0,),
                    scheme='fabio'), ]


def create_datasets(folder):
    """create a set of DataUrl containing each one image"""
    urls = []
    file_ = os.path.join(folder, 'myh5file.h5')
    urls.extend(create_h5py_urls(n_url=5, file_name=file_))
    file_ = os.path.join(folder, 'secondH5file.h5')
    urls.extend(create_h5py_urls(n_url=2, file_name=file_))
    file_ = os.path.join(folder, 'firstnumpy_file.npy')
    urls.extend(create_numpy_url(file_name=file_))
    file_ = os.path.join(folder, 'secondnumpy_file.npy')
    urls.extend(create_numpy_url(file_name=file_))
    file_ = os.path.join(folder, 'single_edf_file.edf')
    urls.extend(create_edf_url(file_name=file_))
    file_ = os.path.join(folder, 'single_edf_file_2.edf')
    urls.extend(create_edf_url(file_name=file_))
    return urls


class MyOwnUrlLoader(UrlLoader):
    """
    Thread use to load DataUrl
    """
    def __init__(self, parent, url):
        super(MyOwnUrlLoader, self).__init__(parent=parent, url=url)
        self.url = url
        self.data = None

    def run(self):
        # just to see the waiting interface...
        time.sleep(1.0)
        if self.url.scheme() == 'numpy':
            self.data = numpy.load(self.url.file_path())
        else:
            self.data = get_data(self.url)


def main():
    dataset_folder = tempfile.mkdtemp()

    qapp = qt.QApplication([])
    widget = ImageStack()
    widget.setUrlLoaderClass(MyOwnUrlLoader)
    widget.setNPrefetch(1)
    urls = create_datasets(folder=dataset_folder)
    widget.setUrls(urls=urls)
    widget.show()
    qapp.exec_()
    widget.close()

    shutil.rmtree(dataset_folder)


if __name__ == '__main__':
    main()
    exit(0)
