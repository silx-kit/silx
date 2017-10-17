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
This module contains utilitaries used by other dialog modules.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/10/2017"

import fabio
from silx.gui import qt


class FileTypeComboBox(qt.QComboBox):
    """
    A combobox providing all image file formats supported by fabio and silx.

    It provides access for each fabio codecs individually.
    """

    EXTENSIONS_ROLE = qt.Qt.UserRole + 1

    FABIO_CODEC_ROLE = qt.Qt.UserRole + 2

    def __init__(self, parent=None):
        qt.QComboBox.__init__(self, parent)
        self.__initItems()

    def __initItems(self):
        formats = fabio.fabioformats.get_classes(reader=True)

        extensions = []
        allExtensions = set([])

        for reader in formats:
            if not hasattr(reader, "DESCRIPTION"):
                continue
            if not hasattr(reader, "DEFAULT_EXTENTIONS"):
                continue

            ext = reader.DEFAULT_EXTENTIONS
            ext = ["*.%s" % e for e in ext]
            allExtensions.update(ext)
            if ext == []:
                ext = ["*"]
            extensions.append((reader.DESCRIPTION, ext, reader.codec_name()))
        extensions = list(sorted(extensions))

        allExtensions = list(sorted(list(allExtensions)))
        index = self.count()
        self.addItem("All supported files")
        self.setItemData(index, allExtensions, role=self.EXTENSIONS_ROLE)
        self.__allSupportedFilesIndex = index

        for e in extensions:
            index = self.count()
            if len(e[1]) < 10:
                self.addItem("%s (%s)" % (e[0], " ".join(e[1])))
            else:
                self.addItem(e[0])
            self.setItemData(index, e[1], role=self.EXTENSIONS_ROLE)
            self.setItemData(index, e[2], role=self.FABIO_CODEC_ROLE)

        index = self.count()
        self.addItem("All files (*)")
        self.setItemData(index, e[1], role=self.EXTENSIONS_ROLE)
        self.__allFilesIndex = index

    def itemExtensions(self, index):
        """Returns the extensions associated to an index."""
        result = self.itemData(index, self.EXTENSIONS_ROLE)
        if result == qt.QVariant.Invalid:
            result = None
        return result

    def currentExtensions(self):
        """Returns the current selected extensions."""
        index = self.currentIndex()
        return self.itemExtensions(index)

    def itemFabioCodec(self, index):
        """Returns the fabio codec associated to an index."""
        result = self.itemData(index, self.FABIO_CODEC_ROLE)
        if result == qt.QVariant.Invalid:
            result = None
        return result

    def currentFabioCodec(self):
        """Returns the current selected fabio codec. None if nothing selected
        or if the item is not a fabio codec"""
        index = self.currentIndex()
        return self.itemFabioCodec(index)
