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
__date__ = "17/01/2019"

import fabio
import silx.io
from silx.gui import qt


class Codec(object):

    def __init__(self, any_fabio=False, any_silx=False, fabio_codec=None, auto=False):
        self.__any_fabio = any_fabio
        self.__any_silx = any_silx
        self.fabio_codec = fabio_codec
        self.__auto = auto

    def is_autodetect(self):
        return self.__auto

    def is_fabio_codec(self):
        return self.__any_fabio or self.fabio_codec is not None

    def is_silx_codec(self):
        return self.__any_silx


class FileTypeComboBox(qt.QComboBox):
    """
    A combobox providing all image file formats supported by fabio and silx.

    It provides access for each fabio codecs individually.
    """

    EXTENSIONS_ROLE = qt.Qt.UserRole + 1

    CODEC_ROLE = qt.Qt.UserRole + 2

    INDENTATION = u"\u2022 "

    def __init__(self, parent=None):
        qt.QComboBox.__init__(self, parent)
        self.__fabioUrlSupported = True
        self.__initItems()

    def setFabioUrlSupproted(self, isSupported):
        if self.__fabioUrlSupported == isSupported:
            return
        self.__fabioUrlSupported = isSupported
        self.__initItems()

    def __initItems(self):
        self.clear()
        if self.__fabioUrlSupported:
            self.__insertFabioFormats()
        self.__insertSilxFormats()
        self.__insertAllSupported()
        self.__insertAnyFiles()

    def __insertAnyFiles(self):
        index = self.count()
        self.addItem("All files (*)")
        self.setItemData(index, ["*"], role=self.EXTENSIONS_ROLE)
        self.setItemData(index, Codec(auto=True), role=self.CODEC_ROLE)

    def __insertAllSupported(self):
        allExtensions = set([])
        for index in range(self.count()):
            ext = self.itemExtensions(index)
            allExtensions.update(ext)
        allExtensions = allExtensions - set("*")
        list(sorted(list(allExtensions)))
        index = 0
        self.insertItem(index, "All supported files")
        self.setItemData(index, allExtensions, role=self.EXTENSIONS_ROLE)
        self.setItemData(index, Codec(auto=True), role=self.CODEC_ROLE)

    def __insertSilxFormats(self):
        formats = silx.io.supported_extensions()

        extensions = []
        allExtensions = set([])

        for description, ext in formats.items():
            allExtensions.update(ext)
            if ext == []:
                ext = ["*"]
            extensions.append((description, ext, "silx"))
        extensions = list(sorted(extensions))

        allExtensions = list(sorted(list(allExtensions)))
        index = self.count()
        self.addItem("All supported files, using Silx")
        self.setItemData(index, allExtensions, role=self.EXTENSIONS_ROLE)
        self.setItemData(index, Codec(any_silx=True), role=self.CODEC_ROLE)

        for e in extensions:
            index = self.count()
            if len(e[1]) < 10:
                self.addItem("%s%s (%s)" % (self.INDENTATION, e[0], " ".join(e[1])))
            else:
                self.addItem("%s%s" % (self.INDENTATION, e[0]))
            codec = Codec(any_silx=True)
            self.setItemData(index, e[1], role=self.EXTENSIONS_ROLE)
            self.setItemData(index, codec, role=self.CODEC_ROLE)

    def __insertFabioFormats(self):
        formats = fabio.fabioformats.get_classes(reader=True)

        from fabio import fabioutils
        if hasattr(fabioutils, "COMPRESSED_EXTENSIONS"):
            compressedExtensions = fabioutils.COMPRESSED_EXTENSIONS
        else:
            # Support for fabio < 0.9
            compressedExtensions = set(["gz", "bz2"])

        extensions = []
        allExtensions = set([])

        def extensionsIterator(reader):
            for extension in reader.DEFAULT_EXTENSIONS:
                yield "*.%s" % extension
            for compressedExtension in compressedExtensions:
                for extension in reader.DEFAULT_EXTENSIONS:
                    yield "*.%s.%s" % (extension, compressedExtension)

        for reader in formats:
            if not hasattr(reader, "DESCRIPTION"):
                continue
            if not hasattr(reader, "DEFAULT_EXTENSIONS"):
                continue

            displayext = reader.DEFAULT_EXTENSIONS
            displayext = ["*.%s" % e for e in displayext]
            ext = list(extensionsIterator(reader))
            allExtensions.update(ext)
            if ext == []:
                ext = ["*"]
            extensions.append((reader.DESCRIPTION, displayext, ext, reader.codec_name()))
        extensions = list(sorted(extensions))

        allExtensions = list(sorted(list(allExtensions)))
        index = self.count()
        self.addItem("All supported files, using Fabio")
        self.setItemData(index, allExtensions, role=self.EXTENSIONS_ROLE)
        self.setItemData(index, Codec(any_fabio=True), role=self.CODEC_ROLE)

        for e in extensions:
            description, displayExt, allExt, _codecName = e
            index = self.count()
            if len(e[1]) < 10:
                self.addItem("%s%s (%s)" % (self.INDENTATION, description, " ".join(displayExt)))
            else:
                self.addItem("%s%s" % (self.INDENTATION, description))
            codec = Codec(fabio_codec=_codecName)
            self.setItemData(index, allExt, role=self.EXTENSIONS_ROLE)
            self.setItemData(index, codec, role=self.CODEC_ROLE)

    def itemExtensions(self, index):
        """Returns the extensions associated to an index."""
        result = self.itemData(index, self.EXTENSIONS_ROLE)
        if result is None:
            result = None
        return result

    def currentExtensions(self):
        """Returns the current selected extensions."""
        index = self.currentIndex()
        return self.itemExtensions(index)

    def indexFromCodec(self, codecName):
        for i in range(self.count()):
            codec = self.itemCodec(i)
            if codecName == "auto":
                if codec.is_autodetect():
                    return i
            elif codecName == "silx":
                if codec.is_silx_codec():
                    return i
            elif codecName == "fabio":
                if codec.is_fabio_codec() and codec.fabio_codec is None:
                    return i
            elif codecName == codec.fabio_codec:
                return i
        return -1

    def itemCodec(self, index):
        """Returns the codec associated to an index."""
        result = self.itemData(index, self.CODEC_ROLE)
        if result is None:
            result = None
        return result

    def currentCodec(self):
        """Returns the current selected codec. None if nothing selected
        or if the item is not a codec"""
        index = self.currentIndex()
        return self.itemCodec(index)
