#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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
"""silx.gui.plot PlotWidget Qt designer plugin."""

__authors__ = ["T. Vincent", "F. Mengoni"]
__license__ = "MIT"
__date__ = "09/03/2016"


from silx.gui import qt

if qt.BINDING == 'PyQt4':
    from PyQt4 import QtDesigner
elif qt.BINDING == 'PyQt5':
    from PyQt5 import QtDesigner
else:
    raise RuntimeError("Unsupport Qt BINDING: %s" % qt.BINDING)

from silx.gui.plot import PlotWidget


class PlotWidgetPlugin(QtDesigner.QPyDesignerCustomWidgetPlugin):

    def __init__(self, parent=None):
        super(PlotWidgetPlugin, self).__init__(parent)
        self.initialized = False

    def initialize(self, core):
        if self.initialized:
            return

        self.initialized = True

    def isInitialized(self):
        return self.initialized

    def createWidget(self, parent):
        return PlotWidget(parent, autoreplot=False)

    def name(self):
        return "PlotWidget"

    def group(self):
        return "silx"

    def icon(self):
        return qt.QIcon(qt.QPixmap(_logo_pixmap))

    def toolTip(self):
        return ""

    def whatsThis(self):
        return ""

    def isContainer(self):
        return False

    def includeFile(self):
        return "silx.gui.plot.PlotWidget"


_logo_pixmap = [
    "32 32 25 1",
    " 	c None",
    ".	c #7AB77F",
    "+	c #BDD3BB",
    "@	c #84BB88",
    "#	c #E7E8E9",
    "$	c #CFD0D2",
    "%	c #C3C5C7",
    "&	c #3E3E40",
    "*	c #000000",
    "=	c #ABADB0",
    "-	c #505052",
    ";	c #7C9C7F",
    ">	c #9FA1A4",
    ",	c #696A6C",
    "'	c #1F1F20",
    ")	c #7F8083",
    "!	c #DBDCDD",
    "~	c #747578",
    "{	c #8A8C8F",
    "]	c #949699",
    "^	c #5D5E60",
    "/	c #B7B8BB",
    "(	c #707173",
    "_	c #61AF6C",
    ":	c #83AC84",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "       .+++++++++++++++++++++.  ",
    "       @########$%###########@  ",
    "       @########&*=##########@  ",
    "       @#######=**-##########@  ",
    "       ;>######,')*%#########@  ",
    "       @######!'~$*)#########@  ",
    "       @######{*=#-&#########@  ",
    "       @####!~*,##]*$########@  ",
    "       ;>###,*,!##%*>#%!#####@  ",
    "  * *  @####*^#####*~>*,#####@  ",
    "   **  @####*~#####-',**#####@  ",
    "   *   @###/*{#####)**'*/####@  ",
    "       ;>#>'*$#####$**)*]####@  ",
    "       @#/*'/#######/>#*~####@  ",
    "       @#{*%###########,&%###@  ",
    "       @#,^############>**{##@  ",
    "       @%']#############&**%#@  ",
    "       ;(-!#############=>']#@  ",
    "       @!!################%!#@  ",
    "       @#####################@  ",
    "       @###//###{###{###//###@  ",
    "       _@@@::@@@;@@@;@@@::@@@_  ",
    "                                ",
    "                                ",
    "                 ***            ",
    "                 **             ",
    "                 ***            ",
    "                                "]
