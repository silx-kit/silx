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
"""silx.gui.plot PlotWindow Qt designer plugin."""

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

from silx.gui.plot import PlotWindow


class PlotWindowPlugin(QtDesigner.QPyDesignerCustomWidgetPlugin):

    def __init__(self, parent=None):
        super(PlotWindowPlugin, self).__init__(parent)
        self.initialized = False

    def initialize(self, core):
        if self.initialized:
            return

        self.initialized = True

    def isInitialized(self):
        return self.initialized

    def createWidget(self, parent):
        return PlotWindow(parent, autoreplot=False)

    def name(self):
        return "PlotWindow"

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
        return "silx.gui.plot.PlotWindow"


_logo_pixmap = [
    "32 32 69 1",
    " 	c None",
    ".	c #9FA1A4",
    "+	c #BEC1C2",
    "@	c #AABCB1",
    "#	c #B7BFBB",
    "$	c #BFC0C3",
    "%	c #A5A7AA",
    "&	c #AEB0B3",
    "*	c #C1CC9C",
    "=	c #14B168",
    "-	c #00A754",
    ";	c #FFFFFF",
    ">	c #16231B",
    ",	c #6A582C",
    "'	c #94938B",
    ")	c #FEF1E2",
    "!	c #C4888A",
    "~	c #9EC6C9",
    "{	c #4A4A4C",
    "]	c #8E7D6B",
    "^	c #635D56",
    "/	c #FFFCF9",
    "(	c #FDC385",
    "_	c #F9A13D",
    ":	c #CAA94B",
    "<	c #365331",
    "[	c #A76C2A",
    "}	c #B4C3A2",
    "|	c #DBE1C3",
    "1	c #D0AB8C",
    "2	c #CDBCA9",
    "3	c #D5D6D8",
    "4	c #94816C",
    "5	c #605E5B",
    "6	c #FFFEFC",
    "7	c #A7A9AC",
    "8	c #C3C5C7",
    "9	c #CAC0B3",
    "0	c #C7C2BD",
    "a	c #A9ACAB",
    "b	c #AEB4B0",
    "c	c #B8B9BB",
    "d	c #ACAEB1",
    "e	c #C1C2C4",
    "f	c #E7E8E9",
    "g	c #D8D9DA",
    "h	c #7EBC82",
    "i	c #96C697",
    "j	c #98C898",
    "k	c #E1E2E3",
    "l	c #F0F0F1",
    "m	c #212122",
    "n	c #7C7D7F",
    "o	c #819783",
    "p	c #C7C8CA",
    "q	c #87898C",
    "r	c #6F7072",
    "s	c #000000",
    "t	c #545557",
    "u	c #D4D5D6",
    "v	c #454547",
    "w	c #8FB090",
    "x	c #B9BBBD",
    "y	c #939598",
    "z	c #626365",
    "A	c #8AA88C",
    "B	c #63B16D",
    "C	c #8EAF8F",
    "D	c #7F9582",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    "                                ",
    " .+@#$$$$$$$$$$$$$$$$$$$$$$$$$$%",
    " &*=-;;>,';)!~;;{;;;]^/;;;;;;;;$",
    " &(_:;;<[};|12;;.3;;456;;;;;;;;$",
    " 789088abc888888888888888888888d",
    " efffffffffffffffffffffffffffffg",
    " efffffffhiiiiiiiiiiiiiiihfffffg",
    " efffffffj;;;;;;kl;;;;;;;jfffffg",
    " efffffffj;;;;;kmn;;;;;;;jfffffg",
    " efffffffop;;;;qrsl;;;;;;jfffffg",
    " efffffffj;;;;;turd;;;;;;jfffffg",
    " efffffffj;;;;uv;.r;;;;;;jfffffg",
    " efffffffwk;;dmx;ks;;;;;;jfffffg",
    " efffffffwk;;sl;;;txyk;;;jfffffg",
    " efffffffj;;pt;;;;qrsn;;;jfffffg",
    " efffffffwk;.r;;;;uszs;;;jfffffg",
    " efffffffwkrvu;;;;;xkvp;;jfffffg",
    " efffffffjksl;;;;;;;;nyu;jfffffg",
    " efffffffjqz;;;;;;;;;xstljfffffg",
    " efffffffAtx;;;;;;;;;;qndjfffffg",
    " efffffffj;;;;;;;;;;;;;;ljfffffg",
    " efffffffj;;kk;;p;;p;;p;;jfffffg",
    " efffffffBiiCCiiDiiDiiDiiBfffffg",
    " efffffffffffffffffffffffffffffg",
    "  eeeeeeeeeeeeeeeeeeeeeeeeeeeee&",
    "                                ",
    "                                "]
