# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""Try to close system popup before testing our application.

The application can be tested with a set of samples:

>>> scp -r www.silx.org:/data/distributions/ci_popups .
>>> python ./ci/close_popup.py ci_popups
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "05/09/2017"


import os
import sys
import logging
import time
import pynput

try:
    from PyQt5 import Qt as qt
except ImportError:
    qt = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("close_popup")


def getScreenShot(qapp):
    """
    Get a screenshot of the full screen.

    :rtype: qt.QImage
    """
    if not hasattr(qapp, "primaryScreen"):
        # Qt4
        winId = qt.QApplication.desktop().winId()
        pixmap = qt.QPixmap.grabWindow(winId)
    else:
        # Qt5
        screen = qapp.primaryScreen()
        pixmap = screen.grabWindow(0)
    image = pixmap.toImage()
    return image


class CheckPopup(object):
    """Generic class to detect a popup and the location of the button to
    close it"""

    def check(self, image):
        """Return true if the popup is there

        :param qt.QImage image: Image of the screen
        """
        raise NotImplementedError()

    def clickPosition(self):
        """Return the x,y coord defined to close the popup

        :rtype: tuple(int, int)
        """
        raise NotImplementedError()

    def isMostlySameColor(self, color1, color2):
        """
        Returns true if color1 and color2 are mostly the same.

        The delta is based on the sum of the difference between each RBG
        components.

        :rtype: bool
        """
        delta = 0
        delta += abs(color1.red() - color2.red())
        delta += abs(color1.green() - color2.green())
        delta += abs(color1.blue() - color2.blue())
        return delta < 10

    def checkColors(self, image, pixelsDescription):
        """
        Returns true if the pixel description match with the image.

        :param qt.QImage image: Image to check
        :param pixelsDescription: List of pixel expectation containing a
            position, a text description, and an expected color.
        :rtype: bool
        """
        for description in pixelsDescription:
            pos, _description, expectedColor = description
            rgb = image.pixel(pos[0], pos[1])
            color = qt.QColor(rgb)
            if not self.isMostlySameColor(color, expectedColor):
                return False
        return True


class CheckWindowsPopup_NetworkDeviceDiscovery(CheckPopup):

    platform = "win32"
    name = "network device discovery"

    def check(self, image):
        screenSize = image.width(), image.height()
        if screenSize != (1024, 768):
            return False

        expectedPixelColors = [
            ((926, 88), "popup", qt.QColor("#061f5e")),
            ((798, 372), "button", qt.QColor("#0077c6")),
            ((726, 165), "text", qt.QColor("#ffffff")),
        ]
        return self.checkColors(image, expectedPixelColors)

    def clickPosition(self):
        return (798, 372)


class CheckMacOsXPopup_NameAsBeenChanged(CheckPopup):

    platform = "darwin"
    name = "computer renamed"

    def check(self, image):
        screenSize = image.width(), image.height()
        if screenSize != (1024, 768):
            return False

        delta_locations = [-5, 0, 5, 10, 15, 20]

        for delta in delta_locations:
            expectedPixelColors = [
                ((430, 150 + delta), "header", qt.QColor("#f6f6f6")),
                ((388, 190 + delta), "popup", qt.QColor("#ececec")),
                ((637, 324 + delta), "yes button", qt.QColor("#ffffff")),
                ((364, 213 + delta), "logo", qt.QColor("#ecc520")),
            ]
            detected = self.checkColors(image, expectedPixelColors)
            if detected:
                self.delta = delta
                return True
        return False

    def clickPosition(self):
        return (660, 324 + self.delta)


def checkPopups(popupList, filename):
    """Check if an image contains a popup from the provided list

    :param str filename: Name of the file to check or a directory.
    """
    if os.path.isdir(filename):
        base_dir = filename
        filenames = os.listdir(base_dir)
        filenames = [os.path.join(base_dir, filename) for filename in filenames]
    else:
        filenames = [filename]

    for filename in filenames:
        print(filename)
        pixmap = qt.QPixmap(filename)
        if pixmap.isNull():
            logger.debug("File %s skipped.", filename)
            continue
        image = pixmap.toImage()
        detected = False
        for popup in popupList:
            if popup.check(image):
                print("- Popup '%s' is visible." % popup.name)
                detected = True
        if not detected:
            print("- No popups detected.")


def closePopup(qapp, popupList):
    """Check the list of popups and close them

    :param qt.QApplication qapp: Qt application
    :param list popupList: List of popup definitions
    """
    popup_found_count = 0
    for _ in range(10):
        image = getScreenShot(qapp)

        popup_found = False
        for popup in popupList:
            if sys.platform != popup.platform:
                logger.debug("Popup %s skipped, wrong platform.", popup.name)
                continue

            if popup.check(image):
                logger.info("Popup '%s' found. Try to close it.", popup.name)
                mouse = pynput.mouse.Controller()
                mouse.position = popup.clickPosition()
                mouse.click(pynput.mouse.Button.left)
                time.sleep(5)
                popup_found = True
                popup_found_count += 1

        if not popup_found:
            break

    if popup_found_count == 0:
        logger.info("No popup found.")
    else:
        logger.info("No more popup found.")


def main():
    if qt is None:
        logger.info("Qt is not available.")
        return

    popupList = [
        CheckWindowsPopup_NetworkDeviceDiscovery(),
        CheckMacOsXPopup_NameAsBeenChanged(),
    ]
    logger.info("Popup database: %d.", len(popupList))

    qapp = qt.QApplication([])

    if len(sys.argv) == 2:
        logger.info("Check input path.")
        checkPopups(popupList, sys.argv[1])
    else:
        logger.info("Check and close popups.")
        closePopup(qapp, popupList)


if __name__ == "__main__":
    main()
