# coding: utf-8
#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
# #########################################################################*/
"""
"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "21/09/2016"

from silx.gui import qt

class TabsDialog(qt.QDialog):
    """Dialog widget containing a QTabWidget :attr:`tabWidget`
    and a buttons:

        # - buttonHelp
        - buttonDefaults
        - buttonOk
        - buttonCancel

    This dialog defines a __len__ returning the number of tabs,
    and an __iter__ method yielding the tab widgets.
    """
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.tabWidget = qt.QTabWidget(self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.tabWidget)

        layout2 = qt.QHBoxLayout(None)

        # self.buttonHelp = qt.QPushButton(self)
        # self.buttonHelp.setText("Help")
        # layout2.addWidget(self.buttonHelp)

        self.buttonDefaults = qt.QPushButton(self)
        self.buttonDefaults.setText("Defaults")
        layout2.addWidget(self.buttonDefaults)

        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout2.addItem(spacer)

        self.buttonOk = qt.QPushButton(self)
        self.buttonOk.setText("OK")
        layout2.addWidget(self.buttonOk)
        
        self.buttonCancel = qt.QPushButton(self)
        self.buttonCancel.setText(str("Cancel"))
        layout2.addWidget(self.buttonCancel)
        
        layout.addLayout(layout2)

        self.buttonOk.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)

    def __len__(self):
        """Return number of tabs"""
        return self.tabWidget.count()

    def __iter__(self):
        """Return the next tab widget in  :attr:`tabWidget` every
        time this method is called.

        :return: Tab widget
        :rtype: QWidget
        """
        for widget_index in range(len(self)):
            yield self.tabWidget.widget(widget_index)

    def addTab(self, page, label):
        """Add a new tab

        :param page: Content of new page. Must be a widget with
            a get() method returning a dictionary.
        :param str label: Tab label
        """
        self.tabWidget.addTab(page, label)

    def getTabLabels(self):
        """
        Return a list of all tab labels in :attr:`tabWidget`
        """
        return [self.tabWidget.tabText(i) for i in range(len(self))]


class TabsDialogData(TabsDialog):
    """This dialog adds a data attribute to :class:`TabsDialog`.

    Data input in widgets, such as text entries or checkboxes, is stored in an
    attribute :attr:`output` when the user clicks the OK button.

    A default dictionary can be supplied when this dialog is initialized, to
    be used as default data for :attr:`output`.
    """
    def __init__(self, parent=None, modal=True, default=None):
        """

        :param parent: Parent :class:`QWidget`
        :param modal: If `True`, dialog is modal, meaning this dialog remains
            in front of it's parent window and disables it until the user is
            done interacting with the dialog
        :param default: Default dictionary, used to initialize and reset
            :attr:`output`.
        """
        TabsDialog.__init__(self, parent)
        self.setModal(modal)

        self.output = {}

        self.default = {} if default is None else default

    def accept(self):
        """When *OK* is clicked, update :attr:`output` with data from
        various widgets
        """
        self.output.update(self.default)

        # loop over all tab widgets (uses TabsDialog.__iter__)
        for tabWidget in self:
            self.output.update(tabWidget.get())

        # avoid pathological None cases
        for key in self.output.keys():
            if self.output[key] is None:
                if key in self.default:
                    self.output[key] = self.default[key]
        super(TabsDialogData, self).accept()

    def reject(self):
        """When the *Cancel* button is clicked, reinitialize :attr:`output`
        and quit
        """
        self.defaults()
        super(TabsDialogData, self).reject()

    def defaults(self):
        """Reinitialize :attr:`output` with :attr:`default`
        """
        self.output = {}
        self.output.update(self.default)

# class ConstraintsPage(qt.QWidget):
#     def __init__(self, parent=None):
#         super(ConstraintsPage, self).__init__(parent)
#
#         layout =  qt.QVBoxLayout(self)
#         # positive height
#         # ...
#
#
#     def get(self):
#         ...
#
# class SearchPage(qt.QWidget):
#     def __init__(self, parent=None):
#         super(SearchPage, self).__init__(parent)
#
#         layout =  qt.QVBoxLayout(self)
#         # fwhm points
#         # ...
#
#     def get(self):
#         ...
#
# class BackgroundPage(qt.QWidget):
#     def __init__(self, parent=None):
#         super(BackgroundPage, self).__init__(parent)
#
#         layout =  qt.QVBoxLayout(self)
#         # strip width
#         # ...
#
#     def get(self):
#         ...


def main():
    a = qt.QApplication([])

    mw = qt.QMainWindow()
    mw.show()

    td = TabsDialog(mw)
    td.show()
    td.exec_()
    print("TabsDialog result: ", td.result())

    class MyTabWidget(qt.QWidget):
        def __init__(self, key, value):
            qt.QWidget.__init__(self)
            self.key = key
            self.value = value

        def get(self):
            return {self.key: self.value}

    tdd = TabsDialogData(mw, default={"a": 1})
    tdd.addTab(MyTabWidget("b", 2), label="tab b")
    tdd.addTab(MyTabWidget("c", 3), label="tab c")
    tdd.show()
    tdd.exec_()
    print("TabsDialogData result: ", tdd.result())
    print("TabsDialogData output: ", tdd.output)

    a.exec_()

if __name__ == "__main__":
    main()
