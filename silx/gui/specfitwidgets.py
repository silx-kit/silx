#/*##########################################################################
# Copyright (C) 2004-2016 European Synchrotron Radiation Facility
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
"""Widgets used to build :class:`specfitgui.SpecfitGui`"""

from collections import OrderedDict
from .parameters import Parameters

from silx.gui import qt

QTVERSION = qt.qVersion()

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "06/06/2016"

class FitActionsButtons(qt.QWidget):
    """Widget with 3 ``QPushButton``:

    The buttons can be accessed as public attributes::

        - ``EstimateButton``
        - ``StartfitButton``
        - ``DismissButton``

    You will typically need to access these attributes to connect the buttons
    to actions. For instance, if you have 3 functions ``estimate``,
    ``startfit`` and  ``dismiss``, you can connect them like this::

        >>> fit_actions_buttons = FitActionsButtons()
        >>> fit_actions_buttons.EstimateButton.clicked.connect(estimate)
        >>> fit_actions_buttons.StartfitButton.clicked.connect(startfit)
        >>> fit_actions_buttons.DismissButton.clicked.connect(dismiss)

    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.resize(234, 53)

        grid_layout = qt.QGridLayout(self)
        grid_layout.setContentsMargins(11, 11, 11, 11)
        grid_layout.setSpacing(6)
        layout = qt.QHBoxLayout(None)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.EstimateButton = qt.QPushButton(self)
        self.EstimateButton.setText("Estimate")
        layout.addWidget(self.EstimateButton)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout.addItem(spacer)

        self.StartfitButton = qt.QPushButton(self)
        self.StartfitButton.setText("Start Fit")
        layout.addWidget(self.StartfitButton)
        spacer_2 = qt.QSpacerItem(20, 20,
                                  qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        layout.addItem(spacer_2)

        self.DismissButton = qt.QPushButton(self)
        self.DismissButton.setText("Dismiss")
        layout.addWidget(self.DismissButton)

        grid_layout.addLayout(layout, 0, 0)


class FitStatusLines(qt.QWidget):
    """Widget with 2 greyed out write-only ``QLineEdit``.

    These text widgets can be accessed as public attributes::

        - ``StatusLine``
        - ``ChisqLine``

    You will typically need to access these widgets to update the displayed
    text::

        >>> fit_status_lines = FitStatusLines()
        >>> fit_status_lines.StatusLine.setText("Ready")
        >>> fit_status_lines.ChisqLine.setText("%6.2f" % 0.01)

    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self,  parent)

        self.resize(535, 47)

        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(11, 11, 11, 11)
        layout.setSpacing(6)

        self.StatusLabel = qt.QLabel(self)
        self.StatusLabel.setText("Status:")
        layout.addWidget(self.StatusLabel)

        self.StatusLine = qt.QLineEdit(self)
        self.StatusLine.setText("Ready")
        self.StatusLine.setReadOnly(1)
        layout.addWidget(self.StatusLine)

        self.ChisqLabel = qt.QLabel(self)
        self.ChisqLabel.setText("Chisq:")
        layout.addWidget(self.ChisqLabel)

        self.ChisqLine = qt.QLineEdit(self)
        self.ChisqLine.setMaximumSize(qt.QSize(16000, 32767))
        self.ChisqLine.setText("")
        self.ChisqLine.setReadOnly(1)
        layout.addWidget(self.ChisqLine)


class FitConfigWidget(qt.QWidget):
    """Widget with 2 ``QComboBox``, 4 ``QCheckBox`` and 2 ``QPushButtons``.

    These text widgets can be accessed as public attributes::

        - ``BkgComBox``
        - ``FunComBox``
        - ``WeightCheckBox``
        - ``MCACheckBox``
        - ``AutoFWHMCheckBox``
        - ``AutoScalingCheckBox``
        - ``PrintPushButton``
        - ``ConfigureButton``
    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.setWindowTitle("FitConfigGUI")

        FitConfigGUILayout = qt.QHBoxLayout(self)
        FitConfigGUILayout.setContentsMargins(11, 11, 11, 11)
        FitConfigGUILayout.setSpacing(6)

        layout9 = qt.QHBoxLayout(None)
        layout9.setContentsMargins(0, 0, 0, 0)
        layout9.setSpacing(6)

        layout2 = qt.QGridLayout(None)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.setSpacing(6)

        self.BkgComBox = qt.QComboBox(self)
        self.BkgComBox.addItem("Add Background")

        layout2.addWidget(self.BkgComBox, 1, 1)

        self.BkgLabel = qt.QLabel(self)
        self.BkgLabel.setText("Background")

        layout2.addWidget(self.BkgLabel, 1, 0)

        self.FunComBox = qt.QComboBox(self)
        self.FunComBox.addItem("Add Function(s)")

        layout2.addWidget(self.FunComBox, 0, 1)

        self.FunLabel = qt.QLabel(self)
        self.FunLabel.setText("Function")

        layout2.addWidget(self.FunLabel, 0, 0)
        layout9.addLayout(layout2)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout9.addItem(spacer)

        layout6 = qt.QGridLayout(None)
        layout6.setContentsMargins(0, 0, 0, 0)
        layout6.setSpacing(6)

        self.WeightCheckBox = qt.QCheckBox(self)
        self.WeightCheckBox.setText("Weight")

        layout6.addWidget(self.WeightCheckBox, 0, 0)

        self.MCACheckBox = qt.QCheckBox(self)
        self.MCACheckBox.setText("MCA Mode")

        layout6.addWidget(self.MCACheckBox, 1, 0)
        layout9.addLayout(layout6)

        layout6_2 = qt.QGridLayout(None)
        layout6_2.setContentsMargins(0, 0, 0, 0)
        layout6_2.setSpacing(6)

        self.AutoFWHMCheckBox = qt.QCheckBox(self)
        self.AutoFWHMCheckBox.setText("Auto FWHM")

        layout6_2.addWidget(self.AutoFWHMCheckBox, 0, 0)

        self.AutoScalingCheckBox = qt.QCheckBox(self)
        self.AutoScalingCheckBox.setText("Auto Scaling")

        layout6_2.addWidget(self.AutoScalingCheckBox, 1, 0)
        layout9.addLayout(layout6_2)
        spacer_2 = qt.QSpacerItem(20, 20, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        layout9.addItem(spacer_2)

        layout5 = qt.QGridLayout(None)
        layout5.setContentsMargins(0, 0, 0, 0)
        layout5.setSpacing(6)

        self.PrintPushButton = qt.QPushButton(self)
        self.PrintPushButton.setText("Print")

        layout5.addWidget(self.PrintPushButton, 1, 0)

        self.ConfigureButton = qt.QPushButton(self)
        self.ConfigureButton.setText("Configure")

        layout5.addWidget(self.ConfigureButton, 0, 0)
        layout9.addLayout(layout5)
        FitConfigGUILayout.addLayout(layout9)


class McaTable(qt.QTableWidget):
    sigMcaTableSignal = qt.pyqtSignal(object)

    def __init__(self, labels=None, *args, **kw):
        qt.QTableWidget.__init__(self, *args)
        self.setRowCount(1)
        self.setColumnCount(1)

        self.code_options = ["FREE", "POSITIVE", "QUOTED",
                             "FIXED", "FACTOR", "DELTA", "SUM", "IGNORE", "ADD", "SHOW"]

        if labels is not None:
            self.labels = labels
        else:
            self.labels = ['Position', 'Fit Area', 'MCA Area', 'Sigma',
                           'Fwhm', 'Chisq', 'Region', 'XBegin', 'XEnd']

        self.setColumnCount(len(self.labels))
        for i, label in enumerate(self.labels):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
                self.setHorizontalHeaderItem(i, item)
            item.setText(label)
            self.resizeColumnToContents(i)

        self.regionlist = []
        self.regiondict = {}

        self.cellClicked[int, int].connect(self.__myslot)
        self.itemSelectionChanged[()].connect(self.__myslot)

    def fillfrommca(self, mcaresult, diag=1):
        line0 = 0
        region = 0
        alreadyforced = 0
        for result in mcaresult:
            region += 1
            if result['chisq'] is not None:
                chisq = "%6.2f" % (result['chisq'])
            else:
                chisq = "Fit Error"
            if 1:
                xbegin = "%6g" % (result['xbegin'])
                xend = "%6g" % (result['xend'])
                fitlabel, fitpars, fitsigmas = self.__getfitpar(result)
                if QTVERSION < '4.0.0':
                    qt.QHeader.setLabel(
                        self.horizontalHeader(), 1, "Fit " + fitlabel)
                else:
                    item = self.horizontalHeaderItem(1)
                    item.setText("Fit " + fitlabel)
                i = 0
                for (pos, area, sigma, fwhm) in result['mca_areas']:
                    line0 += 1
                    if QTVERSION < '4.0.0':
                        nlines = self.numRows()
                        if (line0 > nlines):
                            self.setNumRows(line0)
                    else:
                        nlines = self.rowCount()
                        if (line0 > nlines):
                            self.setRowCount(line0)
                    line = line0 - 1
                    pos = "%6g" % (pos)
                    fitpar = "%6g" % (fitpars[i])
                    if fitlabel == 'Area':
                        sigma = max(sigma, fitsigmas[i])
                    areastr = "%6g" % (area)
                    sigmastr = "%6.3g" % (sigma)
                    fwhm = "%6g" % (fwhm)
                    tregion = "%6g" % (region)
                    fields = [pos, fitpar, areastr, sigmastr,
                              fwhm, chisq, tregion, xbegin, xend]
                    col = 0
                    recolor = 0
                    if fitlabel == 'Area':
                        if diag:
                            if abs(fitpars[i] - area) > (3.0 * sigma):
                                color = qt.QColor(255, 182, 193)
                                recolor = 1
                    for field in fields:
                        key = self.item(line, col)
                        if key is None:
                            key = qt.QTableWidgetItem(field)
                            self.setItem(line, col, key)
                        else:
                            item.setText(field)
                        if recolor:
                            # function introduced in Qt 4.2.0
                            if QTVERSION >= '4.2.0':
                                item.setBackground(qt.QBrush(color))
                        item.setFlags(qt.Qt.ItemIsSelectable |
                                      qt.Qt.ItemIsEnabled)
                        col = col + 1
                    if recolor:
                        if not alreadyforced:
                            alreadyforced = 1
                            self.scrollToItem(self.item(line, 0))
                    i += 1

        for i in range(len(self.labels)):
            self.resizeColumnToContents(i)
        ndict = {}
        ndict['event'] = 'McaTableFilled'
        self.sigMcaTableSignal.emit(ndict)

    def __getfitpar(self, result):
        if result['fitconfig']['fittheory'].find("Area") != -1:
            fitlabel = 'Area'
        elif result['fitconfig']['fittheory'].find("Hypermet") != -1:
            fitlabel = 'Area'
        else:
            fitlabel = 'Height'
        values = []
        sigmavalues = []
        for param in result['paramlist']:
            if param['name'].find('ST_Area') != -1:
                # value and sigmavalue known via fitlabel
                values[-1] = value * (1.0 + param['fitresult'])
                # just an approximation
                sigmavalues[-1] = sigmavalue * (1.0 + param['fitresult'])
            elif param['name'].find('LT_Area') != -1:
                pass
            elif param['name'].find(fitlabel) != -1:
                value = param['fitresult']
                sigmavalue = param['sigma']
                values.append(value)
                sigmavalues.append(sigmavalue)
        return fitlabel, values, sigmavalues

    def __myslot(self, *var):
        ddict = {}
        if len(var) == 0:
            # selection changed event
            # get the current selection
            ddict['event'] = 'McaTableClicked'
            row = self.currentRow()
        else:
            # Header click
            ddict['event'] = 'McaTableRowHeaderClicked'
            row = var[0]
        ccol = self.currentColumn()
        ddict['row'] = row
        ddict['col'] = ccol
        ddict['labelslist'] = self.labels
        if row >= 0:
            col = 0
            for label in self.labels:
                text = str(self.item(row, col).text())
                try:
                    ddict[label] = float(text)
                except:
                    ddict[label] = text
                col += 1
        self.sigMcaTableSignal.emit(ddict)


class ParametersTab(qt.QTabWidget):
    sigMultiParametersSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, name="FitParameters"):
        qt.QTabWidget.__init__(self, parent)
        self.setWindowTitle(name)

        # the widgets in the notebook
        self.views = OrderedDict()
        # the widgets/tables themselves
        self.tables = {}
        self.mcatable = None
        self.setContentsMargins(10, 10, 10, 10)
        self.setview(name="Region 1")

    def setview(self, name=None, fitparameterslist=None):
        if name is None:
            name = self.current

        if name in self.tables.keys():
            table = self.tables[name]
        else:
            # create the parameters instance
            self.tables[name] = Parameters(self)
            table = self.tables[name]
            self.views[name] = table
            self.addTab(table, str(name))

        if fitparameterslist is not None:
            table.fillfromfit(fitparameterslist)

        if QTVERSION < '4.0.0':
            self.showPage(self.views[name])
        else:
            self.setCurrentWidget(self.views[name])

        self.current = name

    def renameview(self, oldname=None, newname=None):
        error = 1
        if newname is not None:
            if newname not in self.views.keys():
                if oldname in self.views.keys():
                    parameterlist = self.tables[oldname].fillfitfromtable()
                    self.setview(name=newname,
                                 fitparameterslist=parameterlist)
                    self.removeview(oldname)
                    error = 0
        return error

    def fillfromfit(self, fitparameterslist, current=None):
        if current is None:
            current = self.current

        self.setview(fitparameterslist=fitparameterslist,
                     name=current)

    def fillfitfromtable(self, name=None):
        if name is None:
            name = self.current

        if hasattr(self.tables[name], 'fillfitfromtable'):
            return self.tables[name].fillfitfromtable()
        else:
            return None

    def removeview(self, view=None):
        error = 1
        if view is None:
            return error

        if view == self.current:
            return error

        if view in self.views.keys():
            if QTVERSION < '4.0.0':
                self.removePage(self.tables[view])
                self.removePage(self.views[view])
            else:
                index = self.indexOf(self.tables[view])
                self.removeTab(index)
                index = self.indexOf(self.views[view])
                self.removeTab(index)
            del self.tables[view]
            del self.views[view]
            error = 0
        return error

    def removeallviews(self, keep='Fit'):
        for view in list(self.tables.keys()):
            if view != keep:
                self.removeview(view)

    def fillfrommca(self, mcaresult):
        self.removeallviews()
        region = 0

        for result in mcaresult:
            region = region + 1
            self.fillfromfit(result['paramlist'],
                             current='Region %d' % region)
        name = 'MCA'
        if name in self.tables:
            table = self.tables[name]
        else:
            self.tables[name] = McaTable(self)
            table = self.tables[name]
            self.views[name] = table
            self.addTab(table, str(name))
            table.sigMcaTableSignal.connect(self.__forward)
        table.fillfrommca(mcaresult)
        self.setview(name=name)
        return

    def __forward(self, ddict):
        self.sigMultiParametersSignal.emit(ddict)

    def gettext(self, name=None):
        if name is None:
            name = self.current
        table = self.tables[name]
        lemon = ("#%x%x%x" % (255, 250, 205)).upper()
        if QTVERSION < '4.0.0':
            hb = table.horizontalHeader().paletteBackgroundColor()
            hcolor = ("#%x%x%x" % (hb.red(), hb.green(), hb.blue())).upper()
        else:
            hcolor = ("#%x%x%x" % (230, 240, 249)).upper()
        text = ""
        text += ("<nobr>")
        text += ("<table>")
        text += ("<tr>")
        if QTVERSION < '4.0.0':
            ncols = table.numCols()
        else:
            ncols = table.columnCount()
        for l in range(ncols):
            text += ('<td align="left" bgcolor="%s"><b>' % hcolor)
            if QTVERSION < '4.0.0':
                text += (str(table.horizontalHeader().label(l)))
            else:
                text += (str(table.horizontalHeaderItem(l).text()))
            text += ("</b></td>")
        text += ("</tr>")
        if QTVERSION < '4.0.0':
            nrows = table.numRows()
        else:
            nrows = table.rowCount()
        for r in range(nrows):
            text += ("<tr>")
            if QTVERSION < '4.0.0':
                newtext = str(table.text(r, 0))
            else:
                item = table.item(r, 0)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
            if len(newtext):
                color = "white"
                b = "<b>"
            else:
                b = ""
                color = lemon
            try:
                # MyQTable item has color defined
                cc = table.item(r, 0).color
                cc = ("#%x%x%x" % (cc.red(), cc.green(), cc.blue())).upper()
                color = cc
            except:
                pass
            for c in range(ncols):
                if QTVERSION < '4.0.0':
                    newtext = str(table.text(r, c))
                else:
                    item = table.item(r, c)
                    newtext = ""
                    if item is not None:
                        newtext = str(item.text())
                if len(newtext):
                    finalcolor = color
                else:
                    finalcolor = "white"
                if c < 2:
                    text += ('<td align="left" bgcolor="%s">%s' %
                             (finalcolor, b))
                else:
                    text += ('<td align="right" bgcolor="%s">%s' %
                             (finalcolor, b))
                text += (newtext)
                if len(b):
                    text += ("</td>")
                else:
                    text += ("</b></td>")
            if QTVERSION < '4.0.0':
                newtext = str(table.text(r, 0))
            else:
                item = table.item(r, 0)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
            if len(newtext):
                text += ("</b>")
            text += ("</tr>")
            text += ("\n")
        text += ("</table>")
        text += ("</nobr>")
        return text

    def getHTMLText(self, name=None):
        return self.gettext(name)

    if QTVERSION > '4.0.0':
        def getText(self, name=None):
            if name is None:
                name = self.current
            table = self.tables[name]
            text = ""
            ncols = table.columnCount()
            for l in range(ncols):
                text += (str(table.horizontalHeaderItem(l).text())) + "\t"
            text += ("\n")
            nrows = table.rowCount()
            for r in range(nrows):
                for c in range(ncols):
                    newtext = ""
                    if c != 4:
                        item = table.item(r, c)
                        if item is not None:
                            newtext = str(item.text())
                    else:
                        item = table.cellWidget(r, c)
                        if item is not None:
                            newtext = str(item.currentText())
                    text += (newtext) + "\t"
                text += ("\n")
            text += ("\n")
            return text


def test():
    import os
    import PyMca5
    from ..math import specfit
    from PyMca5 import PyMcaDataDir
    import numpy
    a = qt.QApplication([])
    a.lastWindowClosed.connect(a.quit)
    w = ParametersTab()
    w.show()

    y = numpy.loadtxt(os.path.join(PyMcaDataDir.PYMCA_DATA_DIR,
                            "XRFSpectrum.mca"))

    x = numpy.arange(len(y)) * 0.0502883 - 0.492773
    fit = specfit.Specfit()
    fit.setdata(x=x, y=y)
    fit.importfun(PyMca5.PyMcaMath.fitting.SpecfitFunctions.__file__)
    fit.settheory('Hypermet')
    fit.configure(Yscaling=1.,
                  WeightFlag=1,
                  PosFwhmFlag=1,
                  HeightAreaFlag=1,
                  FwhmPoints=16,
                  PositionFlag=1,
                  HypermetTails=1)
    fit.setbackground('Linear')
    # mcaresult = fit.mcafit(x=x, xmin=x[300], xmax=x[1000])
    # w.fillfrommca(mcaresult)
    fit.estimate()
    fit.startfit()
    w.fillfromfit(fit.fit_results, current='Fit')
    w.removeview(view='Region 1')
    a.exec_()

if __name__ == "__main__":
    test()
