# coding: utf-8
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
"""Collection of widgets used to build
:class:`silx.gui.fit.FitWidget.FitWidget`"""

from collections import OrderedDict

from silx.gui import qt
from silx.gui.fit.Parameters import Parameters

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

    The purpose of this widget, as it is used in
    :class:`silx.gui.fit.FitWidget.FitWidget`, is to offer an interface
    to quickly modify the main parameters prior to running a fitting:

      - select a fitting function through :attr:`FunComBox`
      - select a background function through :attr:`BkgComBox`
      - enable auto-scaling through :attr:`AutoScalingCheckBox`
      - enable automatic estimation of peaks' full-width at half maximum
        through :attr:`AutoFWHMCheckBox`
      - open a dialog for modifying advanced parameters through
        :attr:`ConfigureButton`
    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.setWindowTitle("FitConfigGUI")

        fitconfigguilayout = qt.QHBoxLayout(self)
        fitconfigguilayout.setContentsMargins(11, 11, 11, 11)
        fitconfigguilayout.setSpacing(6)

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

        # layout6 = qt.QGridLayout(None)
        # layout6.setContentsMargins(0, 0, 0, 0)
        # layout6.setSpacing(6)

        # self.WeightCheckBox = qt.QCheckBox(self)
        # self.WeightCheckBox.setText("Weight")

        # layout6.addWidget(self.WeightCheckBox, 0, 0)

        # self.MCACheckBox = qt.QCheckBox(self)
        # self.MCACheckBox.setText("MCA Mode")

        # layout6.addWidget(self.MCACheckBox, 1, 0)
        # layout9.addLayout(layout6)

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

        # self.PrintPushButton = qt.QPushButton(self)
        # self.PrintPushButton.setText("Print")
        #
        # layout5.addWidget(self.PrintPushButton, 1, 0)

        self.ConfigureButton = qt.QPushButton(self)
        self.ConfigureButton.setText("Configure")

        layout5.addWidget(self.ConfigureButton, 0, 0)
        layout9.addLayout(layout5)
        fitconfigguilayout.addLayout(layout9)

#
# class McaTable(qt.QTableWidget):
#     """This widget provides a table to input or display complex data,
#     such as fit configuration parameters together with estimation values
#     and final results of fitting.
#
#     """
#     sigMcaTableSignal = qt.pyqtSignal(object)
#
#     def __init__(self, labels=None, *args):
#         """
#
#         :param labels: List of labels used as column headers in the table
#             and
#         :param args: All arguments, besides ``label``, are used for
#             initializing the base class ``QTableWidget``
#         """
#         qt.QTableWidget.__init__(self, *args)
#         self.setRowCount(1)
#         self.setColumnCount(1)
#
#         self.code_options = ["FREE", "POSITIVE", "QUOTED",
#                              "FIXED", "FACTOR", "DELTA", "SUM", "IGNORE", "ADD", "SHOW"]
#
#         if labels is not None:
#             self.labels = labels
#         else:
#             self.labels = ['Position', 'Fit Area', 'MCA Area', 'Sigma',
#                            'Fwhm', 'Chisq', 'Region', 'XBegin', 'XEnd']
#
#         self.setColumnCount(len(self.labels))
#         for i, label in enumerate(self.labels):
#             item = self.horizontalHeaderItem(i)
#             if item is None:
#                 item = qt.QTableWidgetItem(label,
#                                            qt.QTableWidgetItem.Type)
#                 self.setHorizontalHeaderItem(i, item)
#             item.setText(label)
#             self.resizeColumnToContents(i)
#
#         self.regionlist = []
#         self.regiondict = {}
#
#         self.cellClicked[int, int].connect(self.__myslot)
#         self.itemSelectionChanged[()].connect(self.__myslot)
#
#     def fillfrommca(self, mcaresult, diag=1):
#         line0 = 0
#         region = 0
#         alreadyforced = False
#         for result in mcaresult:
#             region += 1
#             if result['chisq'] is not None:
#                 chisq = "%6.2f" % (result['chisq'])
#             else:
#                 chisq = "Fit Error"
#             if 1:
#                 xbegin = "%6g" % (result['xbegin'])
#                 xend = "%6g" % (result['xend'])
#                 fitlabel, fitpars, fitsigmas = self.__getfitpar(result)
#                 if QTVERSION < '4.0.0':
#                     qt.QHeader.setLabel(
#                         self.horizontalHeader(), 1, "Fit " + fitlabel)
#                 else:
#                     item = self.horizontalHeaderItem(1)
#                     item.setText("Fit " + fitlabel)
#                 i = 0
#                 for (pos, area, sigma, fwhm) in result['mca_areas']:
#                     line0 += 1
#                     if QTVERSION < '4.0.0':
#                         nlines = self.numRows()
#                         if line0 > nlines:
#                             self.setNumRows(line0)
#                     else:
#                         nlines = self.rowCount()
#                         if line0 > nlines:
#                             self.setRowCount(line0)
#                     line = line0 - 1
#                     pos = "%6g" % pos
#                     fitpar = "%6g" % fitpars[i]
#                     if fitlabel == 'Area':
#                         sigma = max(sigma, fitsigmas[i])
#                     areastr = "%6g" % area
#                     sigmastr = "%6.3g" % sigma
#                     fwhm = "%6g" % fwhm
#                     tregion = "%6g" % region
#                     fields = [pos, fitpar, areastr, sigmastr,
#                               fwhm, chisq, tregion, xbegin, xend]
#                     col = 0
#                     color = None
#                     if fitlabel == 'Area':
#                         if diag:
#                             if abs(fitpars[i] - area) > (3.0 * sigma):
#                                 color = qt.QColor(255, 182, 193)
#                     for field in fields:
#                         key = self.item(line, col)
#                         if key is None:
#                             key = qt.QTableWidgetItem(field)
#                             self.setItem(line, col, key)
#                         else:
#                             item.setText(field)
#                         if color is not None:
#                             # function introduced in Qt 4.2.0
#                             if QTVERSION >= '4.2.0':
#                                 item.setBackground(qt.QBrush(color))
#                         item.setFlags(qt.Qt.ItemIsSelectable |
#                                       qt.Qt.ItemIsEnabled)
#                         col += 1
#                     if color is not None:
#                         if not alreadyforced:
#                             alreadyforced = True
#                             self.scrollToItem(self.item(line, 0))
#                     i += 1
#
#         for i in range(len(self.labels)):
#             self.resizeColumnToContents(i)
#         ndict = {}
#         ndict['event'] = 'McaTableFilled'
#         self.sigMcaTableSignal.emit(ndict)
#
#     def __getfitpar(self, result):
#         if result['fitconfig']['fittheory'].find("Area") != -1:
#             fitlabel = 'Area'
#         elif result['fitconfig']['fittheory'].find("Hypermet") != -1:
#             fitlabel = 'Area'
#         else:
#             fitlabel = 'Height'
#         values = []
#         sigmavalues = []
#         for param in result['paramlist']:
#             if param['name'].find('ST_Area') != -1:
#                 # value and sigmavalue known via fitlabel
#                 values[-1] = value * (1.0 + param['fitresult'])
#                 # just an approximation
#                 sigmavalues[-1] = sigmavalue * (1.0 + param['fitresult'])
#             elif param['name'].find('LT_Area') != -1:
#                 pass
#             elif param['name'].find(fitlabel) != -1:
#                 value = param['fitresult']
#                 sigmavalue = param['sigma']
#                 values.append(value)
#                 sigmavalues.append(sigmavalue)
#         return fitlabel, values, sigmavalues
#
#     def __myslot(self, *var):
#         """Emit a signal with a dictionary containing the data in the active
#         row of the table.
#
#         The dictionary contains also special fields:
#
#            - 'event': string 'McaTableClicked' or 'McaTableRowHeaderClicked'
#            - 'row': 0-based row index (integer)
#            - 'col': 0-based column index (integer)
#            - 'labelslist': list of all labels (:attr:`labels`) which are
#              the keys to the remaining dictionary items.
#
#         The table values are stored as strings."""
#         ddict = {}
#         if len(var) == 0:
#             # selection changed event
#             # get the current selection
#             ddict['event'] = 'McaTableClicked'
#             row_idx = self.currentRow()
#         else:
#             # Header click
#             ddict['event'] = 'McaTableRowHeaderClicked'
#             row_idx = var[0]
#         ccol = self.currentColumn()
#         ddict['row'] = row_idx
#         ddict['col'] = ccol
#         ddict['labelslist'] = self.labels
#         if row_idx >= 0:
#             for col_idx, label in enumerate(self.labels):
#                 text = str(self.item(row_idx, col_idx).text())
#                 try:
#                     ddict[label] = float(text)
#                 except:
#                     ddict[label] = text
#         self.sigMcaTableSignal.emit(ddict)


class ParametersTab(qt.QTabWidget):
    """This widget provides tabs to display and modify fit parameters. Each
    tab contains a table with fit data such as parameter names, estimated
    values, fit constraints, and final fit results.

    The usual way to initialize the table is to fill it with the fit
    parameters from a :class:`silx.math.fit.fitmanager.FitManager` object, after
    the estimation process or after the final fit.

    In the following example we use a :class:`ParametersTab` to display the
    results of two separate fits::

        from silx.math.fit import fittheories
        from silx.math.fit import fitmanager
        from silx.math.fit import functions
        from silx.gui import qt
        import numpy

        a = qt.QApplication([])

        # Create synthetic data
        x = numpy.arange(1000)
        y1 = functions.sum_gauss(x, 100, 400, 100)

        fit = fitmanager.FitManager(x=x, y=y1)

        fitfuns = fittheories.FitTheories()
        fit.addtheory(theory="Gaussian",
                      function=functions.sum_gauss,
                      parameters=("height", "peak center", "fwhm"),
                      estimate=fitfuns.estimate_height_position_fwhm)
        fit.settheory('Gaussian')
        fit.configure(PositiveFwhmFlag=True,
                      PositiveHeightAreaFlag=True,
                      AutoFwhm=True,)

        # Fit
        fit.estimate()
        fit.startfit()

        # Show first fit result in a tab in our widget
        w = ParametersTab()
        w.show()
        w.fillfromfit(fit.fit_results, view='Gaussians')

        # new synthetic data
        y2 = functions.sum_splitgauss(x,
                                  100, 400, 100, 40,
                                  10, 600, 50, 500,
                                  80, 850, 10, 50)
        fit.setdata(x=x, y=y2)

        # Define new theory
        fit.addtheory(theory="Asymetric gaussian",
                      function=functions.sum_splitgauss,
                      parameters=("height", "peak center", "left fwhm", "right fwhm"),
                      estimate=fitfuns.estimate_splitgauss)
        fit.settheory('Asymetric gaussian')

        # Fit
        fit.estimate()
        fit.startfit()

        # Show first fit result in another tab in our widget
        w.fillfromfit(fit.fit_results, view='Asymetric gaussians')
        a.exec_()

    """
    sigMultiParametersSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, name="FitParameters"):
        """

        :param parent: Parent widget
        :param name: Widget title
        """
        qt.QTabWidget.__init__(self, parent)
        self.setWindowTitle(name)

        # the widgets in the notebook
        self.views = OrderedDict()

        # the widgets/tables themselves
        self.tables = {}
        """Dictionary of :class:`silx.gui.fit.parameters.Parameters` objects.
        These objects store fit results
        """
        # self.mcatable = None  # Fixme: probably not used
        self.setContentsMargins(10, 10, 10, 10)

    def setview(self, view, fitresults=None):
        """Add or update a table. Fill it with data from a fit

        :param view: Tab name to be added or updated.
        :param fitresults: Fit data to be added to the table
        """
        if view in self.tables.keys():
            table = self.tables[view]
        else:
            # create the parameters instance
            self.tables[view] = Parameters(self)
            table = self.tables[view]
            self.views[view] = table
            self.addTab(table, str(view))

        if fitresults is not None:
            table.fillfromfit(fitresults)

        self.setCurrentWidget(self.views[view])

    def renameview(self, oldname=None, newname=None):
        """Rename a view (tab)

        :param oldname: Name of the view to be renamed
        :param newname: New name of the view"""
        error = 1
        if newname is not None:
            if newname not in self.views.keys():
                if oldname in self.views.keys():
                    parameterlist = self.tables[oldname].getfitresults()
                    self.setview(view=newname, fitresults=parameterlist)
                    self.removeview(oldname)
                    error = 0
        return error

    def fillfromfit(self, fitparameterslist, view=None):
        """Update a view with data from a fit (alias for :meth:`setview`)

        :param view: Tab name to be added or updated.
        :param fitparameterslist: Fit data to be added to the table
        """
        self.setview(view=view, fitresults=fitparameterslist)

    def getfitresults(self, name):
        """Call :meth:`getfitresults` for the
        :class:`silx.gui.fit.parameters.Parameters` corresponding to the
        currently active table or to the named table (if ``name`` is not
        ``None``). This return a list of dictionaries in the format used by
        :class:`silx.math.fit.fitmanager.FitManager` to store fit parameter
        results.

        :param name: View name.
        """
        return self.tables[name].getfitresults()

    def removeview(self, name):
        """Remove a view by name.

        :param name: View name.
        """
        if name in self.views.keys():
            index = self.indexOf(self.tables[name])
            self.removeTab(index)
            index = self.indexOf(self.views[name])
            self.removeTab(index)
            del self.tables[name]
            del self.views[name]

    def removeallviews(self, keep=None):
        """Remove all views, except the one specified (argument
        ``keep``)

        :param keep: Name of the view to be kept."""
        for view in list(self.tables.keys()):
            if view != keep:
                self.removeview(view)

    # def fillfrommca(self, mcaresult):
    #     self.removeallviews()
    #     region = 0
    #
    #     for result in mcaresult:
    #         region = region + 1
    #         self.fillfromfit(result['paramlist'],
    #                          view='Region %d' % region)
    #     name = 'MCA'
    #     if name in self.tables:
    #         table = self.tables[name]
    #     else:
    #         self.tables[name] = McaTable(self)
    #         table = self.tables[name]
    #         self.views[name] = table
    #         self.addTab(table, str(name))
    #         table.sigMcaTableSignal.connect(self.__forward)
    #     table.fillfrommca(mcaresult)
    #     self.setview(name=name)
    #     return
    #
    # def __forward(self, ddict):
    #     self.sigMultiParametersSignal.emit(ddict)

    def getHTMLtext(self, name):
        """Return the table data as HTML

        :param name: View name."""
        table = self.tables[name]
        lemon = ("#%x%x%x" % (255, 250, 205)).upper()
        hcolor = ("#%x%x%x" % (230, 240, 249)).upper()
        text = ""
        text += "<nobr>"
        text += "<table>"
        text += "<tr>"
        ncols = table.columnCount()
        for l in range(ncols):
            text += ('<td align="left" bgcolor="%s"><b>' % hcolor)
            if QTVERSION < '4.0.0':
                text += (str(table.horizontalHeader().label(l)))
            else:
                text += (str(table.horizontalHeaderItem(l).text()))
            text += "</b></td>"
        text += "</tr>"
        nrows = table.rowCount()
        for r in range(nrows):
            text += "<tr>"
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
                text += newtext
                if len(b):
                    text += "</td>"
                else:
                    text += "</b></td>"
            item = table.item(r, 0)
            newtext = ""
            if item is not None:
                newtext = str(item.text())
            if len(newtext):
                text += "</b>"
            text += "</tr>"
            text += "\n"
        text += "</table>"
        text += "</nobr>"
        return text

    def gettext(self, name):
        """Return the table data as CSV formatted text, using tabulation
        characters as separators.

        :param name: View name."""
        table = self.tables[name]
        text = ""
        ncols = table.columnCount()
        for l in range(ncols):
            text += (str(table.horizontalHeaderItem(l).text())) + "\t"
        text += "\n"
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
                text += newtext + "\t"
            text += "\n"
        text += "\n"
        return text


def test():
    from silx.math.fit import fittheories
    from silx.math.fit import fitmanager
    from silx.math.fit import functions
    from silx.gui import qt
    from silx.gui.plot.PlotWindow import PlotWindow
    import numpy

    a = qt.QApplication([])

    x = numpy.arange(1000)
    y1 = functions.sum_gauss(x, 100, 400, 100)

    fit = fitmanager.FitManager(x=x, y=y1)

    fitfuns = fittheories.FitTheories()
    fit.addtheory(theory="Gaussian",
                  function=functions.sum_gauss,
                  parameters=("height", "peak center", "fwhm"),
                  estimate=fitfuns.estimate_height_position_fwhm)
    fit.settheory('Gaussian')
    fit.configure(PositiveFwhmFlag=True,
                  PositiveHeightAreaFlag=True,
                  AutoFwhm=True,)

    # Fit
    fit.estimate()
    fit.startfit()

    w = ParametersTab()
    w.show()
    w.fillfromfit(fit.fit_results, view='Gaussians')

    y2 = functions.sum_splitgauss(x,
                                  100, 400, 100, 40,
                                  10, 600, 50, 500,
                                  80, 850, 10, 50)
    fit.setdata(x=x, y=y2)

    # Define new theory
    fit.addtheory(theory="Asymetric gaussian",
                  function=functions.sum_splitgauss,
                  parameters=("height", "peak center", "left fwhm", "right fwhm"),
                  estimate=fitfuns.estimate_splitgauss)
    fit.settheory('Asymetric gaussian')

    # Fit
    fit.estimate()
    fit.startfit()

    w.fillfromfit(fit.fit_results, view='Asymetric gaussians')

    # Plot
    pw = PlotWindow(control=True)
    pw.addCurve(x, y1, "Gaussians")
    pw.addCurve(x, y2, "Asymetric gaussians")
    pw.show()

    a.exec_()


if __name__ == "__main__":
    test()
